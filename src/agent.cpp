#include "llm/agent.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace llm {
namespace {

CompletionResult ProviderThrew(const std::string& message) {
  CompletionResult completion;
  completion.status = CompletionStatus::ProviderError;
  ModelError error;
  error.kind = ErrorKind::Unknown;
  error.message = message;
  completion.error = std::move(error);
  return completion;
}

/// Providers are supposed to report failures in the result, but a custom one
/// may throw: nothing may let that escape into a game loop.
CompletionResult CallProvider(const std::shared_ptr<IModelProvider>& provider,
                              const ModelRequest& request) {
  if (!provider) {
    return ProviderThrew("no provider is configured");
  }
  try {
    return provider->Complete(request);
  } catch (const std::exception& error) {
    return ProviderThrew(std::string("provider threw: ") + error.what());
  } catch (...) {
    return ProviderThrew("provider threw an unknown error");
  }
}

DecisionResult Refused(DecisionStatus status,
                       const std::string& revision,
                       const std::string& message) {
  DecisionResult result;
  result.status = status;
  result.context_revision = revision;
  ModelError error;
  error.kind = ErrorKind::InvalidRequest;
  error.message = message;
  result.completion.status = CompletionStatus::ProviderError;
  result.completion.error = std::move(error);
  return result;
}

/// A token whose death or closure invalidates already-prepared commands.
///
/// The Agent keeps the shared_ptr while a decision is pending; a
/// PreparedCommand only holds a weak_ptr, so discarding or recording the
/// decision (which destroys the token) or closing the agent makes the command
/// refuse to run.
struct Liveness {
  std::atomic<bool> closed{false};
};

}  // namespace

/// Everything an Agent owns.
///
/// There is deliberately no mutex: an Agent has an owner thread, and only that
/// thread ever touches this object. The model request runs elsewhere, but that
/// thread only calls the provider and returns a CompletionResult through a
/// future; it never reads or writes Agent state.
struct Agent::Impl {
  std::shared_ptr<IModelProvider> provider;
  std::string id;
  std::string model;
  std::string system_prompt;
  CommandRegistry registry;
  std::optional<std::int32_t> max_output_tokens;
  std::optional<std::size_t> max_command_calls;
  std::shared_ptr<ILogger> logger;
  /// The SDK's own batch validator, applied to every provider result so a
  /// provider that reports Success without validating cannot smuggle an
  /// invalid batch into the pending decision.
  std::shared_ptr<const CommandValidator> validator = DefaultValidator();

  Context context;
  bool closed = false;
  std::uint64_t next_decision_id = 1;
  /// Diagnostics id for each accepted decision attempt; it travels with the
  /// request so provider events can be attributed to it.
  std::uint64_t next_request_id = 1;

  /// The single decision awaiting command results, if any.
  struct Pending {
    std::uint64_t decision_id = 0;
    std::uint64_t request_id = 0;
    std::string context_revision;
    /// Registry version the decision was taken with; preparing against a
    /// changed registry is refused.
    std::uint64_t registry_version = 0;
    /// The framing the application supplied when the decision started.
    std::string decision_context;
    Message assistant;
    /// One call and the one-shot token every prepared batch of it shares.
    struct Call {
      CommandCall call;
      std::shared_ptr<CommandExecution> execution =
          std::make_shared<CommandExecution>();
    };
    std::vector<Call> calls;  // in call order
    std::shared_ptr<Liveness> live = std::make_shared<Liveness>();
  };
  std::optional<Pending> pending;

  /// The single model request currently running, if any.
  struct InFlight {
    std::string context_revision;
    std::string decision_context;
    std::uint64_t request_id = 0;
    /// The request snapshot, kept so the result can be validated against the
    /// exact definitions the model was offered.
    std::shared_ptr<const ModelRequest> request;
    std::future<CompletionResult> future;
  };
  std::optional<InFlight> in_flight;

  explicit Impl(ContextConfig config) : context(std::move(config)) {}

  /// The request this agent would send for the current context: the rendered
  /// history plus this agent's system prompt, commands and model parameters.
  ModelRequest BuildRequest(std::uint64_t request_id,
                            bool* exceeds_budget) const {
    ModelRequest request;
    request.model = model;

    const std::vector<CommandDefinition> commands = registry.Definitions();
    // The token budget covers everything the provider will receive: the system
    // prompt, the command schemas and the rendered transcript.
    std::size_t fixed_tokens =
        system_prompt.empty() ? 0 : EstimateTokens(system_prompt);
    for (const CommandDefinition& command : commands) {
      fixed_tokens += EstimateTokens(command.name) +
                      EstimateTokens(command.description) +
                      EstimateTokens(command.parameters_schema_json);
    }

    RenderBudget budget;
    budget.fixed_tokens = fixed_tokens;
    const std::size_t configured = context.config().max_input_tokens;
    if (configured > 0) {
      const std::size_t reserved = context.config().reserved_output_tokens;
      budget.max_input_tokens =
          configured > reserved ? configured - reserved : 1;
    }

    const RenderedContext rendered = context.Render(budget);
    *exceeds_budget = rendered.exceeds_budget;

    request.messages.reserve(rendered.messages.size() + 1);
    if (!system_prompt.empty()) {
      request.messages.push_back(Message::System(system_prompt));
    }
    request.messages.insert(request.messages.end(), rendered.messages.begin(),
                            rendered.messages.end());

    request.commands = commands;
    request.command_choice = CommandChoice::Required();
    request.max_tokens = max_output_tokens;
    request.max_command_calls = max_command_calls;
    request.log = LogContext{id, request_id};
    return request;
  }

  /// Turns one finished model call into a DecisionResult, staging a pending
  /// decision when the completion succeeded. Runs on the owner thread.
  DecisionResult FinishDecision(const InFlight& flight,
                                CompletionResult completion) {
    DecisionResult result;
    const std::shared_ptr<ILogger> log_sink = logger;
    const LogContext log_context{id, flight.request_id};
    result.context_revision = flight.context_revision;
    result.status = DecisionStatus::Completed;
    result.completion = std::move(completion);

    if (closed) {
      result = Refused(DecisionStatus::Closed, flight.context_revision,
                       "agent was closed while the decision was in flight");
    } else if (result.completion.IsSuccess()) {
      // The SDK validates every provider's output here, not only the OpenAI
      // one: batch size, known commands, unique ids, an argument object and
      // the command's schema. A provider that reported Success without
      // checking cannot reach a callback.
      const ValidationReport report =
          validator->Validate(*flight.request, result.completion.command_calls);
      if (!report.ok) {
        const bool schema_issue = report.HasSchemaIssues();
        result.completion.status = schema_issue
                                       ? CompletionStatus::Unsupported
                                       : CompletionStatus::InvalidCommandCalls;
        ModelError error;
        error.kind =
            schema_issue ? ErrorKind::Unsupported : ErrorKind::InvalidRequest;
        error.message = report.Summary();
        result.completion.error = std::move(error);
      } else {
        Pending staged;
        staged.decision_id = next_decision_id++;
        staged.request_id = flight.request_id;
        staged.context_revision = flight.context_revision;
        staged.registry_version = registry.version();
        staged.decision_context = flight.decision_context;
        staged.assistant = result.completion.ToAssistantMessage();
        staged.calls.reserve(result.completion.command_calls.size());
        for (const CommandCall& call : result.completion.command_calls) {
          Pending::Call entry;
          entry.call = call;
          staged.calls.push_back(std::move(entry));
        }
        result.decision_id = staged.decision_id;
        pending = std::move(staged);
      }
    }

    EmitLog(
        log_sink,
        result.completion.IsSuccess() ? LogLevel::Info : LogLevel::Warning,
        log_event::kDecisionCompleted,
        "status=" + ToString(result.completion.status) + ", commands=" +
            std::to_string(result.completion.command_calls.size()) +
            ", tokens=" + std::to_string(result.completion.usage.total_tokens) +
            ", attempts=" + std::to_string(result.completion.http_attempts) +
            ", decision_id=" + std::to_string(result.decision_id) +
            ", model=" + result.completion.model,
        log_context);
    return result;
  }
};

Agent::Agent(AgentConfig config)
    : impl_(std::make_unique<Impl>(std::move(config.context))) {
  impl_->id = std::move(config.id);
  impl_->model = std::move(config.model);
  impl_->system_prompt = std::move(config.system_prompt);
  impl_->max_output_tokens = config.max_output_tokens;
  impl_->max_command_calls = config.max_command_calls;
  impl_->logger = std::move(config.logger);

  EmitLog(impl_->logger, LogLevel::Info, log_event::kSessionCreated,
          "model=" + impl_->model + ", max_messages=" +
              std::to_string(impl_->context.config().max_messages) +
              ", session=" + impl_->id,
          LogContext{impl_->id, 0});
}

Agent::Agent(std::shared_ptr<IModelProvider> provider, AgentConfig config)
    : Agent(std::move(config)) {
  SetProvider(std::move(provider));
}

Agent::~Agent() {
  // The owner is destroying the agent: invalidate prepared commands and let the
  // in-flight future's destructor wait for a running model call. A request that
  // is already on the wire is never abandoned.
  impl_->closed = true;
  if (impl_->pending.has_value()) {
    impl_->pending->live->closed = true;
  }
}

ConfigOutcome Agent::SetProvider(std::shared_ptr<IModelProvider> provider) {
  if (!provider) {
    return {ConfigStatus::InvalidProvider, "provider must not be null"};
  }
  if (impl_->closed) {
    return {ConfigStatus::AgentClosed, "agent is closed"};
  }
  if (impl_->in_flight.has_value() || impl_->pending.has_value()) {
    return {ConfigStatus::AgentBusy,
            "cannot change the provider while a decision is outstanding"};
  }
  impl_->provider = std::move(provider);
  return {};
}

ConfigOutcome Agent::SetModel(std::string model) {
  if (model.empty()) {
    return {ConfigStatus::InvalidModel, "model must not be empty"};
  }
  if (impl_->closed) {
    return {ConfigStatus::AgentClosed, "agent is closed"};
  }
  impl_->model = std::move(model);
  return {};
}

ConfigOutcome Agent::SetSystemPrompt(std::string prompt) {
  if (impl_->closed) {
    return {ConfigStatus::AgentClosed, "agent is closed"};
  }
  impl_->system_prompt = std::move(prompt);
  return {};
}

std::string Agent::model() const {
  return impl_->model;
}

CommandRegistrationResult Agent::RegisterCommandErased(
    RegisteredCommand command) {
  Impl& impl = *impl_;
  const std::string name = command.definition.name;

  CommandRegistrationResult result;
  if (impl.closed) {
    result = {RegistrationStatus::AgentClosed, "agent is closed"};
  } else if (impl.in_flight.has_value() || impl.pending.has_value()) {
    result = {RegistrationStatus::AgentBusy,
              "commands cannot change while a decision is in flight or "
              "awaiting results"};
  } else {
    result = impl.registry.Add(std::move(command));
  }

  if (result.ok()) {
    EmitLog(impl.logger, LogLevel::Info, log_event::kCommandRegistered,
            "command=" + name +
                ", commands=" + std::to_string(impl.registry.size()),
            LogContext{impl.id, 0});
  } else {
    EmitLog(impl.logger, LogLevel::Warning,
            log_event::kCommandRegistrationRejected, "command=" + name,
            LogContext{impl.id, 0});
  }
  return result;
}

CommandRegistrationResult Agent::UnregisterCommand(std::string_view name) {
  Impl& impl = *impl_;
  if (impl.closed) {
    return {RegistrationStatus::AgentClosed, "agent is closed"};
  }
  if (impl.in_flight.has_value() || impl.pending.has_value()) {
    return {RegistrationStatus::AgentBusy,
            "commands cannot change while a decision is in flight or awaiting "
            "results"};
  }
  if (!impl.registry.Remove(name)) {
    return {RegistrationStatus::NotRegistered,
            "no command named '" + std::string(name) + "' is registered"};
  }
  return {};
}

CommandRegistrationResult Agent::ClearCommands() {
  Impl& impl = *impl_;
  if (impl.closed) {
    return {RegistrationStatus::AgentClosed, "agent is closed"};
  }
  if (impl.in_flight.has_value() || impl.pending.has_value()) {
    return {RegistrationStatus::AgentBusy,
            "commands cannot change while a decision is in flight or awaiting "
            "results"};
  }
  impl.registry.Clear();
  return {};
}

std::vector<CommandDefinition> Agent::Commands() const {
  return impl_->registry.Definitions();
}

void Agent::UpdateContext(ContextUpdate update) {
  impl_->context.SetContextUpdate(std::move(update));
}

std::string Agent::current_revision() const {
  return impl_->context.context_revision();
}

Context Agent::ContextSnapshot() const {
  return impl_->context;
}

StartOutcome Agent::DecideAsync() {
  Impl& impl = *impl_;
  const auto reject = [&](StartStatus status, std::string message) {
    EmitLog(impl.logger, LogLevel::Warning, log_event::kDecisionRejected,
            message, LogContext{impl.id, 0});
    return StartOutcome{status, impl.context.context_revision(),
                        std::move(message)};
  };

  if (impl.closed) {
    return reject(StartStatus::Closed, "agent is closed");
  }
  if (!impl.provider || impl.model.empty()) {
    return reject(StartStatus::NotConfigured,
                  !impl.provider ? "agent has no provider configured"
                                 : "agent has no model configured");
  }
  if (std::string config_error; !impl.context.config().IsValid(&config_error)) {
    return reject(StartStatus::InvalidConfig, std::move(config_error));
  }
  if (impl.registry.size() == 0) {
    return reject(StartStatus::NoCommands,
                  "no commands are registered; register at least one before "
                  "asking for a decision");
  }
  if (impl.in_flight.has_value() || impl.pending.has_value()) {
    return reject(StartStatus::Busy,
                  impl.in_flight.has_value()
                      ? "a decision is already in flight"
                      : "a decision is still awaiting command results; call "
                        "RecordCommandResults or DiscardDecision first");
  }

  const std::uint64_t request_id = impl.next_request_id++;
  bool exceeds_budget = false;
  auto request = std::make_shared<const ModelRequest>(
      impl.BuildRequest(request_id, &exceeds_budget));
  if (exceeds_budget) {
    return reject(StartStatus::ContextTooLarge,
                  "the rendered context, system prompt and command schemas "
                  "exceed the configured input token budget");
  }
  // A schema the SDK validator cannot enforce must not be sent: the provider
  // would either reject it or be credited with a guarantee it never made.
  if (impl.validator != nullptr) {
    const ValidationReport schemas =
        impl.validator->ValidateCommandSchemas(*request);
    if (!schemas.ok) {
      return reject(StartStatus::Failed,
                    "the request uses a command schema that cannot be "
                    "enforced: " +
                        schemas.Summary());
    }
  }

  const std::string revision = impl.context.context_revision();
  const std::string decision_context = impl.context.decision_context();
  const std::size_t command_count = request->commands.size();
  const std::size_t message_count = request->messages.size();
  std::shared_ptr<IModelProvider> provider = impl.provider;

  try {
    Impl::InFlight flight;
    flight.context_revision = revision;
    flight.decision_context = decision_context;
    flight.request_id = request_id;
    flight.request = request;
    // The model thread only sees the provider and the request: it cannot touch
    // any Agent state.
    flight.future = std::async(std::launch::async, [provider, request]() {
      return CallProvider(provider, *request);
    });
    impl.in_flight = std::move(flight);
  } catch (const std::exception& error) {
    return reject(
        StartStatus::Failed,
        std::string("could not start the model call: ") + error.what());
  } catch (...) {
    return reject(StartStatus::Failed, "could not start the model call");
  }

  EmitLog(impl.logger, LogLevel::Debug, log_event::kDecisionStarted,
          "rev=" + revision + ", commands=" + std::to_string(command_count) +
              ", messages=" + std::to_string(message_count),
          LogContext{impl.id, request_id});
  return {StartStatus::Started, revision, {}};
}

std::optional<DecisionResult> Agent::PollDecision() {
  Impl& impl = *impl_;
  if (!impl.in_flight.has_value()) {
    return std::nullopt;
  }
  if (impl.in_flight->future.wait_for(std::chrono::milliseconds(0)) !=
      std::future_status::ready) {
    return std::nullopt;
  }

  Impl::InFlight flight = std::move(*impl.in_flight);
  impl.in_flight.reset();

  CompletionResult completion;
  try {
    completion = flight.future.get();
  } catch (const std::exception& error) {
    completion =
        ProviderThrew(std::string("model call failed: ") + error.what());
  } catch (...) {
    completion = ProviderThrew("model call failed");
  }
  return impl.FinishDecision(flight, std::move(completion));
}

PrepareResult Agent::PrepareCommands(std::uint64_t decision_id) {
  Impl& impl = *impl_;
  if (impl.closed) {
    return {PrepareStatus::Closed, {}, "agent is closed"};
  }
  if (!impl.pending.has_value() || impl.pending->decision_id != decision_id) {
    return {PrepareStatus::UnknownDecision,
            {},
            "no decision with that id is awaiting results"};
  }
  Impl::Pending& pending = *impl.pending;
  if (pending.calls.empty()) {
    return {PrepareStatus::NoCommands,
            {},
            "the decision contains no command calls"};
  }
  if (pending.registry_version != impl.registry.version()) {
    return {PrepareStatus::RegistryChanged,
            {},
            "the registered commands changed after this decision was taken"};
  }

  const std::weak_ptr<Liveness> live = pending.live;
  PrepareResult result;
  result.commands.reserve(pending.calls.size());

  for (const Impl::Pending::Call& entry : pending.calls) {
    const CommandCall& call = entry.call;
    const RegisteredCommand* command = impl.registry.Find(call.name);
    if (command == nullptr) {
      result.status = PrepareStatus::InvalidArguments;
      result.error = "command '" + call.name + "' is not registered";
      result.commands.clear();
      return result;
    }
    CommandBinding binding = command->bind(call.arguments_json);
    if (!binding.ok) {
      result.status = PrepareStatus::InvalidArguments;
      result.error = "command '" + call.name + "': " + binding.error;
      result.commands.clear();
      return result;
    }
    result.commands.emplace_back(
        call.id, call.name, std::move(binding.invoke), entry.execution,
        [live]() {
          const std::shared_ptr<Liveness> token = live.lock();
          return token != nullptr && !token->closed.load();
        });
  }

  result.status = PrepareStatus::Ok;
  return result;
}

RecordOutcome Agent::RecordCommandResults(std::uint64_t decision_id,
                                          std::vector<CommandResult> results) {
  Impl& impl = *impl_;
  std::uint64_t request_id = 0;
  const auto reject = [&](const char* reason) {
    EmitLog(impl.logger, LogLevel::Warning, log_event::kDecisionRecordRejected,
            std::string("decision_id=") + std::to_string(decision_id) + ", " +
                reason,
            LogContext{impl.id, request_id});
  };

  if (impl.closed) {
    reject("agent is closed");
    return RecordOutcome::Closed;
  }
  if (!impl.pending.has_value() || impl.pending->decision_id != decision_id) {
    reject("no decision with that id is awaiting results");
    return RecordOutcome::UnknownDecision;
  }
  Impl::Pending& pending = *impl.pending;
  request_id = pending.request_id;

  // Every command call needs exactly one matching result, and nothing else: a
  // decision is recorded as a whole or not at all.
  if (results.size() != pending.calls.size()) {
    reject("results do not cover every command call exactly once");
    return RecordOutcome::IncompleteResults;
  }
  // A malformed result would be embedded verbatim into the transcript and
  // poison every later request, so it is refused at the boundary.
  for (const CommandResult& candidate : results) {
    if (!candidate.outcome.content_json.empty() &&
        !IsValidJson(candidate.outcome.content_json)) {
      reject("a command result contains content that is not valid JSON");
      return RecordOutcome::InvalidResultJson;
    }
  }
  std::vector<Message> result_messages;
  result_messages.reserve(pending.calls.size());
  for (const Impl::Pending::Call& entry : pending.calls) {
    const CommandCall& call = entry.call;
    const CommandResult* match = nullptr;
    std::size_t matches = 0;
    for (const CommandResult& candidate : results) {
      if (candidate.command_call_id == call.id) {
        match = &candidate;
        ++matches;
      }
    }
    if (matches != 1) {
      reject("results do not cover every command call exactly once");
      return RecordOutcome::IncompleteResults;
    }
    if (match->name != call.name) {
      reject("a result names a different command than the call did");
      return RecordOutcome::NameMismatch;
    }
    result_messages.push_back(Message::CommandResultMessage(*match));
  }

  DecisionRecord record;
  record.decision_id = pending.decision_id;
  record.context_revision = pending.context_revision;
  record.decision_context = pending.decision_context;
  const std::string record_revision = pending.context_revision;
  record.assistant = pending.assistant;
  record.command_results = std::move(result_messages);

  // The decision is committed; any prepared command for it is now stale.
  pending.live->closed = true;
  impl.context.AppendDecision(std::move(record));
  impl.pending.reset();

  EmitLog(impl.logger, LogLevel::Info, log_event::kDecisionRecorded,
          "decision_id=" + std::to_string(decision_id) + ", results=" +
              std::to_string(results.size()) + ", rev=" + record_revision,
          LogContext{impl.id, request_id});
  return RecordOutcome::Recorded;
}

bool Agent::DiscardDecision(std::uint64_t decision_id) {
  Impl& impl = *impl_;
  if (!impl.pending.has_value() || impl.pending->decision_id != decision_id) {
    return false;
  }
  const std::uint64_t request_id = impl.pending->request_id;
  impl.pending->live->closed = true;
  impl.pending.reset();

  EmitLog(impl.logger, LogLevel::Info, log_event::kDecisionDiscarded,
          "decision_id=" + std::to_string(decision_id),
          LogContext{impl.id, request_id});
  return true;
}

bool Agent::IsBusy() const {
  return impl_->in_flight.has_value() || impl_->pending.has_value();
}

bool Agent::IsClosed() const {
  return impl_->closed;
}

std::size_t Agent::InFlightDecisionCount() const {
  return impl_->in_flight.has_value() ? 1u : 0u;
}

std::size_t Agent::PendingDecisionCount() const {
  return impl_->pending.has_value() ? 1u : 0u;
}

bool Agent::WaitForIdle(std::chrono::milliseconds timeout) const {
  const Impl& impl = *impl_;
  if (!impl.in_flight.has_value()) {
    return true;
  }
  return impl.in_flight->future.wait_for(timeout) == std::future_status::ready;
}

void Agent::Close() {
  Impl& impl = *impl_;
  const bool was_open = !impl.closed;
  impl.closed = true;
  if (impl.pending.has_value()) {
    impl.pending->live->closed = true;
  }
  if (was_open) {
    EmitLog(impl.logger, LogLevel::Info, log_event::kSessionClosed,
            "session=" + impl.id, LogContext{impl.id, 0});
  }
}

}  // namespace llm
