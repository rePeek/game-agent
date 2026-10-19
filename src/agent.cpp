#include "llm/agent.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
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

}  // namespace

/// Everything an Agent owns.
///
/// The worker thread only ever touches this object (never the Agent), and the
/// destructor joins it before it is destroyed, so an in-flight decision cannot
/// reach a dead agent.
struct Agent::Impl {
  mutable std::mutex mutex;
  mutable std::condition_variable
      work_cv;  // a task arrived, or we are stopping
  mutable std::condition_variable idle_cv;  // nothing is in flight any more

  std::shared_ptr<IModelProvider> provider;
  std::string id;
  std::string model;
  std::string system_prompt;
  std::vector<CommandDefinition> commands;
  std::optional<std::int32_t> max_output_tokens;
  std::optional<std::size_t> max_command_calls;
  std::shared_ptr<ILogger> logger;

  Context context;
  bool closed = false;
  std::uint64_t next_decision_id = 1;
  /// Diagnostics id for each accepted decision attempt; it travels with the
  /// request so provider events can be attributed to it.
  std::uint64_t next_request_id = 1;

  /// Revision of the decision currently being asked about, if any.
  std::optional<std::string> in_flight_revision;

  /// The single decision awaiting command results, if any.
  struct Pending {
    std::uint64_t decision_id = 0;
    /// Request id of the attempt that produced this decision, so recording and
    /// discarding can be correlated with the events of that attempt.
    std::uint64_t request_id = 0;
    std::string context_revision;
    Message assistant;
    std::vector<std::string> command_call_ids;  // in call order
  };
  std::vector<Pending> pending;  // at most one in this version

  /// One accepted decision: the request is snapshotted on the submitting
  /// thread, so a later context update cannot change what is being asked.
  struct Attempt {
    bool started = false;
    std::string context_revision;
    std::uint64_t request_id = 0;
    /// Provider snapshotted when the attempt starts, so swapping the
    /// provider while a decision is in flight cannot change what runs.
    std::shared_ptr<IModelProvider> provider;
    ModelRequest request;
    DecisionResult refusal;  // valid when !started
  };

  // Worker plumbing.
  std::thread worker;
  std::deque<std::function<void()>> tasks;
  bool stopping = false;

  explicit Impl(ContextConfig config) : context(std::move(config)) {}

  const Pending* FindPendingLocked(std::uint64_t decision_id) const {
    for (const Pending& candidate : pending) {
      if (candidate.decision_id == decision_id) {
        return &candidate;
      }
    }
    return nullptr;
  }

  /// The request this agent would send for the current context: the rendered
  /// history plus this agent's system prompt, commands and model parameters.
  ModelRequest BuildRequestLocked(std::uint64_t request_id) const {
    ModelRequest request;
    request.model = model;

    const RenderedContext rendered = context.Render();
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

  /// Reserves the agent for one decision and snapshots the request, or returns
  /// a refusal (Closed/Busy) without touching anything. The context is whatever
  /// UpdateContext() recorded last.
  Attempt BeginAttempt() {
    Attempt attempt;
    const std::shared_ptr<ILogger> log_sink = logger;
    const std::string agent_id = id;

    {
      std::lock_guard<std::mutex> lock(mutex);

      if (closed) {
        attempt.refusal =
            Refused(DecisionStatus::Closed, context.context_revision(),
                    "agent is closed");
      } else if (in_flight_revision.has_value() || !pending.empty()) {
        attempt.refusal =
            Refused(DecisionStatus::Busy, context.context_revision(),
                    in_flight_revision.has_value()
                        ? "a decision is already in flight"
                        : "a decision is still awaiting command results; call "
                          "RecordCommandResults or DiscardDecision first");
      } else {
        attempt.started = true;
        attempt.context_revision = context.context_revision();
        attempt.request_id = next_request_id++;
        attempt.request = BuildRequestLocked(attempt.request_id);
        attempt.provider = provider;
        in_flight_revision = attempt.context_revision;
      }
    }

    // Emitted outside the lock: a logger may call back into the agent.
    if (attempt.started) {
      EmitLog(
          log_sink, LogLevel::Debug, log_event::kDecisionStarted,
          "rev=" + attempt.context_revision +
              ", commands=" + std::to_string(attempt.request.commands.size()) +
              ", messages=" + std::to_string(attempt.request.messages.size()),
          attempt.request.log);
    } else {
      EmitLog(log_sink, LogLevel::Warning, log_event::kDecisionRejected,
              attempt.refusal.completion.ErrorMessage(),
              LogContext{agent_id, 0});
    }
    return attempt;
  }

  /// Clears the in-flight mark and stages the decision when it succeeded.
  DecisionResult FinishAttempt(const Attempt& attempt,
                               CompletionResult completion) {
    DecisionResult result;
    const std::shared_ptr<ILogger> log_sink = logger;
    const LogContext log_context{id, attempt.request_id};
    result.context_revision = attempt.context_revision;
    result.status = DecisionStatus::Completed;
    result.completion = std::move(completion);

    {
      std::lock_guard<std::mutex> lock(mutex);
      in_flight_revision.reset();
      idle_cv.notify_all();

      if (closed) {
        result = Refused(DecisionStatus::Closed, attempt.context_revision,
                         "agent was closed while the decision was in flight");
      } else if (result.completion.IsSuccess()) {
        Pending staged;
        staged.decision_id = next_decision_id++;
        staged.request_id = attempt.request_id;
        staged.context_revision = attempt.context_revision;
        staged.assistant = result.completion.ToAssistantMessage();
        for (const CommandCall& call : result.completion.command_calls) {
          staged.command_call_ids.push_back(call.id);
        }
        result.decision_id = staged.decision_id;
        pending.push_back(std::move(staged));
      }
    }

    // Emitted with the lock released: a logger may call back into the agent.
    EmitLog(
        log_sink,
        result.completion.IsSuccess() ? LogLevel::Info : LogLevel::Warning,
        log_event::kDecisionCompleted,
        "status=" + ToString(result.completion.status) +
            ", commands=" + std::to_string(result.completion.command_calls.size()) +
            ", tokens=" + std::to_string(result.completion.usage.total_tokens) +
            ", attempts=" + std::to_string(result.completion.http_attempts) +
            ", decision_id=" + std::to_string(result.decision_id) +
            ", model=" + result.completion.model,
        log_context);
    return result;
  }

  /// Last-resort completion for a task that failed before it could stage.
  /// The in-flight mark is cleared and the caller still receives a decision,
  /// so a worker failure can never leave the agent permanently Busy.
  DecisionResult AbortAttempt(const Attempt& attempt, std::string message) {
    DecisionResult result;
    result.context_revision = attempt.context_revision;
    result.status = DecisionStatus::Completed;
    result.completion = ProviderThrew(message);
    {
      std::lock_guard<std::mutex> lock(mutex);
      in_flight_revision.reset();
      idle_cv.notify_all();
    }
    // The failure text is the worker's own, never model output.
    EmitLog(logger, LogLevel::Error, log_event::kDecisionCompleted,
            "status=provider error, decision_id=0",
            LogContext{id, attempt.request_id});
    return result;
  }

  /// Runs queued decisions until the agent stops; accepted decisions are
  /// drained before the loop returns.
  void Run() {
    for (;;) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex);
        work_cv.wait(lock, [this] { return stopping || !tasks.empty(); });
        if (tasks.empty()) {
          if (stopping) {
            return;  // drained: accepted decisions have been answered
          }
          continue;
        }
        task = std::move(tasks.front());
        tasks.pop_front();
      }
      try {
        task();
      } catch (...) {
        // A task must never escape and kill the worker; clear the in-flight
        // mark so a failure cannot leave the agent permanently Busy.
        std::lock_guard<std::mutex> lock(mutex);
        in_flight_revision.reset();
        idle_cv.notify_all();
      }
    }
  }
};

Agent::Agent(std::shared_ptr<IModelProvider> provider, AgentConfig config)
    : impl_(std::make_unique<Impl>(std::move(config.context))) {
  impl_->provider = std::move(provider);
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

Agent::~Agent() {
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->closed = true;
    impl_->stopping = true;
  }
  impl_->work_cv.notify_all();
  if (impl_->worker.joinable()) {
    impl_->worker.join();
  }
}

void Agent::SetSystemPrompt(std::string prompt) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->system_prompt = std::move(prompt);
}

void Agent::SetCommands(std::vector<CommandDefinition> commands) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->commands = std::move(commands);
}

void Agent::SetModel(std::string model) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->model = std::move(model);
}

std::string Agent::model() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->model;
}

std::vector<CommandDefinition> Agent::commands() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->commands;
}

void Agent::UpdateContext(ContextUpdate update) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->context.SetContextUpdate(std::move(update));
}

std::string Agent::current_revision() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->context.context_revision();
}

Context Agent::ContextSnapshot() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->context;
}

std::future<DecisionResult> Agent::DecideAsync() {
  // A shared promise: the queued task must be copy-constructible, and
  // std::promise is not.
  auto promise = std::make_shared<std::promise<DecisionResult>>();
  std::future<DecisionResult> future = promise->get_future();

  if (!impl_->provider) {
    promise->set_value(
        Refused(DecisionStatus::Closed, {}, "agent has no provider"));
    return future;
  }

  const Impl::Attempt attempt = impl_->BeginAttempt();
  if (!attempt.started) {
    promise->set_value(attempt.refusal);
    return future;
  }

  try {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    Impl* impl = impl_.get();
    impl->tasks.push_back([impl, attempt, promise]() mutable {
      DecisionResult result;
      try {
        CompletionResult completion =
            CallProvider(attempt.provider, attempt.request);
        // The worker finishes against the Impl, never against the Agent
        // object.
        result = impl->FinishAttempt(attempt, std::move(completion));
      } catch (const std::exception& error) {
        result = impl->AbortAttempt(attempt, std::string("decision worker ", "failed: ") + error.what());
      } catch (...) {
        result = impl->AbortAttempt(attempt, "decision worker failed");
      }
      try {
        promise->set_value(std::move(result));
      } catch (...) {
        // The future is already satisfied or gone; nothing left to report.
      }
    });
    if (!impl_->worker.joinable()) {
      impl_->worker = std::thread([impl = impl_.get()] { impl->Run(); });
    }
  } catch (const std::exception& error) {
    // Starting the worker failed: undo the reservation so the agent does not
    // stay Busy forever, and answer the caller with a real result.
    {
      std::lock_guard<std::mutex> lock(impl_->mutex);
      if (!impl_->tasks.empty()) {
        impl_->tasks.pop_back();
      }
      impl_->in_flight_revision.reset();
      impl_->idle_cv.notify_all();
    }
    promise->set_value(Refused(
        DecisionStatus::Busy, attempt.context_revision,
        std::string("could not start the decision worker: ") + error.what()));
    return future;
  }
  impl_->work_cv.notify_all();
  return future;
}

RecordOutcome Agent::RecordCommandResults(std::uint64_t decision_id,
                                       std::vector<CommandResult> results) {
  std::unique_lock<std::mutex> lock(impl_->mutex);
  const std::shared_ptr<ILogger> logger = impl_->logger;
  const std::string id = impl_->id;

  // Filled once the decision is found, so even a rejected record keeps the
  // request id of the attempt it belongs to.
  std::uint64_t request_id = 0;
  const auto reject = [&](const char* reason) {
    lock.unlock();
    EmitLog(logger, LogLevel::Warning, log_event::kDecisionRecordRejected,
            std::string("decision_id=") + std::to_string(decision_id) + ", " +
                reason,
            LogContext{id, request_id});
  };

  if (impl_->closed) {
    reject("agent is closed");
    return RecordOutcome::Closed;
  }
  const Impl::Pending* pending = impl_->FindPendingLocked(decision_id);
  if (pending == nullptr) {
    reject("no decision with that id is awaiting results");
    return RecordOutcome::UnknownDecision;
  }
  request_id = pending->request_id;

  // Every command call needs exactly one result, and nothing else: a decision is
  // recorded as a whole or not at all.
  std::vector<Message> result_messages;
  result_messages.reserve(pending->command_call_ids.size());
  for (const std::string& call_id : pending->command_call_ids) {
    const CommandResult* match = nullptr;
    std::size_t matches = 0;
    for (const CommandResult& candidate : results) {
      if (candidate.command_call_id == call_id) {
        match = &candidate;
        ++matches;
      }
    }
    if (matches != 1) {
      reject("results do not cover every command call exactly once");
      return RecordOutcome::IncompleteResults;
    }
    result_messages.push_back(Message::CommandResultMessage(*match));
  }
  if (results.size() != pending->command_call_ids.size()) {
    // Extra results, or results for a call that was never made.
    reject("results do not cover every command call exactly once");
    return RecordOutcome::IncompleteResults;
  }

  DecisionRecord record;
  record.decision_id = pending->decision_id;
  record.context_revision = pending->context_revision;
  const std::string record_revision = pending->context_revision;
  record.assistant = pending->assistant;
  record.command_results = std::move(result_messages);
  impl_->context.AppendDecision(std::move(record));
  impl_->pending.erase(
      std::remove_if(impl_->pending.begin(), impl_->pending.end(),
                     [decision_id](const Impl::Pending& candidate) {
                       return candidate.decision_id == decision_id;
                     }),
      impl_->pending.end());
  impl_->idle_cv.notify_all();
  lock.unlock();

  EmitLog(logger, LogLevel::Info, log_event::kDecisionRecorded,
          "decision_id=" + std::to_string(decision_id) + ", results=" +
              std::to_string(results.size()) + ", rev=" + record_revision,
          LogContext{id, request_id});
  return RecordOutcome::Recorded;
}

bool Agent::DiscardDecision(std::uint64_t decision_id) {
  std::unique_lock<std::mutex> lock(impl_->mutex);
  const std::shared_ptr<ILogger> logger = impl_->logger;
  const std::string id = impl_->id;
  std::uint64_t request_id = 0;
  for (const Impl::Pending& candidate : impl_->pending) {
    if (candidate.decision_id == decision_id) {
      request_id = candidate.request_id;
      break;
    }
  }
  const std::size_t before = impl_->pending.size();
  impl_->pending.erase(
      std::remove_if(impl_->pending.begin(), impl_->pending.end(),
                     [decision_id](const Impl::Pending& candidate) {
                       return candidate.decision_id == decision_id;
                     }),
      impl_->pending.end());
  const bool removed = impl_->pending.size() != before;
  if (removed) {
    impl_->idle_cv.notify_all();
  }
  lock.unlock();

  if (removed) {
    EmitLog(logger, LogLevel::Info, log_event::kDecisionDiscarded,
            "decision_id=" + std::to_string(decision_id),
            LogContext{id, request_id});
  }
  return removed;
}

bool Agent::IsBusy() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->in_flight_revision.has_value() || !impl_->pending.empty();
}

bool Agent::IsClosed() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->closed;
}

std::size_t Agent::InFlightDecisionCount() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->in_flight_revision.has_value() ? 1u : 0u;
}

std::size_t Agent::PendingDecisionCount() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->pending.size();
}

bool Agent::WaitForIdle(std::chrono::milliseconds timeout) const {
  std::unique_lock<std::mutex> lock(impl_->mutex);
  return impl_->idle_cv.wait_for(
      lock, timeout, [this] { return !impl_->in_flight_revision.has_value(); });
}

void Agent::Close() {
  std::unique_lock<std::mutex> lock(impl_->mutex);
  const bool was_open = !impl_->closed;
  impl_->closed = true;
  const std::shared_ptr<ILogger> logger = impl_->logger;
  const std::string id = impl_->id;
  lock.unlock();

  if (was_open) {
    EmitLog(logger, LogLevel::Info, log_event::kSessionClosed, "session=" + id,
            LogContext{id, 0});
  }
}

}  // namespace llm
