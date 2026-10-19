#include "llm/context/session.h"

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

/// State shared by the session and its worker thread.
///
/// The worker only ever touches this object (never the Session), so a decision
/// that is already running cannot reach a destroyed session.
struct Session::State {
  mutable std::mutex mutex;
  mutable std::condition_variable
      work_cv;  // a task arrived, or we are stopping
  mutable std::condition_variable idle_cv;  // nothing is in flight any more

  Context context;
  std::string system_prompt;
  std::string model;
  std::optional<std::int32_t> max_tokens;
  std::optional<std::size_t> max_tool_calls;
  std::shared_ptr<ILogger> logger;
  std::string session_id;
  bool closed = false;
  std::uint64_t next_decision_id = 1;
  /// Diagnostics id for each accepted decision attempt; it travels with the
  /// request so provider events can be attributed to it.
  std::uint64_t next_request_id = 1;

  /// Revision of the decision currently being asked about, if any.
  std::optional<std::string> in_flight_revision;

  /// The single decision awaiting tool results, if any.
  struct Pending {
    std::uint64_t decision_id = 0;
    std::string context_revision;
    Message assistant;
    std::vector<std::string> tool_call_ids;  // in call order
  };
  std::vector<Pending> pending;  // at most one in this version

  // Worker plumbing.
  std::thread worker;
  std::deque<std::function<void()>> tasks;
  bool stopping = false;

  explicit State(ContextConfig config) : context(std::move(config)) {}

  const Pending* FindPendingLocked(std::uint64_t decision_id) const {
    for (const Pending& candidate : pending) {
      if (candidate.decision_id == decision_id) {
        return &candidate;
      }
    }
    return nullptr;
  }
};

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
/// may throw: neither path may let that escape into a game loop.
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

/// One accepted decision: the request is snapshotted on the submitting thread
/// so later context updates cannot change what is being asked.
struct Session::Attempt {
  bool started = false;
  std::string context_revision;
  std::uint64_t request_id = 0;
  ModelRequest request;
  DecisionResult refusal;  // valid when !started
};

Session::Session(std::shared_ptr<IModelProvider> provider, SessionConfig config)
    : state_(std::make_shared<State>(std::move(config.context))),
      provider_(std::move(provider)) {
  state_->system_prompt = std::move(config.system_prompt);
  state_->model = std::move(config.model);
  state_->max_tokens = config.max_tokens;
  state_->max_tool_calls = config.max_tool_calls;
  state_->logger = std::move(config.logger);
  state_->session_id = std::move(config.session_id);

  EmitLog(state_->logger, LogLevel::Info, log_event::kSessionCreated,
          "model=" + state_->model + ", max_messages=" +
              std::to_string(state_->context.config().max_messages) +
              ", session=" + state_->session_id,
          LogContext{state_->session_id, 0});
  if (!state_->system_prompt.empty()) {
    state_->context.SetSystemPrompt(state_->system_prompt);
  }
}

Session::~Session() {
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->closed = true;
    state_->stopping = true;
  }
  state_->work_cv.notify_all();
  if (state_->worker.joinable()) {
    state_->worker.join();
  }
}

void Session::SetTools(std::vector<ToolDefinition> tools) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->context.SetTools(std::move(tools));
}

std::vector<ToolDefinition> Session::tools() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->context.tools();
}

void Session::SetSystemPrompt(std::string prompt) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->system_prompt = prompt;
  state_->context.SetSystemPrompt(std::move(prompt));
}

void Session::SetModel(std::string model) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->model = std::move(model);
}

void Session::SetContextConfig(ContextConfig config) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->context.SetConfig(config);
}

std::string Session::model() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->model;
}

Context Session::ContextSnapshot() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->context;
}

Session::Attempt Session::BeginAttempt(ContextUpdate update) {
  Attempt attempt;
  const std::shared_ptr<ILogger> logger = state_->logger;
  const std::string session_id = state_->session_id;

  {
    std::lock_guard<std::mutex> lock(state_->mutex);

    if (state_->closed) {
      attempt.refusal =
          Refused(DecisionStatus::Closed, state_->context.context_revision(),
                  "session is closed");
    } else if (state_->in_flight_revision.has_value() ||
               !state_->pending.empty()) {
      attempt.refusal =
          Refused(DecisionStatus::Busy, state_->context.context_revision(),
                  state_->in_flight_revision.has_value()
                      ? "a decision is already in flight"
                      : "a decision is still awaiting tool results; call "
                        "RecordToolResults or DiscardDecision first");
    } else {
      state_->context.SetContextUpdate(std::move(update));
      attempt.started = true;
      attempt.context_revision = state_->context.context_revision();
      attempt.request_id = state_->next_request_id++;
      attempt.request = state_->context.BuildRequest(state_->model);
      attempt.request.max_tokens = state_->max_tokens;
      attempt.request.max_tool_calls = state_->max_tool_calls;
      attempt.request.log = LogContext{state_->session_id, attempt.request_id};
      state_->in_flight_revision = attempt.context_revision;
    }
  }

  // Emitted outside the lock: a logger may call back into the session.
  if (attempt.started) {
    EmitLog(logger, LogLevel::Debug, log_event::kDecisionStarted,
            "rev=" + attempt.context_revision +
                ", tools=" + std::to_string(attempt.request.tools.size()) +
                ", messages=" + std::to_string(attempt.request.messages.size()),
            attempt.request.log);
  } else {
    EmitLog(logger, LogLevel::Warning, log_event::kDecisionRejected,
            attempt.refusal.completion.ErrorMessage(),
            LogContext{session_id, 0});
  }
  return attempt;
}

DecisionResult Session::FinishAttempt(const std::shared_ptr<State>& state,
                                      const Attempt& attempt,
                                      CompletionResult completion) {
  DecisionResult result;
  const std::shared_ptr<ILogger> logger = state->logger;
  const LogContext context{state->session_id, attempt.request_id};
  result.context_revision = attempt.context_revision;
  result.status = DecisionStatus::Completed;
  result.completion = std::move(completion);

  std::lock_guard<std::mutex> lock(state->mutex);
  state->in_flight_revision.reset();
  state->idle_cv.notify_all();

  if (state->closed) {
    result = Refused(DecisionStatus::Closed, attempt.context_revision,
                     "session was closed while the decision was in flight");
  } else if (result.completion.IsSuccess()) {
    State::Pending pending;
    pending.decision_id = state->next_decision_id++;
    pending.context_revision = attempt.context_revision;
    pending.assistant = result.completion.ToAssistantMessage();
    for (const ToolCall& call : result.completion.tool_calls) {
      pending.tool_call_ids.push_back(call.id);
    }
    result.decision_id = pending.decision_id;
    state->pending.push_back(std::move(pending));
  }

  // Emitted outside the lock.
  EmitLog(
      logger,
      result.completion.IsSuccess() ? LogLevel::Info : LogLevel::Warning,
      log_event::kDecisionCompleted,
      "status=" + ToString(result.completion.status) +
          ", tools=" + std::to_string(result.completion.tool_calls.size()) +
          ", tokens=" + std::to_string(result.completion.usage.total_tokens) +
          ", attempts=" + std::to_string(result.completion.http_attempts) +
          ", decision_id=" + std::to_string(result.decision_id) +
          ", model=" + result.completion.model,
      context);
  return result;
}

DecisionResult Session::Complete(ContextUpdate update) {
  if (!provider_) {
    return Refused(DecisionStatus::Closed, {}, "session has no provider");
  }
  const Attempt attempt = BeginAttempt(std::move(update));
  if (!attempt.started) {
    return attempt.refusal;
  }
  // The provider call happens outside the lock.
  return FinishAttempt(state_, attempt,
                       CallProvider(provider_, attempt.request));
}

std::future<DecisionResult> Session::CompleteAsync(ContextUpdate update) {
  // A shared promise: the queued task must be copy-constructible, and
  // std::promise is not.
  auto promise = std::make_shared<std::promise<DecisionResult>>();
  std::future<DecisionResult> future = promise->get_future();

  if (!provider_) {
    promise->set_value(
        Refused(DecisionStatus::Closed, {}, "session has no provider"));
    return future;
  }

  const Attempt attempt = BeginAttempt(std::move(update));
  if (!attempt.started) {
    promise->set_value(attempt.refusal);
    return future;
  }

  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->tasks.push_back([state = state_, provider = provider_, attempt,
                             promise]() mutable {
      CompletionResult completion = CallProvider(provider, attempt.request);
      // The worker finishes against the shared state, never against the
      // Session object.
      promise->set_value(FinishAttempt(state, attempt, std::move(completion)));
    });
    if (!state_->worker.joinable()) {
      state_->worker = std::thread(&Session::RunWorker, state_);
    }
  }
  state_->work_cv.notify_all();
  return future;
}

void Session::RunWorker(std::shared_ptr<State> state) {
  for (;;) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(state->mutex);
      state->work_cv.wait(
          lock, [&state] { return state->stopping || !state->tasks.empty(); });
      if (state->tasks.empty()) {
        if (state->stopping) {
          return;  // drained: accepted decisions have been answered
        }
        continue;
      }
      task = std::move(state->tasks.front());
      state->tasks.pop_front();
    }
    task();
  }
}

RecordOutcome Session::RecordToolResults(std::uint64_t decision_id,
                                         std::vector<ToolResult> results) {
  std::unique_lock<std::mutex> lock(state_->mutex);
  const std::shared_ptr<ILogger> logger = state_->logger;
  const std::string session_id = state_->session_id;

  const auto reject = [&](const char* reason) {
    lock.unlock();
    EmitLog(logger, LogLevel::Warning, log_event::kDecisionRecordRejected,
            std::string("decision_id=") + std::to_string(decision_id) + ", " +
                reason,
            LogContext{session_id, 0});
  };

  if (state_->closed) {
    reject("session is closed");
    return RecordOutcome::Closed;
  }
  const State::Pending* pending = state_->FindPendingLocked(decision_id);
  if (pending == nullptr) {
    reject("no decision with that id is awaiting results");
    return RecordOutcome::UnknownDecision;
  }

  // Every tool call needs exactly one result, and nothing else: a decision is
  // recorded as a whole or not at all.
  std::vector<Message> tool_messages;
  tool_messages.reserve(pending->tool_call_ids.size());
  for (const std::string& call_id : pending->tool_call_ids) {
    const ToolResult* match = nullptr;
    std::size_t matches = 0;
    for (const ToolResult& candidate : results) {
      if (candidate.tool_call_id == call_id) {
        match = &candidate;
        ++matches;
      }
    }
    if (matches != 1) {
      return RecordOutcome::IncompleteResults;
    }
    tool_messages.push_back(Message::ToolResult(*match));
  }
  if (results.size() != pending->tool_call_ids.size()) {
    return RecordOutcome::IncompleteResults;
  }

  DecisionRecord record;
  record.decision_id = pending->decision_id;
  record.context_revision = pending->context_revision;
  const std::string record_revision = pending->context_revision;
  record.assistant = pending->assistant;
  record.tool_results = std::move(tool_messages);
  state_->context.AppendDecision(std::move(record));
  state_->pending.erase(
      std::remove_if(state_->pending.begin(), state_->pending.end(),
                     [decision_id](const State::Pending& candidate) {
                       return candidate.decision_id == decision_id;
                     }),
      state_->pending.end());
  state_->idle_cv.notify_all();
  lock.unlock();

  EmitLog(logger, LogLevel::Info, log_event::kDecisionRecorded,
          "decision_id=" + std::to_string(decision_id) + ", results=" +
              std::to_string(results.size()) + ", rev=" + record_revision,
          LogContext{session_id, 0});
  return RecordOutcome::Recorded;
}

bool Session::DiscardDecision(std::uint64_t decision_id) {
  std::unique_lock<std::mutex> lock(state_->mutex);
  const std::shared_ptr<ILogger> logger = state_->logger;
  const std::string session_id = state_->session_id;
  const std::size_t before = state_->pending.size();
  state_->pending.erase(
      std::remove_if(state_->pending.begin(), state_->pending.end(),
                     [decision_id](const State::Pending& candidate) {
                       return candidate.decision_id == decision_id;
                     }),
      state_->pending.end());
  const bool removed = state_->pending.size() != before;
  if (removed) {
    state_->idle_cv.notify_all();
  }
  lock.unlock();

  if (removed) {
    EmitLog(logger, LogLevel::Info, log_event::kDecisionDiscarded,
            "decision_id=" + std::to_string(decision_id),
            LogContext{session_id, 0});
  }
  return removed;
}

std::size_t Session::in_flight_decision_count() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->in_flight_revision.has_value() ? 1u : 0u;
}

std::size_t Session::pending_decision_count() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->pending.size();
}

bool Session::busy() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->in_flight_revision.has_value() || !state_->pending.empty();
}

bool Session::WaitForIdle(std::chrono::milliseconds timeout) const {
  const std::shared_ptr<State> state = state_;
  std::unique_lock<std::mutex> lock(state->mutex);
  return state->idle_cv.wait_for(lock, timeout, [&state] {
    return !state->in_flight_revision.has_value();
  });
}

std::vector<std::uint64_t> Session::pending_decision_ids() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  std::vector<std::uint64_t> ids;
  ids.reserve(state_->pending.size());
  for (const State::Pending& pending : state_->pending) {
    ids.push_back(pending.decision_id);
  }
  return ids;
}

std::string Session::current_revision() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->context.context_revision();
}

bool Session::closed() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->closed;
}

void Session::Close() {
  std::unique_lock<std::mutex> lock(state_->mutex);
  const bool was_open = !state_->closed;
  state_->closed = true;
  const std::shared_ptr<ILogger> logger = state_->logger;
  const std::string session_id = state_->session_id;
  lock.unlock();

  if (was_open) {
    EmitLog(logger, LogLevel::Info, log_event::kSessionClosed,
            "session=" + session_id, LogContext{session_id, 0});
  }
}

std::shared_ptr<Session> CreateSession(std::shared_ptr<IModelProvider> provider,
                                       SessionConfig config) {
  return std::make_shared<Session>(std::move(provider), std::move(config));
}

}  // namespace llm
