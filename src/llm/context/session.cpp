#include "llm/session.h"

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
  bool closed = false;
  std::uint64_t next_decision_id = 1;

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
  ModelRequest request;
  DecisionResult refusal;  // valid when !started
};

Session::Session(std::shared_ptr<IModelProvider> provider, SessionConfig config)
    : state_(std::make_shared<State>(std::move(config.context))),
      provider_(std::move(provider)) {
  state_->system_prompt = std::move(config.system_prompt);
  state_->model = std::move(config.model);
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
  std::lock_guard<std::mutex> lock(state_->mutex);

  if (state_->closed) {
    attempt.refusal =
        Refused(DecisionStatus::Closed, state_->context.context_revision(),
                "session is closed");
    return attempt;
  }
  if (state_->in_flight_revision.has_value() || !state_->pending.empty()) {
    attempt.refusal =
        Refused(DecisionStatus::Busy, state_->context.context_revision(),
                state_->in_flight_revision.has_value()
                    ? "a decision is already in flight"
                    : "a decision is still awaiting tool results; call "
                      "RecordToolResults or DiscardDecision first");
    return attempt;
  }

  state_->context.SetContextUpdate(std::move(update));
  attempt.started = true;
  attempt.context_revision = state_->context.context_revision();
  attempt.request = state_->context.BuildRequest(state_->model);
  state_->in_flight_revision = attempt.context_revision;
  return attempt;
}

DecisionResult Session::FinishAttempt(const std::shared_ptr<State>& state,
                                      const Attempt& attempt,
                                      CompletionResult completion) {
  DecisionResult result;
  result.context_revision = attempt.context_revision;
  result.status = DecisionStatus::Completed;
  result.completion = std::move(completion);

  std::lock_guard<std::mutex> lock(state->mutex);
  state->in_flight_revision.reset();
  state->idle_cv.notify_all();

  if (state->closed) {
    return Refused(DecisionStatus::Closed, attempt.context_revision,
                   "session was closed while the decision was in flight");
  }
  if (!result.completion.IsSuccess()) {
    return result;  // nothing to confirm, nothing to record
  }

  State::Pending pending;
  pending.decision_id = state->next_decision_id++;
  pending.context_revision = attempt.context_revision;
  pending.assistant = result.completion.ToAssistantMessage();
  for (const ToolCall& call : result.completion.tool_calls) {
    pending.tool_call_ids.push_back(call.id);
  }
  result.decision_id = pending.decision_id;
  state->pending.push_back(std::move(pending));
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
  std::lock_guard<std::mutex> lock(state_->mutex);

  if (state_->closed) {
    return RecordOutcome::Closed;
  }
  const State::Pending* pending = state_->FindPendingLocked(decision_id);
  if (pending == nullptr) {
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
  return RecordOutcome::Recorded;
}

bool Session::DiscardDecision(std::uint64_t decision_id) {
  std::lock_guard<std::mutex> lock(state_->mutex);
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
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->closed = true;
}

std::shared_ptr<Session> CreateSession(std::shared_ptr<IModelProvider> provider,
                                       SessionConfig config) {
  return std::make_shared<Session>(std::move(provider), std::move(config));
}

}  // namespace llm
