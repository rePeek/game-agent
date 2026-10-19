#include "llm/session.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace llm {

/// State shared by the session and (from CTX-3 on) its background worker: the
/// worker must never touch the Session object itself, so that a pending
/// completion cannot outlive the object that started it.
struct Session::State {
  mutable std::mutex mutex;
  Context context;
  std::string system_prompt;
  std::string model;
  bool closed = false;
  std::uint64_t next_decision_id = 1;

  /// The single decision awaiting tool results, if any.
  struct Pending {
    std::uint64_t decision_id = 0;
    std::string context_revision;
    Message assistant;
    std::vector<std::string> tool_call_ids;  // in call order
  };
  std::vector<Pending> pending;  // at most one in this version

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

Session::Session(std::shared_ptr<IModelProvider> provider, SessionConfig config)
    : state_(std::make_shared<State>(std::move(config.context))),
      provider_(std::move(provider)) {
  state_->system_prompt = std::move(config.system_prompt);
  state_->model = std::move(config.model);
  if (!state_->system_prompt.empty()) {
    state_->context.SetSystemPrompt(state_->system_prompt);
  }
}

Session::~Session() = default;

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

DecisionResult Session::Complete(ContextUpdate update) {
  if (!provider_) {
    return Refused(DecisionStatus::Closed, {}, "session has no provider");
  }

  ModelRequest request;
  std::string revision;
  {
    std::lock_guard<std::mutex> lock(state_->mutex);

    if (state_->closed) {
      return Refused(DecisionStatus::Closed, state_->context.context_revision(),
                     "session is closed");
    }
    if (!state_->pending.empty()) {
      return Refused(DecisionStatus::Busy, state_->context.context_revision(),
                     "a decision is still awaiting tool results; call "
                     "RecordToolResults or DiscardDecision first");
    }

    state_->context.SetContextUpdate(std::move(update));
    revision = state_->context.context_revision();
    request = state_->context.BuildRequest(state_->model);
  }

  // The provider call happens outside the lock: a slow completion must not
  // block status queries, and the worker in CTX-3 relies on this shape.
  CompletionResult completion = provider_->Complete(request);

  DecisionResult result;
  result.context_revision = revision;
  result.completion = completion;
  result.status = DecisionStatus::Completed;
  if (!completion.IsSuccess()) {
    return result;  // nothing to confirm, nothing to record
  }

  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->closed) {
      return Refused(DecisionStatus::Closed, revision,
                     "session was closed while the decision was in flight");
    }
    State::Pending pending;
    pending.decision_id = state_->next_decision_id++;
    pending.context_revision = revision;
    pending.assistant = completion.ToAssistantMessage();
    for (const ToolCall& call : completion.tool_calls) {
      pending.tool_call_ids.push_back(call.id);
    }
    result.decision_id = pending.decision_id;
    state_->pending.push_back(std::move(pending));
  }

  return result;
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
    // Extra results, or results for a call that was never made.
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
  return RecordOutcome::Recorded;
}

bool Session::DiscardDecision(std::uint64_t decision_id) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  const auto before = state_->pending.size();
  state_->pending.erase(
      std::remove_if(state_->pending.begin(), state_->pending.end(),
                     [decision_id](const State::Pending& candidate) {
                       return candidate.decision_id == decision_id;
                     }),
      state_->pending.end());
  return state_->pending.size() != before;
}

std::size_t Session::pending_decision_count() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->pending.size();
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
