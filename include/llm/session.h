#pragma once

/// One decision stream: context, tools, provider and the decisions awaiting
/// confirmation.
///
/// A Session is what a game talks to. It owns the message history
/// (llm::Context), builds the request, asks the provider once, and stages the
/// result until the application reports what it actually executed.
///
/// Two invariants keep the model's view of the world honest:
///
///  * at most one decision may be awaiting results at a time, so a new
///    decision is never taken with an incomplete transcript;
///  * a decision enters the history atomically — a decision is either fully
///    confirmed (its tool calls and all their results) or not recorded at all.
///
/// A Session never executes tools and never decides on its own to continue:
/// both are the application's call.

#include "context.h"
#include "provider.h"
#include "request.h"
#include "result.h"
#include "types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llm {

/// Session-level outcome of a decision attempt. `completion` is only meaningful
/// when this is Completed.
enum class DecisionStatus {
  Completed,  // the provider answered; branch on completion.status
  Busy,       // a previous decision is still awaiting results
  Closed,     // the session was closed
};

struct DecisionResult {
  /// 0 when no decision was started (Busy/Closed).
  std::uint64_t decision_id = 0;
  /// Revision of the context this decision was taken from.
  std::string context_revision;
  DecisionStatus status = DecisionStatus::Closed;
  CompletionResult completion;
};

/// What happened to an attempt to record tool results.
enum class RecordOutcome {
  Recorded,           // the decision is now part of the history
  UnknownDecision,    // no such decision is awaiting results
  IncompleteResults,  // the results do not cover exactly the pending tool calls
  Closed,
};

struct SessionConfig {
  ContextConfig context;
  std::string system_prompt;
  std::string model;  // the model every request in this session uses
};

/// One coach, one side, one history. Independent Sessions share nothing.
///
/// Thread-safe: the state is guarded, and the provider call runs outside the
/// lock so a slow completion does not block status queries.
class Session {
 public:
  explicit Session(std::shared_ptr<IModelProvider> provider,
                   SessionConfig config = {});
  ~Session();

  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  // --- configuration ------------------------------------------------------
  void SetTools(std::vector<ToolDefinition> tools);
  /// Returned by value: a reference into the session would outlive the lock.
  std::vector<ToolDefinition> tools() const;
  void SetSystemPrompt(std::string prompt);
  void SetModel(std::string model);
  void SetContextConfig(ContextConfig config);
  std::string model() const;
  /// A copy of the message history, safe to inspect without holding anything.
  Context ContextSnapshot() const;

  // --- decisions ----------------------------------------------------------
  /// One decision: render the context, ask the provider exactly once, and keep
  /// the result pending until it is confirmed or discarded.
  DecisionResult Complete(ContextUpdate update);

  /// Confirms a pending decision with the results the application executed.
  /// The results must cover every tool call of that decision, exactly once;
  /// otherwise nothing is recorded and the pending decision stays.
  RecordOutcome RecordToolResults(std::uint64_t decision_id,
                                  std::vector<ToolResult> results);

  /// Forgets a pending decision without recording it (a stale revision, a
  /// rolled-back transaction, something the application chose not to apply).
  bool DiscardDecision(std::uint64_t decision_id);

  std::size_t pending_decision_count() const;
  std::vector<std::uint64_t> pending_decision_ids() const;
  /// Revision of the newest context update this session received.
  std::string current_revision() const;

  // --- lifetime -----------------------------------------------------------
  bool closed() const;
  /// Refuses further decisions. Pending futures (see CompleteAsync) are
  /// answered as Closed, and the destructor waits for in-flight work.
  void Close();

 private:
  struct State;
  std::shared_ptr<State> state_;
  std::shared_ptr<IModelProvider> provider_;
};

/// Convenience factory.
std::shared_ptr<Session> CreateSession(std::shared_ptr<IModelProvider> provider,
                                       SessionConfig config = {});

}  // namespace llm
