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

#include "llm/context/context.h"
#include "llm/core/logger.h"
#include "llm/core/provider.h"
#include "llm/core/request.h"
#include "llm/core/result.h"
#include "llm/core/types.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
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

  /// Applied to every request this session builds: a decision round should be
  /// bounded, and the tool-call bound is what the validator enforces.
  std::optional<std::int32_t> max_tokens;
  std::optional<std::size_t> max_tool_calls;

  /// Optional diagnostics sink. Without it the session produces no log events
  /// at all; the same logger can be shared by several sessions and by the
  /// provider they use.
  std::shared_ptr<ILogger> logger;

  /// Identifies this decision stream in log records (for example "home" and
  /// "away" when one provider serves both sides).
  std::string session_id;
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

  /// The same decision, run on the session's own worker thread: submitting
  /// never blocks, and the calling thread is free while the provider works.
  ///
  /// The returned future is always satisfied — Busy and Closed are reported as
  /// DecisionStatus values, never as a broken promise or an exception. The
  /// request is snapshotted at submit time, so later context updates cannot
  /// change what was asked.
  std::future<DecisionResult> CompleteAsync(ContextUpdate update);

  /// Confirms a pending decision with the results the application executed.
  /// The results must cover every tool call of that decision, exactly once;
  /// otherwise nothing is recorded and the pending decision stays.
  RecordOutcome RecordToolResults(std::uint64_t decision_id,
                                  std::vector<ToolResult> results);

  /// Forgets a pending decision without recording it (a stale revision, a
  /// rolled-back transaction, something the application chose not to apply).
  bool DiscardDecision(std::uint64_t decision_id);

  /// Decisions the worker is currently asking the provider about.
  std::size_t in_flight_decision_count() const;
  /// Decisions whose provider call has returned, maybe on the worker thread.
  std::size_t pending_decision_count() const;
  /// True while any decision is in flight or awaiting results.
  bool busy() const;
  /// Waits until nothing is in flight. Returns false if the wait timed out; a
  /// game loop uses this instead of blocking on a future.
  bool WaitForIdle(std::chrono::milliseconds timeout) const;
  std::vector<std::uint64_t> pending_decision_ids() const;
  /// Revision of the newest context update this session received.
  std::string current_revision() const;

  // --- lifetime -----------------------------------------------------------
  bool closed() const;
  /// Refuses further decisions: a submitted-but-unfinished decision is
  /// answered as Closed when it returns, and nothing is staged.
  ///
  /// Closing does not abort an HTTP request that is already running
  /// (ITransport::Send is blocking); it only stops its result from being used.
  /// The destructor joins the worker, so destroying a session can wait for the
  /// provider's timeout. Call Close() early, or WaitForIdle(), when that
  /// matters.
  void Close();

 private:
  struct State;
  struct Attempt;

  /// Applies the update and reserves the session for one decision, or returns
  /// a refusal (Closed/Busy) without touching anything.
  Attempt BeginAttempt(ContextUpdate update);
  /// Clears the in-flight mark and stages the decision when it succeeded.
  /// Static and state-taking so the worker never touches the Session object.
  static DecisionResult FinishAttempt(const std::shared_ptr<State>& state,
                                      const Attempt& attempt,
                                      CompletionResult completion);
  static void RunWorker(std::shared_ptr<State> state);

  std::shared_ptr<State> state_;
  std::shared_ptr<IModelProvider> provider_;
};

/// Convenience factory.
std::shared_ptr<Session> CreateSession(std::shared_ptr<IModelProvider> provider,
                                       SessionConfig config = {});

}  // namespace llm
