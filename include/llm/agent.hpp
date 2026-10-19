#pragma once

/// One decision stream: an agent owns its provider, model, prompt, commands and
/// the history of its decisions.
///
/// The game creates an agent, updates it when the world changes, asks for a
/// decision asynchronously, prepares the returned command calls and executes
/// them itself, then reports the results back. The SDK never executes commands,
/// never advances the conversation on its own and never decides when to ask
/// again.
///
/// Invariants, both about the transcript the model sees:
///  * at most one decision may be in flight or awaiting results at a time, so a
///    decision is never taken with an incomplete history;
///  * a decision enters the history atomically: the assistant turn with its
///    command calls plus one result per call, or nothing at all.

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

/// Agent-level outcome of a decision attempt. `completion` is only meaningful
/// when this is Completed.
enum class DecisionStatus {
  Completed,  // the provider answered; branch on completion.status
  Busy,       // a previous decision is still in flight or awaiting results
  Closed,     // the agent was closed
};

struct DecisionResult {
  /// 0 when no decision was started (Busy/Closed).
  std::uint64_t decision_id = 0;
  /// Revision of the context this decision was taken from.
  std::string context_revision;
  DecisionStatus status = DecisionStatus::Closed;
  CompletionResult completion;
};

/// What happened to an attempt to record command results.
enum class RecordOutcome {
  Recorded,           // the decision is now part of the history
  UnknownDecision,    // no such decision is awaiting results
  IncompleteResults,  // the results do not cover exactly the pending calls
  Closed,
};

struct AgentConfig {
  /// Identifies this decision stream in log records (for example "home" and
  /// "away" when one provider serves both sides).
  std::string id;
  /// The model every request of this agent uses.
  std::string model;
  std::string system_prompt;

  /// Transcript budget: see ContextConfig.
  ContextConfig context;

  /// Optional diagnostics sink. Without it the agent stays silent.
  std::shared_ptr<ILogger> logger;

  /// Applied to every request: a decision round should be bounded, and the
  /// command-call bound is what the validator enforces.
  std::optional<std::int32_t> max_output_tokens;
  std::optional<std::size_t> max_command_calls;
};

/// One coach, one side, one history. Independent agents share nothing, not even
/// when they share a provider.
///
/// Thread-safe: the state is guarded, and the provider call runs on the agent's
/// own worker thread, so a completion never blocks the caller.
class Agent final {
 public:
  Agent(std::shared_ptr<IModelProvider> provider, AgentConfig config);
  ~Agent();

  Agent(const Agent&) = delete;
  Agent& operator=(const Agent&) = delete;

  // --- configuration ------------------------------------------------------
  void SetSystemPrompt(std::string prompt);
  void SetCommands(std::vector<CommandDefinition> commands);
  void SetModel(std::string model);

  std::string model() const;
  std::vector<CommandDefinition> commands() const;

  // --- context ------------------------------------------------------------
  /// Records the latest world state and recent-history window.
  ///
  /// It may be called at any time, including while a decision is in flight: the
  /// update then applies to the *next* decision, never to the request that was
  /// already sent (the request is snapshotted when DecideAsync() is called).
  void UpdateContext(ContextUpdate update);

  /// Revision of the newest context update this agent received.
  std::string current_revision() const;
  /// A copy of the message history, safe to inspect without holding anything.
  Context ContextSnapshot() const;

  // --- decisions ----------------------------------------------------------
  /// Starts one decision from the context recorded so far: renders it, asks the
  /// provider exactly once on the agent's worker, and keeps the result pending
  /// until it is confirmed or discarded.
  ///
  /// Returns Busy when a decision is already in flight or awaiting results, and
  /// Closed after Close(). The returned future is always satisfied — Busy and
  /// Closed are reported as DecisionStatus, never as a broken promise.
  std::future<DecisionResult> DecideAsync();

  /// Confirms a pending decision with the results the application executed.
  /// The results must cover every command call of that decision, exactly once;
  /// otherwise nothing is recorded and the decision stays pending.
  RecordOutcome RecordCommandResults(std::uint64_t decision_id,
                                     std::vector<CommandResult> results);

  /// Forgets a pending decision without recording it (a stale revision, a
  /// rolled-back transaction, something the application chose not to apply).
  bool DiscardDecision(std::uint64_t decision_id);

  // --- state --------------------------------------------------------------
  /// True while any decision is in flight or awaiting results.
  bool IsBusy() const;
  bool IsClosed() const;
  std::size_t InFlightDecisionCount() const;
  std::size_t PendingDecisionCount() const;
  bool WaitForIdle(std::chrono::milliseconds timeout) const;

  /// Refuses further decisions: a submitted-but-unfinished decision is answered
  /// as Closed when it returns, and nothing is staged.
  ///
  /// Closing does not abort an HTTP request that is already running; the
  /// destructor joins the worker, so destroying an agent can wait for the
  /// provider's timeout.
  void Close();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace llm
