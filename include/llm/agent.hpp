#pragma once

/// One decision stream: an agent owns its provider, model, prompt, registered
/// commands and the history of its decisions.
///
/// The game creates an agent, registers the commands the model may call,
/// updates it when the world changes, asks for a decision asynchronously, binds
/// the returned calls to C++ callbacks, runs them itself and reports the
/// results back. The SDK never runs a command, never advances the conversation
/// on its own and never decides when to ask again.
///
/// Invariants, both about the transcript the model sees:
///  * at most one decision may be in flight or awaiting results at a time, so a
///    decision is never taken with an incomplete history;
///  * a decision enters the history atomically: the assistant turn with its
///    command calls plus one result per call, or nothing at all.

#include "llm/commands/prepared_command.hpp"
#include "llm/commands/registry.hpp"
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
#include <string_view>
#include <utility>
#include <vector>

namespace llm {

/// Agent-level outcome of a decision attempt. `completion` is only meaningful
/// when this is Completed.
enum class DecisionStatus {
  Completed,        // the provider answered; branch on completion.status
  NotConfigured,    // no provider or no model has been set yet
  Busy,             // a previous decision is still in flight or awaiting
  ContextTooLarge,  // the request would exceed the input token budget
  Closed,           // the agent was closed
};

struct DecisionResult {
  /// 0 when no decision was started (NotConfigured/Busy/Closed).
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
  NameMismatch,       // a result names a different command than the call did
  Closed,
};

/// What happened to an attempt to change agent configuration.
enum class ConfigStatus {
  Applied,
  InvalidProvider,
  InvalidModel,
  AgentBusy,
  AgentClosed,
};

struct ConfigOutcome {
  ConfigStatus status = ConfigStatus::Applied;
  std::string error;

  bool ok() const { return status == ConfigStatus::Applied; }
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
  explicit Agent(AgentConfig config = {});
  /// Convenience: constructs and applies the provider in one step. Equivalent
  /// to constructing with the config and calling SetProvider().
  Agent(std::shared_ptr<IModelProvider> provider, AgentConfig config);
  ~Agent();

  Agent(const Agent&) = delete;
  Agent& operator=(const Agent&) = delete;

  // --- configuration ------------------------------------------------------
  /// Installs the provider. Refused while a decision is in flight, so a call
  /// that is already running is never re-pointed at a different backend.
  ConfigOutcome SetProvider(std::shared_ptr<IModelProvider> provider);
  ConfigOutcome SetModel(std::string model);
  ConfigOutcome SetSystemPrompt(std::string prompt);

  std::string model() const;

  // --- commands -----------------------------------------------------------
  /// Registers a C++ function or lambda together with its typed parameter
  /// list. The parameter list types the callback, so a mismatch is a
  /// compile-time error; names and constraints are described once and drive
  /// both the JSON Schema and the argument decoder.
  template <typename Fn, typename... Args>
  CommandRegistrationResult RegisterCommand(
      std::string name,
      std::string description,
      CommandParameters<Args...> parameters,
      Fn&& callback) {
    return RegisterCommandErased(MakeRegisteredCommand<Fn, Args...>(
        std::move(name), std::move(description), std::move(parameters),
        std::forward<Fn>(callback)));
  }

  /// Removes a registered command. Refused while a decision is in flight or
  /// awaiting results.
  CommandRegistrationResult UnregisterCommand(std::string_view name);

  /// Removes every registered command.
  CommandRegistrationResult ClearCommands();

  /// The definitions currently offered to the model, in registration order.
  std::vector<CommandDefinition> Commands() const;

  /// Compatibility shim: registers definitions without callbacks. Kept only
  /// while the tests move to RegisterCommand; new code must not use it.
  void SetCommands(std::vector<CommandDefinition> commands);

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
  /// Never runs a command callback. Returns NotConfigured/Busy/Closed as a
  /// DecisionStatus; the returned future is always satisfied.
  std::future<DecisionResult> DecideAsync();

  /// Binds a pending decision's command calls to their callbacks, without
  /// running anything. The whole batch is prepared or nothing is: a single
  /// undecodable call fails the batch so the game never executes half a
  /// decision.
  PrepareResult PrepareCommands(std::uint64_t decision_id);

  /// Confirms a pending decision with the results the game executed. The
  /// results must cover every command call of that decision, exactly once, and
  /// name the same commands; otherwise nothing is recorded and the decision
  /// stays pending. On success the whole decision enters the history atomically.
  RecordOutcome RecordCommandResults(std::uint64_t decision_id,
                                     std::vector<CommandResult> results);

  /// Forgets a pending decision without recording it (a stale revision, a
  /// rolled-back transaction, something the game chose not to apply).
  bool DiscardDecision(std::uint64_t decision_id);

  // --- state --------------------------------------------------------------
  /// True while any decision is in flight or awaiting results.
  bool IsBusy() const;
  bool IsClosed() const;
  std::size_t InFlightDecisionCount() const;
  std::size_t PendingDecisionCount() const;
  /// Waits until no provider call is in flight. This is deliberately *not*
  /// the same as the decision future being ready: a decision that has answered
  /// is idle but still awaiting results, and a queued task may not have started
  /// yet. Use the returned future for the result and this for quiescence.
  bool WaitForIdle(std::chrono::milliseconds timeout) const;

  /// Refuses further decisions: a submitted-but-unfinished decision is answered
  /// as Closed when it returns, and nothing is staged. Already-prepared commands
  /// are invalidated, so they cannot be executed after the close.
  ///
  /// Closing does not abort an HTTP request that is already running; the
  /// destructor joins the worker, so destroying an agent can wait for the
  /// provider's timeout.
  void Close();

 private:
  CommandRegistrationResult RegisterCommandErased(RegisteredCommand command);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace llm
