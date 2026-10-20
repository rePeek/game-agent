#pragma once

/// One decision stream: an agent owns its provider, model, prompt, registered
/// commands and the history of its decisions.
///
/// The game creates an agent, registers the commands the model may call,
/// updates it when the world changes, starts a decision asynchronously, polls
/// for the result at a safe point in its own tick, binds the returned calls to
/// C++ callbacks, runs them and reports the results back. The SDK never runs a
/// command, never advances the conversation on its own and never decides when
/// to ask again.
///
/// @par Thread ownership
///
/// Every public method must be called from the thread that created the agent —
/// its **owner thread**. The SDK keeps no cross-thread state for the Agent: the
/// model request runs on its own thread, but that thread only calls the
/// provider and hands back a CompletionResult through a future. All Agent state
/// (provider, registry, context, pending decision, history) is read and written
/// only by the owner thread.
///
/// Two consequences the caller must respect:
///  * `PollDecision()` is how a finished request is collected; it never blocks,
///    and it is the only place a pending decision is created.
///  * a provider's logger may be invoked from the model thread. A logger must
///    therefore not call back into an Agent; capture what you need into a
///    thread-safe sink instead.
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
#include <string_view>
#include <utility>
#include <vector>

#include "llm/commands/prepared_command.hpp"
#include "llm/commands/registry.hpp"

namespace llm {

/// Agent-level outcome of a completed decision. `completion` is only meaningful
/// when this is Completed.
enum class DecisionStatus {
  Completed,  // the provider answered; branch on completion.status
  Closed,     // the agent was closed while the request was in flight
};

struct DecisionResult {
  /// 0 when no decision was staged (a provider-level failure).
  std::uint64_t decision_id = 0;
  /// Revision of the context this decision was taken from.
  std::string context_revision;
  DecisionStatus status = DecisionStatus::Completed;
  CompletionResult completion;
};

/// Why a decision could not be started.
enum class StartStatus {
  Started,
  NotConfigured,    // no provider or no model has been set yet
  Busy,             // a request is in flight or a decision awaits results
  ContextTooLarge,  // the request would exceed the input token budget
  Closed,           // the agent was closed
  Failed,           // the model call could not be launched
};

struct StartOutcome {
  StartStatus status = StartStatus::Started;
  /// Revision the request was built from (empty when nothing was started).
  std::string context_revision;
  std::string error;

  bool ok() const { return status == StartStatus::Started; }
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
/// The Agent is single-threaded by ownership: see @ref AgentConfig and the
/// class comment above for the contract.
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
  /// Installs the provider. Refused while a decision is outstanding, so a call
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

  /// Removes a registered command. Refused while a decision is outstanding.
  CommandRegistrationResult UnregisterCommand(std::string_view name);

  /// Removes every registered command.
  CommandRegistrationResult ClearCommands();

  /// The definitions currently offered to the model, in registration order.
  std::vector<CommandDefinition> Commands() const;

  // --- context ------------------------------------------------------------
  /// Records the latest world state, recent-history window and decision
  /// framing. The request is snapshotted when DecideAsync() is called, so an
  /// update never changes a request that has already been sent.
  void UpdateContext(ContextUpdate update);

  /// Revision of the newest context update this agent received.
  std::string current_revision() const;
  /// A copy of the message history, safe to inspect without holding anything.
  Context ContextSnapshot() const;

  // --- decisions ----------------------------------------------------------
  /// Starts one decision from the context recorded so far: renders it, asks the
  /// provider exactly once on a model thread, and returns immediately.
  ///
  /// It never runs a command callback and never blocks on the model. The
  /// result is collected with PollDecision(). Returns NotConfigured/Busy/
  /// ContextTooLarge/Closed without starting anything.
  StartOutcome DecideAsync();

  /// Collects a finished decision, or returns std::nullopt when the model has
  /// not answered yet. Non-blocking.
  ///
  /// On success the decision is staged as pending (a decision id is assigned)
  /// and must then be prepared, executed and recorded or discarded. On a
  /// provider-level failure the result carries the error and nothing is
  /// staged.
  std::optional<DecisionResult> PollDecision();

  /// Binds a pending decision's command calls to their callbacks, without
  /// running anything. The whole batch is prepared or nothing is: a single
  /// undecodable call fails the batch so the game never executes half a
  /// decision.
  PrepareResult PrepareCommands(std::uint64_t decision_id);

  /// Confirms a pending decision with the results the game executed. The
  /// results must cover every command call of that decision, exactly once, and
  /// name the same commands; otherwise nothing is recorded and the decision
  /// stays pending. On success the whole decision enters the history
  /// atomically.
  RecordOutcome RecordCommandResults(std::uint64_t decision_id,
                                     std::vector<CommandResult> results);

  /// Forgets a pending decision without recording it (a stale revision, a
  /// rolled-back transaction, something the game chose not to apply).
  bool DiscardDecision(std::uint64_t decision_id);

  // --- state --------------------------------------------------------------
  /// True while a request is in flight or a decision is awaiting results.
  bool IsBusy() const;
  bool IsClosed() const;
  std::size_t InFlightDecisionCount() const;
  std::size_t PendingDecisionCount() const;
  /// Waits until the model call has finished (or none is in flight), up to the
  /// timeout. It does not collect the result: call PollDecision() afterwards.
  bool WaitForIdle(std::chrono::milliseconds timeout) const;

  /// Refuses further decisions and invalidates already-prepared commands. A
  /// request that is already running is not aborted; it is still collected by
  /// PollDecision() (answered as Closed) or waited out when the agent is
  /// destroyed.
  void Close();

 private:
  CommandRegistrationResult RegisterCommandErased(RegisteredCommand command);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace llm
