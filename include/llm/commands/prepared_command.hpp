#pragma once

/// A command that has been prepared but not executed.
///
/// Preparing decodes the model's arguments and binds them to the registered
/// callback, but never runs it. The game decides where in its own loop — a tick
/// boundary, a transaction, a replay-safe point — to call `Invoke()`.

#include "llm/core/types.h"

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace llm {

/// One bound command call.
///
/// Move-only, and `Invoke()` succeeds exactly once: a prepared command cannot be
/// run twice by accident. `Invoke()` is an rvalue operation, which makes the
/// "consume it to run it" intent explicit at the call site:
///
///     CommandResult result = std::move(command).Invoke();
class PreparedCommand {
 public:
  PreparedCommand() = default;
  PreparedCommand(std::string call_id,
                  std::string name,
                  std::function<CommandOutcome()> invoke,
                  std::function<bool()> still_valid = nullptr)
      : call_id_(std::move(call_id)),
        name_(std::move(name)),
        invoke_(std::move(invoke)),
        still_valid_(std::move(still_valid)) {}

  PreparedCommand(PreparedCommand&&) noexcept = default;
  PreparedCommand& operator=(PreparedCommand&&) noexcept = default;
  PreparedCommand(const PreparedCommand&) = delete;
  PreparedCommand& operator=(const PreparedCommand&) = delete;

  const std::string& Id() const { return call_id_; }
  const std::string& Name() const { return name_; }

  /// True while this command is bound and has not been invoked.
  bool ready() const { return invoke_ != nullptr && !invoked_; }

  /// Runs the bound callback once and reports the result.
  ///
  /// Never throws: a callback that throws is reported as a failed outcome, and
  /// a command that is no longer valid (for example after the agent was closed)
  /// is refused instead of run.
  CommandResult Invoke() &&;

 private:
  std::string call_id_;
  std::string name_;
  std::function<CommandOutcome()> invoke_;
  std::function<bool()> still_valid_;
  bool invoked_ = false;
};

using PreparedCommandBatch = std::vector<PreparedCommand>;

/// Why a batch could or could not be prepared.
enum class PrepareStatus {
  Ok,
  UnknownDecision,      // no pending decision with that id
  NotPrepared,          // the pending decision is not a command-call decision
  RegistryChanged,      // commands changed since the decision was taken
  InvalidArguments,     // at least one call could not be decoded
  NoCommands,           // the decision contained no command calls
  Closed,               // the agent was closed
};

struct PrepareResult {
  PrepareStatus status = PrepareStatus::InvalidArguments;
  std::vector<PreparedCommand> commands;
  std::string error;

  bool ok() const { return status == PrepareStatus::Ok; }
};

}  // namespace llm
