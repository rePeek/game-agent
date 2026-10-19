#include "llm/commands/prepared_command.hpp"

#include <exception>
#include <string>
#include <utility>

namespace llm {

CommandResult PreparedCommand::Invoke() && {
  CommandResult result;
  result.command_call_id = call_id_;
  result.name = name_;

  if (invoked_) {
    result.outcome.success = false;
    result.outcome.error_message = "command was already invoked";
    return result;
  }
  if (invoke_ == nullptr) {
    result.outcome.success = false;
    result.outcome.error_message = "command has no bound callback";
    return result;
  }
  if (still_valid_ && !still_valid_()) {
    result.outcome.success = false;
    result.outcome.error_message =
        "command is no longer valid (the agent was closed or the decision was "
        "discarded)";
    invoked_ = true;
    invoke_ = nullptr;
    return result;
  }

  invoked_ = true;
  std::function<CommandOutcome()> callback = std::move(invoke_);
  invoke_ = nullptr;
  try {
    result.outcome = callback();
  } catch (const std::exception& error) {
    result.outcome.success = false;
    result.outcome.error_message = std::string("command threw: ") + error.what();
  } catch (...) {
    result.outcome.success = false;
    result.outcome.error_message = "command threw an unknown error";
  }
  return result;
}

}  // namespace llm
