#pragma once

/// Conversation vocabulary of the llm:: core.
///
/// The command vocabulary stays part of the data model: an assistant turn may
/// carry `command_calls`, and a tool-role turn carries the `command_call_id` it
/// answers. What the SDK does not do is maintain or advance such a conversation
/// on the caller's behalf — llm::context does that, explicitly, per agent.

#include "llm/core/types.h"

#include <string>
#include <vector>

namespace llm {

/// A message in the caller-supplied context.
struct Message {
  Role role = Role::User;
  std::string content;
  std::vector<CommandCall> command_calls;  // assistant turns
  std::string command_call_id;             // tool-role turns
  std::string name;                        // optional participant/command name

  static Message System(std::string content);
  static Message User(std::string content);
  static Message Assistant(std::string content);
  /// Assistant turn that only requests commands, with no natural-language text.
  static Message AssistantCommandCalls(std::vector<CommandCall> calls);
  /// Tool-role turn answering `result.command_call_id`.
  static Message CommandResultMessage(const CommandResult& result);

  bool HasCommandCalls() const { return !command_calls.empty(); }
};

}  // namespace llm
