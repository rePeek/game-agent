#pragma once

/// Conversation vocabulary of the llm:: core.
///
/// The tool vocabulary stays part of the data model: an assistant turn may
/// carry `tool_calls`, and a tool turn carries the `tool_call_id` it answers.
/// What the SDK does not do is maintain or advance such a conversation on the
/// caller's behalf — llm::context does that, explicitly, per session.

#include "llm/core/types.h"

#include <string>
#include <vector>

namespace llm {

/// A message in the caller-supplied context.
struct Message {
  Role role = Role::User;
  std::string content;
  std::vector<ToolCall> tool_calls;  // assistant turns
  std::string tool_call_id;          // tool turns
  std::string name;                  // optional participant/tool name

  static Message System(std::string content);
  static Message User(std::string content);
  static Message Assistant(std::string content);
  /// Assistant turn that only requests tools, with no natural-language text.
  static Message AssistantToolCalls(std::vector<ToolCall> tool_calls);
  /// Tool turn answering `result.tool_call_id`.
  static Message ToolResult(const ToolResult& result);

  bool HasToolCalls() const { return !tool_calls.empty(); }
};

}  // namespace llm
