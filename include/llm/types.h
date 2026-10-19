#pragma once

/// STL-only value types of the llm:: core.
///
/// Nothing in this header may pull in a third-party dependency. The core is
/// built without JSON, HTTP or TLS so that it can be embedded anywhere; JSON
/// text crosses its boundary as std::string and is parsed by the layers that
/// actually own a JSON library (request building, response parsing and the
/// tool-call validator).

#include <cstdint>
#include <string>
#include <vector>

namespace llm {

/// Roles a message can have in the context the caller builds.
enum class Role {
  System,
  User,
  Assistant,
  Tool,
};

/// Token accounting exactly as the provider reported it. `total_tokens` is
/// kept as reported rather than derived, so provider quirks stay visible.
struct Usage {
  std::int64_t prompt_tokens = 0;
  std::int64_t completion_tokens = 0;
  std::int64_t total_tokens = 0;
};

/// One tool invocation requested by the model.
///
/// `arguments_json` keeps the argument object exactly as the provider returned
/// it, serialized as text: ids, order and formatting are preserved, and the
/// core never parses it. Validation (which does parse it) happens in a layer
/// that is allowed to depend on JSON.
struct ToolCall {
  std::string id;
  std::string name;
  std::string arguments_json;

  bool IsValid() const { return !name.empty(); }
  std::string ToString() const;
};

/// Outcome of executing a tool call, kept so the caller can feed it back as
/// context for a later request.
struct ToolResult {
  std::string tool_call_id;
  std::string name;
  std::string content_json;  // tool output as JSON text, usually an object
  bool is_error = false;
};

/// A message in the caller-supplied context.
///
/// The tool vocabulary stays part of the data model: an assistant turn may
/// carry `tool_calls`, and a tool turn carries the `tool_call_id` it answers.
/// What the SDK no longer does is maintain or advance such a conversation on
/// the caller's behalf.
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

/// Why the model stopped generating.
enum class FinishReason {
  Unknown,
  Stop,
  Length,
  ContentFilter,
  ToolCalls,
  Error,
};

/// Diagnostic spellings of the core enums. Providers decide the actual wire
/// values; these are only for logs, errors and test failure messages.
std::string ToString(Role role);
std::string ToString(FinishReason reason);

}  // namespace llm
