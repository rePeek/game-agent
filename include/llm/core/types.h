#pragma once

/// STL-only value types of the llm:: core.
///
/// Nothing in this header may pull in a third-party dependency. The core is
/// built without JSON, HTTP or TLS so that it can be embedded anywhere; JSON
/// text crosses its boundary as std::string and is parsed by the layers that
/// actually own a JSON library (request building, response parsing and the
/// command-call validator).
///
/// Vocabulary: the SDK's unit of work is a *command* — a function the game
/// registers and the model asks to run. The OpenAI wire protocol still calls
/// these "tools"; that translation lives entirely in the provider.

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

/// One command invocation requested by the model.
///
/// `arguments_json` keeps the argument object exactly as the provider returned
/// it, serialized as text: ids, order and formatting are preserved, and the
/// core never parses it. Validation (which does parse it) happens in a layer
/// that is allowed to depend on JSON.
struct CommandCall {
  std::string id;
  std::string name;
  std::string arguments_json;

  bool IsValid() const { return !name.empty(); }
  std::string ToString() const;
};

/// Outcome of executing a command call, kept so the caller can feed it back as
/// context for a later request. `content_json` is the command's own output (or
/// the model-visible error rendering).
struct CommandResult {
  std::string command_call_id;
  std::string name;
  std::string content_json;  // command output as JSON text, usually an object
  bool is_error = false;
};

/// Why the model stopped generating.
enum class FinishReason {
  Unknown,
  Stop,
  Length,
  ContentFilter,
  CommandCalls,
  Error,
};

/// Diagnostic spellings of the core enums. Providers decide the actual wire
/// values; these are only for logs, errors and test failure messages.
std::string ToString(Role role);
std::string ToString(FinishReason reason);

}  // namespace llm
