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

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
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

/// What a game callback reports after running one command.
///
/// `content_json` is the command's own output as JSON text (usually an
/// object). `error_message` explains a failure; it is what the model sees in
/// the unified result rendering. A game never fills in the model's call id.
struct CommandOutcome {
  bool success = false;
  std::string content_json;
  std::string error_message;
};

/// One executed command, complete enough to feed back into the transcript:
/// which model call it answers, what command it was, and its outcome.
struct CommandResult {
  std::string command_call_id;
  std::string name;
  CommandOutcome outcome;

  bool IsError() const { return !outcome.success; }
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

/// A deliberately simple token estimate: roughly four characters per token.
///
/// It exists only to bound a request before it is sent and is documented as
/// approximate on purpose; it is not a model-specific tokenizer and callers
/// must not treat it as billing-accurate.
std::size_t EstimateTokens(std::string_view text);

}  // namespace llm
