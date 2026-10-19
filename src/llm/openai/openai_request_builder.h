#pragma once

#include "llm/request.h"

#include <string>

#include <nlohmann/json.hpp>

namespace llm {
namespace openai {
namespace detail {

struct BuildOutcome {
  bool ok = false;
  nlohmann::json body;
  std::string error;  // set when ok == false
};

/// Translates a ModelRequest into an OpenAI Chat Completions request body.
///
/// Notes on intent:
///  * tool call arguments are emitted as the raw JSON *text* the core carries,
///    never re-serialized from a parsed object;
///  * `tool_choice` is forwarded as configured and the provider never adds a
///    `parallel_tool_calls: false` cap;
///  * temperature/top_p are omitted for the gpt-5 family, which rejects
///    sampling controls on Chat Completions;
///  * `max_tokens` is used rather than `max_completion_tokens` for
///    compatibility with OpenAI-compatible servers.
BuildOutcome BuildChatCompletionRequest(const ModelRequest& request);

/// Wire spelling of a role ("system"/"user"/"assistant"/"tool").
const char* WireRole(Role role);

}  // namespace detail
}  // namespace openai
}  // namespace llm
