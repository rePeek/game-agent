#pragma once

#include "llm/core/request.h"
#include "llm/openai.h"

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
///  * the token limit field and the sampling controls follow the endpoint's
///    ParameterDialect: reasoning-first families take `max_completion_tokens`
///    and reject temperature/top_p/seed on Chat Completions.
BuildOutcome BuildChatCompletionRequest(
    const ModelRequest& request,
    ParameterDialect dialect = ParameterDialect::Auto);

/// Wire spelling of a role ("system"/"user"/"assistant"/"tool").
const char* WireRole(Role role);

}  // namespace detail
}  // namespace openai
}  // namespace llm
