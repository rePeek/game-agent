#pragma once

/// Umbrella header for the llm:: core.
///
/// The core is deliberately free of third-party dependencies: it carries the
/// request, message, tool-call and completion vocabulary as plain STL types.
/// Providers (which do own JSON/HTTP/TLS) translate to and from it.

#include "provider.h"
#include "request.h"
#include "result.h"
#include "transport.h"
#include "types.h"

/// AI SDK C++ core — structured tool calling without chat bookkeeping.
///
/// One request in, one validated batch of tool calls out:
///
/// ```cpp
/// llm::ModelRequest request;
/// request.model = "gpt-5.6";
/// request.messages.push_back(llm::Message::System("You are a coach."));
/// request.messages.push_back(llm::Message::User("Opponent is pressing
/// high.")); request.tools.push_back(llm::ToolDefinition{
///     .name = "set_pressing_intensity",
///     .description = "Raise or lower the pressing intensity",
///     .parameters_schema_json =
///     R"({"type":"object","properties":{"level":{"type":"integer"}},"required":["level"]})",
///     .strict = true,
/// });
///
/// llm::CompletionResult result = provider.Complete(request);
/// if (result.IsSuccess()) {
///   for (const auto& call : result.tool_calls) {
///     // The caller validates against its own registry and executes.
///   }
/// }
/// ```
namespace llm {}
