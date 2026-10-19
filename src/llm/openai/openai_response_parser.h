#pragma once

#include "llm/core/result.h"
#include "llm/core/types.h"

#include <string>
#include <vector>

namespace llm {
namespace openai {
namespace detail {

/// Wire-level view of one Chat Completions response, before the provider maps
/// it onto CompletionStatus.
struct ParsedCompletion {
  bool ok = false;    // false: the body could not be read as a completion
  std::string error;  // set when ok == false
  std::string model;
  std::string finish_reason;  // raw wire value ("stop", "length", ...)
  std::string assistant_text;
  std::string refusal;  // non-empty when the model refused the request
  std::vector<ToolCall> tool_calls;
  Usage usage;
};

/// Parses a Chat Completions response body.
///
/// Tool call ids, order and argument text are preserved as returned. A
/// conforming server sends `arguments` as a JSON string; some compatible
/// servers send an object, which is accepted and re-serialized so the value can
/// still be carried as text.
ParsedCompletion ParseChatCompletion(const std::string& body);

/// Maps a wire finish reason onto the core enum.
FinishReason MapFinishReason(const std::string& wire_value);

/// Extracts a human-readable message from an error body ({"error": {...}}),
/// falling back to the raw body.
std::string ExtractErrorMessage(const std::string& body);

}  // namespace detail
}  // namespace openai
}  // namespace llm
