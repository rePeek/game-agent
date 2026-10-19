#pragma once

#include "llm/core/request.h"
#include "llm/core/result.h"

#include <string>

namespace llm {

/// What a provider can actually guarantee. Callers and the validator use this
/// to refuse requirements instead of assuming them: a provider that cannot
/// force tool calling or enforce a schema must not be credited with doing so.
struct ProviderCapabilities {
  bool multiple_tool_calls = true;  // one completion may request several tools
  bool forced_tool_choice = false;  // honours tool_choice = required
  bool strict_tool_schema = false;  // enforces the parameter schema server-side
  bool tool_result_history =
      true;  // accepts tool results in the request context
};

/// The whole provider surface: one request in, one validated batch out.
///
/// Implementations must not execute tools, must not retry into a conversation
/// and must never turn unvalidated output into CompletionStatus::Success.
class IModelProvider {
 public:
  virtual ~IModelProvider() = default;

  virtual std::string Name() const = 0;
  virtual ProviderCapabilities Capabilities() const { return {}; }
  virtual CompletionResult Complete(const ModelRequest& request) = 0;
};

}  // namespace llm
