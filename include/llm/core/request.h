#pragma once

#include "llm/core/message.h"
#include "llm/core/types.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace llm {

/// How the provider should pick tools for a request.
enum class ToolChoiceMode {
  Auto,      // the model decides whether to call a tool
  Required,  // the model must call at least one tool
  None,      // tools are not offered for this request
  Specific,  // the model must call `tool_name`
};

struct ToolChoice {
  ToolChoiceMode mode = ToolChoiceMode::Required;
  std::string tool_name;  // only meaningful for ToolChoiceMode::Specific

  static ToolChoice Auto();
  static ToolChoice Required();
  static ToolChoice None();
  static ToolChoice Specific(std::string tool_name);

  /// Provider-independent label: "auto", "required", "none" or the tool name.
  std::string ToString() const;
};

/// A tool the model may call.
///
/// `parameters_schema_json` is a JSON Schema object serialized as text, so the
/// core does not need a JSON library to carry it. `strict` asks the provider
/// to enforce that schema itself; providers that cannot must report it through
/// ProviderCapabilities instead of pretending the schema is enforced.
struct ToolDefinition {
  std::string name;
  std::string description;
  std::string parameters_schema_json;
  bool strict = false;

  bool IsValid() const { return !name.empty(); }
};

/// One completion request: the caller's context plus the tools it may use.
struct ModelRequest {
  std::string model;
  std::vector<Message> messages;
  std::vector<ToolDefinition> tools;

  /// Forced tool calling is the SDK's contract, so `required` is the default;
  /// callers that also accept plain answers set ToolChoice::Auto().
  ToolChoice tool_choice = ToolChoice::Required();

  std::optional<std::int32_t> max_tokens;
  std::optional<double> temperature;
  std::optional<double> top_p;
  std::optional<std::int64_t> seed;

  /// Upper bound on the number of tool calls a single completion may return.
  /// Enforced by the SDK validator, not by the provider.
  std::optional<std::size_t> max_tool_calls;

  bool HasTools() const { return !tools.empty(); }
  const ToolDefinition* FindTool(const std::string& name) const;
};

}  // namespace llm
