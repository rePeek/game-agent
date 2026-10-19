#pragma once

#include "enums.h"
#include "message.h"
#include "usage.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace ai {

using JsonValue = nlohmann::json;

struct Tool {
  std::string description;
  JsonValue parameters_schema;

  Tool() = default;

  Tool(std::string desc, JsonValue schema)
      : description(std::move(desc)), parameters_schema(std::move(schema)) {}
};

using ToolSet = std::map<std::string, Tool>;

/// Tool choice options
enum class ToolChoiceType {
  kAuto,      ///< Model decides whether and which tools to call (default)
  kRequired,  ///< Model must call a tool (any available tool)
  kNone,      ///< Model must not call any tools
  kSpecific   ///< Model must call a specific tool
};

struct ToolChoice {
  ToolChoiceType type = ToolChoiceType::kAuto;
  std::optional<std::string> tool_name;  ///< Required when type is kSpecific

  static ToolChoice auto_choice() {
    return ToolChoice{ToolChoiceType::kAuto, std::nullopt};
  }
  static ToolChoice required() {
    return ToolChoice{ToolChoiceType::kRequired, std::nullopt};
  }
  static ToolChoice none() {
    return ToolChoice{ToolChoiceType::kNone, std::nullopt};
  }
  static ToolChoice specific(const std::string& name) {
    return ToolChoice{ToolChoiceType::kSpecific, name};
  }

  std::string to_string() const {
    switch (type) {
      case ToolChoiceType::kAuto:
        return "auto";
      case ToolChoiceType::kRequired:
        return "required";
      case ToolChoiceType::kNone:
        return "none";
      case ToolChoiceType::kSpecific:
        return "specific(" + tool_name.value_or("") + ")";
      default:
        return "unknown";
    }
  }
};

struct ToolCall {
  std::string id;
  std::string tool_name;
  JsonValue arguments;

  ToolCall(std::string call_id, std::string name, JsonValue args)
      : id(std::move(call_id)),
        tool_name(std::move(name)),
        arguments(std::move(args)) {}

  ToolCall() = default;

  bool is_valid() const { return !id.empty() && !tool_name.empty(); }

  std::string to_string() const {
    return "ToolCall{id='" + id + "', tool='" + tool_name +
           "', args=" + arguments.dump() + "}";
  }
};

/// Helper to build a tool definition. Executing it stays with the caller.
inline Tool create_tool_schema(const std::string& description,
                               const JsonValue& parameters_schema) {
  return Tool(description, parameters_schema);
}

/// Helper to create JSON schema for simple object parameters
/// Usage: auto schema = create_object_schema({{"location", "string"}, {"units",
/// "string"}});
inline JsonValue create_object_schema(
    const std::map<std::string, std::string>& properties) {
  JsonValue schema = {{"type", "object"},
                      {"properties", JsonValue::object()},
                      {"required", JsonValue::array()}};

  for (const auto& [name, type] : properties) {
    schema["properties"][name] = {{"type", type}};
    schema["required"].push_back(name);
  }

  return schema;
}

}  // namespace ai