#include "llm/commands/parameter.hpp"

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace llm {

const char* ToString(CommandValueType type) {
  switch (type) {
    case CommandValueType::Boolean:
      return "boolean";
    case CommandValueType::Integer:
      return "integer";
    case CommandValueType::Number:
      return "number";
    case CommandValueType::String:
      return "string";
    case CommandValueType::Array:
      return "array";
    case CommandValueType::Object:
      return "object";
  }
  return "string";
}

std::string BuildCommandSchema(
    const std::vector<ParameterDescriptor>& parameters) {
  nlohmann::json properties = nlohmann::json::object();
  nlohmann::json required = nlohmann::json::array();

  for (const ParameterDescriptor& parameter : parameters) {
    nlohmann::json schema = nlohmann::json::object();
    schema["type"] = ToString(parameter.type);
    if (!parameter.description.empty()) {
      schema["description"] = parameter.description;
    }
    if (parameter.default_json.has_value()) {
      nlohmann::json parsed =
          nlohmann::json::parse(*parameter.default_json, nullptr, false);
      if (!parsed.is_discarded()) {
        schema["default"] = std::move(parsed);
      }
    }
    if (parameter.minimum.has_value()) {
      schema["minimum"] = *parameter.minimum;
    }
    if (parameter.maximum.has_value()) {
      schema["maximum"] = *parameter.maximum;
    }
    if (parameter.min_length.has_value()) {
      schema["minLength"] = *parameter.min_length;
    }
    if (parameter.max_length.has_value()) {
      schema["maxLength"] = *parameter.max_length;
    }
    if (parameter.min_items.has_value()) {
      schema["minItems"] = *parameter.min_items;
    }
    if (parameter.max_items.has_value()) {
      schema["maxItems"] = *parameter.max_items;
    }
    if (!parameter.enum_json.empty()) {
      nlohmann::json allowed = nlohmann::json::array();
      for (const std::string& text : parameter.enum_json) {
        nlohmann::json parsed = nlohmann::json::parse(text, nullptr, false);
        // The values were serialized by Param<T>, so a discarded parse here
        // would be an SDK bug; skipping keeps the schema valid regardless.
        if (!parsed.is_discarded()) {
          allowed.push_back(std::move(parsed));
        }
      }
      schema["enum"] = std::move(allowed);
    }

    properties[parameter.name] = std::move(schema);
    if (parameter.required) {
      required.push_back(parameter.name);
    }
  }

  nlohmann::json root = nlohmann::json::object();
  root["type"] = "object";
  root["properties"] = std::move(properties);
  root["required"] = std::move(required);
  root["additionalProperties"] = false;
  return root.dump();
}

}  // namespace llm
