#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "llm/commands/parameter.hpp"

namespace llm {
namespace {

/// Recursively makes an object schema acceptable to OpenAI's strict function
/// calling: every declared property must appear in `required` (optional
/// parameters are expressed by a nullable type instead).
void MakeStrict(nlohmann::json& schema) {
  if (!schema.is_object()) {
    return;
  }
  // `default` is not part of OpenAI's strict subset; the SDK applies defaults
  // itself when a value is omitted, so dropping it changes nothing observable.
  schema.erase("default");
  if (schema.contains("properties") && schema["properties"].is_object()) {
    nlohmann::json required = nlohmann::json::array();
    for (auto it = schema["properties"].begin();
         it != schema["properties"].end(); ++it) {
      required.push_back(it.key());
      MakeStrict(it.value());
    }
    schema["required"] = std::move(required);
    schema["additionalProperties"] = false;
  }
  if (schema.contains("items") && schema["items"].is_object()) {
    MakeStrict(schema["items"]);
  }
  for (const char* keyword : {"anyOf", "oneOf", "allOf"}) {
    if (schema.contains(keyword) && schema[keyword].is_array()) {
      for (nlohmann::json& alternative : schema[keyword]) {
        MakeStrict(alternative);
      }
    }
  }
}

}  // namespace

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
    // An optional parameter is nullable rather than absent-from-required-
    // and-non-nullable: the same schema then works for strict providers and
    // for the SDK's own validator, and an omitted value can still be null.
    if (parameter.required) {
      schema["type"] = ToString(parameter.type);
    } else {
      schema["type"] =
          nlohmann::json::array({ToString(parameter.type), "null"});
    }
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
      if (!parameter.required) {
        allowed.push_back(nullptr);  // an optional value may be null
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

std::string ToStrictSchema(const std::string& schema_json) {
  nlohmann::json schema = nlohmann::json::parse(schema_json, nullptr, false);
  if (schema.is_discarded() || !schema.is_object()) {
    return schema_json;
  }
  MakeStrict(schema);
  return schema.dump();
}

}  // namespace llm
