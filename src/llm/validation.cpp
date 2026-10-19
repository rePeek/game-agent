#include "llm/validation.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace llm {
namespace {

using Json = nlohmann::json;

std::string DescribeType(const Json& value) {
  if (value.is_null()) {
    return "null";
  }
  if (value.is_boolean()) {
    return "boolean";
  }
  if (value.is_number_integer()) {
    return "integer";
  }
  if (value.is_number()) {
    return "number";
  }
  if (value.is_string()) {
    return "string";
  }
  if (value.is_array()) {
    return "array";
  }
  if (value.is_object()) {
    return "object";
  }
  return "unknown";
}

bool MatchesSimpleType(const std::string& type, const Json& value) {
  if (type == "string") {
    return value.is_string();
  }
  if (type == "integer") {
    return value.is_number_integer();
  }
  if (type == "number") {
    return value.is_number();
  }
  if (type == "boolean") {
    return value.is_boolean();
  }
  if (type == "object") {
    return value.is_object();
  }
  if (type == "array") {
    return value.is_array();
  }
  if (type == "null") {
    return value.is_null();
  }
  return true;  // unknown type name: do not invent a failure
}

bool MatchesType(const Json& schema, const Json& value) {
  if (!schema.contains("type")) {
    return true;
  }
  const auto& type = schema["type"];
  if (type.is_string()) {
    return MatchesSimpleType(type.get<std::string>(), value);
  }
  if (type.is_array()) {
    for (const auto& alternative : type) {
      if (alternative.is_string() &&
          MatchesSimpleType(alternative.get<std::string>(), value)) {
        return true;
      }
    }
    return false;
  }
  return true;
}

/// Validates `value` against `schema`, appending human-readable problems to
/// `errors`. Recursion is bounded by the data, not by the schema.
void ValidateAgainstSchema(const Json& schema,
                           const Json& value,
                           const std::string& path,
                           std::vector<std::string>& errors,
                           std::size_t depth) {
  if (!schema.is_object() || depth > 16) {
    return;
  }

  if (!MatchesType(schema, value)) {
    std::string expected = "the declared type";
    if (schema["type"].is_string()) {
      expected = "'" + schema["type"].get<std::string>() + "'";
    }
    errors.push_back(path + " must be " + expected + " but is '" +
                     DescribeType(value) + "'");
    return;  // further checks would only produce noise
  }

  if (schema.contains("enum") && schema["enum"].is_array()) {
    bool found = false;
    for (const auto& allowed : schema["enum"]) {
      if (allowed == value) {
        found = true;
        break;
      }
    }
    if (!found) {
      errors.push_back(path + " is not one of the allowed values (" +
                       schema["enum"].dump() + ")");
    }
  }

  if (schema.contains("const") && schema["const"] != value) {
    errors.push_back(path + " must be " + schema["const"].dump());
  }

  for (const char* keyword : {"anyOf", "oneOf"}) {
    if (!schema.contains(keyword) || !schema[keyword].is_array()) {
      continue;
    }
    std::size_t matches = 0;
    for (const auto& alternative : schema[keyword]) {
      std::vector<std::string> ignored;
      const std::size_t before_errors = ignored.size();
      ValidateAgainstSchema(alternative, value, path, ignored, depth + 1);
      (void)before_errors;
      if (ignored.empty()) {
        ++matches;
      }
    }
    const bool satisfied =
        std::string(keyword) == "anyOf" ? matches >= 1 : matches == 1;
    if (!satisfied) {
      errors.push_back(path + " matches no allowed shape (" + keyword + ")");
    }
  }

  if (schema.contains("allOf") && schema["allOf"].is_array()) {
    for (const auto& alternative : schema["allOf"]) {
      ValidateAgainstSchema(alternative, value, path, errors, depth + 1);
    }
  }

  if (value.is_object()) {
    const Json empty = Json::object();
    const Json& properties =
        schema.contains("properties") && schema["properties"].is_object()
            ? schema["properties"]
            : empty;

    if (schema.contains("required") && schema["required"].is_array()) {
      for (const auto& required : schema["required"]) {
        if (!required.is_string()) {
          continue;
        }
        const std::string name = required.get<std::string>();
        if (!value.contains(name)) {
          errors.push_back(path + " is missing required property '" + name +
                           "'");
        }
      }
    }

    for (auto it = value.begin(); it != value.end(); ++it) {
      const std::string child = path + "." + it.key();
      if (properties.contains(it.key())) {
        ValidateAgainstSchema(properties[it.key()], it.value(), child, errors,
                              depth + 1);
        continue;
      }
      if (schema.contains("additionalProperties")) {
        const auto& additional = schema["additionalProperties"];
        if (additional.is_boolean() && !additional.get<bool>()) {
          errors.push_back(path + " has unexpected property '" + it.key() +
                           "'");
        } else if (additional.is_object()) {
          ValidateAgainstSchema(additional, it.value(), child, errors,
                                depth + 1);
        }
      }
    }
  }

  if (value.is_array()) {
    if (schema.contains("minItems") &&
        schema["minItems"].is_number_unsigned() &&
        value.size() < schema["minItems"].get<std::size_t>()) {
      errors.push_back(path + " needs at least " +
                       std::to_string(schema["minItems"].get<std::size_t>()) +
                       " items");
    }
    if (schema.contains("maxItems") &&
        schema["maxItems"].is_number_unsigned() &&
        value.size() > schema["maxItems"].get<std::size_t>()) {
      errors.push_back(path + " allows at most " +
                       std::to_string(schema["maxItems"].get<std::size_t>()) +
                       " items");
    }
    if (schema.contains("items") && schema["items"].is_object()) {
      for (std::size_t i = 0; i < value.size(); ++i) {
        ValidateAgainstSchema(schema["items"], value[i],
                              path + "[" + std::to_string(i) + "]", errors,
                              depth + 1);
      }
    }
  }

  if (value.is_number()) {
    if (schema.contains("minimum") && schema["minimum"].is_number() &&
        value.get<double>() < schema["minimum"].get<double>()) {
      errors.push_back(path + " must be >= " + schema["minimum"].dump());
    }
    if (schema.contains("maximum") && schema["maximum"].is_number() &&
        value.get<double>() > schema["maximum"].get<double>()) {
      errors.push_back(path + " must be <= " + schema["maximum"].dump());
    }
  }

  if (value.is_string()) {
    const std::string text = value.get<std::string>();
    if (schema.contains("minLength") &&
        schema["minLength"].is_number_unsigned() &&
        text.size() < schema["minLength"].get<std::size_t>()) {
      errors.push_back(path + " is shorter than " +
                       std::to_string(schema["minLength"].get<std::size_t>()) +
                       " characters");
    }
    if (schema.contains("maxLength") &&
        schema["maxLength"].is_number_unsigned() &&
        text.size() > schema["maxLength"].get<std::size_t>()) {
      errors.push_back(path + " is longer than " +
                       std::to_string(schema["maxLength"].get<std::size_t>()) +
                       " characters");
    }
  }
}

}  // namespace

std::string ValidationIssue::ToString() const {
  std::string where = "tool call #" + std::to_string(index);
  if (!tool_name.empty()) {
    where += " (" + tool_name + ")";
  }
  return where + ": " + message;
}

std::string ValidationReport::Summary() const {
  std::string summary;
  for (const auto& issue : issues) {
    if (!summary.empty()) {
      summary += "; ";
    }
    summary += issue.ToString();
  }
  return summary;
}

JsonSchemaToolCallValidator::JsonSchemaToolCallValidator(
    ValidationLimits limits)
    : limits_(std::move(limits)) {}

ValidationReport JsonSchemaToolCallValidator::Validate(
    const ModelRequest& request,
    const std::vector<ToolCall>& tool_calls) const {
  ValidationReport report;

  const std::size_t effective_max = limits_.max_tool_calls.has_value()
                                        ? limits_.max_tool_calls.value()
                                        : request.max_tool_calls.value_or(0);

  if (tool_calls.size() < limits_.min_tool_calls) {
    ValidationIssue issue;
    issue.message = "expected at least " +
                    std::to_string(limits_.min_tool_calls) +
                    " tool call(s), got " + std::to_string(tool_calls.size());
    report.issues.push_back(std::move(issue));
  }
  if (effective_max > 0 && tool_calls.size() > effective_max) {
    ValidationIssue issue;
    issue.message = "expected at most " + std::to_string(effective_max) +
                    " tool call(s), got " + std::to_string(tool_calls.size());
    report.issues.push_back(std::move(issue));
  }

  for (std::size_t i = 0; i < tool_calls.size(); ++i) {
    const ToolCall& call = tool_calls[i];
    const std::size_t index = i;
    const std::string name = call.name;

    auto add_issue = [&report, index, &name](std::string message) {
      ValidationIssue issue;
      issue.index = index;
      issue.tool_name = name;
      issue.message = std::move(message);
      report.issues.push_back(std::move(issue));
    };

    if (limits_.require_ids && call.id.empty()) {
      add_issue("has no id, so its result could not be sent back");
    }
    if (call.name.empty()) {
      add_issue("has no tool name");
      continue;
    }

    const ToolDefinition* tool =
        limits_.require_known_tools ? request.FindTool(call.name) : nullptr;
    if (limits_.require_known_tools && tool == nullptr) {
      add_issue("is not one of the tools offered in this request");
      continue;
    }

    const Json arguments = Json::parse(call.arguments_json, nullptr, false);
    if (arguments.is_discarded()) {
      add_issue("arguments are not valid JSON");
      continue;
    }
    if (limits_.require_arguments_object && !arguments.is_object()) {
      add_issue("arguments are '" + DescribeType(arguments) +
                "' but must be a JSON object");
      continue;
    }

    if (!limits_.validate_schema || tool == nullptr) {
      continue;
    }
    const Json schema =
        Json::parse(tool->parameters_schema_json, nullptr, false);
    if (schema.is_discarded()) {
      add_issue("the tool's parameters schema is not valid JSON");
      continue;
    }
    std::vector<std::string> schema_errors;
    ValidateAgainstSchema(schema, arguments, "", schema_errors, 0);
    for (auto& error : schema_errors) {
      if (!error.empty() && error.front() == '.') {
        error.erase(error.begin());
      }
      add_issue(std::move(error));
    }
  }

  report.ok = report.issues.empty();
  return report;
}

std::shared_ptr<const ToolCallValidator> DefaultValidator() {
  static const auto validator =
      std::make_shared<const JsonSchemaToolCallValidator>();
  return validator;
}

}  // namespace llm
