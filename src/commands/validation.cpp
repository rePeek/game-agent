#include "llm/commands/validation.h"

#include "utf8.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace llm {
namespace {

using Json = nlohmann::json;

/// Nesting deeper than this is reported instead of validated, so a pathological
/// schema cannot make the validator recurse without bound.
constexpr std::size_t kMaxSchemaDepth = 32;

enum class ProblemKind {
  Arguments,
  Schema,
};

struct Problem {
  ProblemKind kind;
  std::string message;
};

using Problems = std::vector<Problem>;

void AddProblem(Problems& problems, ProblemKind kind, std::string message) {
  problems.push_back(Problem{kind, std::move(message)});
}

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

enum class TypeMatch {
  Match,
  Mismatch,
  Unsupported,  // the schema names a type this validator does not know
};

TypeMatch MatchesSimpleType(const std::string& type, const Json& value) {
  if (type == "string") {
    return value.is_string() ? TypeMatch::Match : TypeMatch::Mismatch;
  }
  if (type == "integer") {
    return value.is_number_integer() ? TypeMatch::Match : TypeMatch::Mismatch;
  }
  if (type == "number") {
    return value.is_number() ? TypeMatch::Match : TypeMatch::Mismatch;
  }
  if (type == "boolean") {
    return value.is_boolean() ? TypeMatch::Match : TypeMatch::Mismatch;
  }
  if (type == "object") {
    return value.is_object() ? TypeMatch::Match : TypeMatch::Mismatch;
  }
  if (type == "array") {
    return value.is_array() ? TypeMatch::Match : TypeMatch::Mismatch;
  }
  if (type == "null") {
    return value.is_null() ? TypeMatch::Match : TypeMatch::Mismatch;
  }
  return TypeMatch::Unsupported;
}

TypeMatch MatchesType(const Json& schema, const Json& value) {
  if (!schema.contains("type")) {
    return TypeMatch::Match;
  }
  const auto& type = schema["type"];
  if (type.is_string()) {
    return MatchesSimpleType(type.get<std::string>(), value);
  }
  if (type.is_array()) {
    bool saw_unsupported = false;
    for (const auto& alternative : type) {
      if (!alternative.is_string()) {
        saw_unsupported = true;
        continue;
      }
      const TypeMatch match =
          MatchesSimpleType(alternative.get<std::string>(), value);
      if (match == TypeMatch::Match) {
        return TypeMatch::Match;
      }
      if (match == TypeMatch::Unsupported) {
        saw_unsupported = true;
      }
    }
    return saw_unsupported ? TypeMatch::Unsupported : TypeMatch::Mismatch;
  }
  return TypeMatch::Unsupported;
}

/// Keywords this validator implements (constraints) or may safely ignore
/// (annotations). Anything else is refused: Success must mean "checked", so a
/// constraint nobody evaluates may not slip through.
bool IsKnownKeyword(const std::string& keyword) {
  static const char* const kKnown[] = {
      // constraints
      "type",
      "enum",
      "const",
      "anyOf",
      "oneOf",
      "allOf",
      "properties",
      "required",
      "additionalProperties",
      "items",
      "minItems",
      "maxItems",
      "minimum",
      "maximum",
      "minLength",
      "maxLength",
      // annotations (no effect on validation)
      "title",
      "description",
      "default",
      "examples",
      "$schema",
      "$id",
      "$comment",
      "deprecated",
      "readOnly",
      "writeOnly",
  };
  for (const char* known : kKnown) {
    if (keyword == known) {
      return true;
    }
  }
  return false;
}

/// Walks a schema once and reports every keyword that will not be enforced.
void CheckSchemaSupported(const Json& schema,
                          const std::string& path,
                          Problems& problems,
                          std::size_t depth) {
  if (!schema.is_object()) {
    if (!path.empty()) {
      AddProblem(problems, ProblemKind::Schema,
                 path + " is not a JSON Schema object");
    }
    return;
  }
  if (depth > kMaxSchemaDepth) {
    AddProblem(problems, ProblemKind::Schema,
               "schema is nested deeper than " +
                   std::to_string(kMaxSchemaDepth) + " levels");
    return;
  }

  for (auto it = schema.begin(); it != schema.end(); ++it) {
    const std::string keyword = path + "." + it.key();
    if (!IsKnownKeyword(it.key())) {
      AddProblem(
          problems, ProblemKind::Schema,
          "schema uses '" + it.key() + "' which this validator cannot enforce");
      continue;  // do not descend into a constraint we do not understand
    }

    if (it.key() == "type") {
      const auto& type = it.value();
      const auto check_name = [&problems, &keyword](const Json& name) {
        if (!name.is_string()) {
          AddProblem(problems, ProblemKind::Schema,
                     keyword + " must be a string or an array of strings");
          return;
        }
        const std::string text = name.get<std::string>();
        if (text != "string" && text != "integer" && text != "number" &&
            text != "boolean" && text != "object" && text != "array" &&
            text != "null") {
          AddProblem(problems, ProblemKind::Schema,
                     "type '" + text + "' is not supported");
        }
      };
      if (type.is_array()) {
        for (const auto& name : type) {
          check_name(name);
        }
      } else {
        check_name(type);
      }
      continue;
    }

    if (it.key() == "enum") {
      if (!it.value().is_array() || it.value().empty()) {
        AddProblem(problems, ProblemKind::Schema,
                   keyword + " must be a non-empty array");
      }
      continue;
    }

    if (it.key() == "required") {
      if (!it.value().is_array()) {
        AddProblem(problems, ProblemKind::Schema,
                   keyword + " must be an array of property names");
      } else {
        for (const auto& name : it.value()) {
          if (!name.is_string()) {
            AddProblem(problems, ProblemKind::Schema,
                       keyword + " must contain only strings");
            break;
          }
        }
      }
      continue;
    }

    if (it.key() == "minimum" || it.key() == "maximum") {
      if (!it.value().is_number()) {
        AddProblem(problems, ProblemKind::Schema,
                   keyword + " must be a number");
      }
      continue;
    }

    if (it.key() == "minItems" || it.key() == "maxItems" ||
        it.key() == "minLength" || it.key() == "maxLength") {
      if (!it.value().is_number_unsigned()) {
        AddProblem(problems, ProblemKind::Schema,
                   keyword + " must be a non-negative integer");
      }
      continue;
    }

    if (it.key() == "properties") {
      if (!it.value().is_object()) {
        AddProblem(problems, ProblemKind::Schema,
                   keyword + " must be an object");
        continue;
      }
      for (auto property = it.value().begin(); property != it.value().end();
           ++property) {
        CheckSchemaSupported(property.value(), keyword + "." + property.key(),
                             problems, depth + 1);
      }
      continue;
    }

    if (it.key() == "additionalProperties") {
      if (it.value().is_object()) {
        CheckSchemaSupported(it.value(), keyword, problems, depth + 1);
      } else if (!it.value().is_boolean()) {
        AddProblem(problems, ProblemKind::Schema,
                   keyword + " must be a boolean or an object");
      }
      continue;
    }

    if (it.key() == "items") {
      if (it.value().is_object()) {
        CheckSchemaSupported(it.value(), keyword, problems, depth + 1);
      } else {
        AddProblem(
            problems, ProblemKind::Schema,
            keyword + " must be an object (tuple items are not supported)");
      }
      continue;
    }

    if (it.key() == "anyOf" || it.key() == "oneOf" || it.key() == "allOf") {
      if (!it.value().is_array() || it.value().empty()) {
        AddProblem(problems, ProblemKind::Schema,
                   keyword + " must be a non-empty array");
        continue;
      }
      for (std::size_t i = 0; i < it.value().size(); ++i) {
        CheckSchemaSupported(it.value()[i],
                             keyword + "[" + std::to_string(i) + "]", problems,
                             depth + 1);
      }
      continue;
    }
  }
}

/// Validates `value` against `schema`, appending problems.
void ValidateAgainstSchema(const Json& schema,
                           const Json& value,
                           const std::string& path,
                           Problems& problems,
                           std::size_t depth) {
  if (!schema.is_object()) {
    return;
  }
  if (depth > kMaxSchemaDepth) {
    AddProblem(problems, ProblemKind::Schema,
               "schema is nested deeper than " +
                   std::to_string(kMaxSchemaDepth) + " levels");
    return;
  }

  switch (MatchesType(schema, value)) {
    case TypeMatch::Unsupported:
      AddProblem(problems, ProblemKind::Schema,
                 path + " is checked against an unsupported 'type'");
      return;
    case TypeMatch::Mismatch: {
      std::string expected = "the declared type";
      if (schema["type"].is_string()) {
        expected = "'" + schema["type"].get<std::string>() + "'";
      }
      AddProblem(problems, ProblemKind::Arguments,
                 path + " must be " + expected + " but is '" +
                     DescribeType(value) + "'");
      return;  // further checks would only produce noise
    }
    case TypeMatch::Match:
      break;
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
      AddProblem(problems, ProblemKind::Arguments,
                 path + " is not one of the allowed values (" +
                     schema["enum"].dump() + ")");
    }
  }

  if (schema.contains("const") && schema["const"] != value) {
    AddProblem(problems, ProblemKind::Arguments,
               path + " must be " + schema["const"].dump());
  }

  for (const char* keyword : {"anyOf", "oneOf"}) {
    if (!schema.contains(keyword) || !schema[keyword].is_array()) {
      continue;
    }
    std::size_t matches = 0;
    for (const auto& alternative : schema[keyword]) {
      Problems ignored;
      ValidateAgainstSchema(alternative, value, path, ignored, depth + 1);
      if (ignored.empty()) {
        ++matches;
      }
    }
    const bool satisfied =
        std::string(keyword) == "anyOf" ? matches >= 1 : matches == 1;
    if (!satisfied) {
      AddProblem(problems, ProblemKind::Arguments,
                 path + " matches no allowed shape (" + keyword + ")");
    }
  }

  if (schema.contains("allOf") && schema["allOf"].is_array()) {
    for (const auto& alternative : schema["allOf"]) {
      ValidateAgainstSchema(alternative, value, path, problems, depth + 1);
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
          AddProblem(problems, ProblemKind::Arguments,
                     path + " is missing required property '" + name + "'");
        }
      }
    }

    for (auto it = value.begin(); it != value.end(); ++it) {
      const std::string child = path + "." + it.key();
      if (properties.contains(it.key())) {
        ValidateAgainstSchema(properties[it.key()], it.value(), child, problems,
                              depth + 1);
        continue;
      }
      if (schema.contains("additionalProperties")) {
        const auto& additional = schema["additionalProperties"];
        if (additional.is_boolean() && !additional.get<bool>()) {
          AddProblem(problems, ProblemKind::Arguments,
                     path + " has unexpected property '" + it.key() + "'");
        } else if (additional.is_object()) {
          ValidateAgainstSchema(additional, it.value(), child, problems,
                                depth + 1);
        }
      }
    }
  }

  if (value.is_array()) {
    if (schema.contains("minItems") &&
        schema["minItems"].is_number_unsigned() &&
        value.size() < schema["minItems"].get<std::size_t>()) {
      AddProblem(problems, ProblemKind::Arguments,
                 path + " needs at least " +
                     std::to_string(schema["minItems"].get<std::size_t>()) +
                     " items");
    }
    if (schema.contains("maxItems") &&
        schema["maxItems"].is_number_unsigned() &&
        value.size() > schema["maxItems"].get<std::size_t>()) {
      AddProblem(problems, ProblemKind::Arguments,
                 path + " allows at most " +
                     std::to_string(schema["maxItems"].get<std::size_t>()) +
                     " items");
    }
    if (schema.contains("items") && schema["items"].is_object()) {
      for (std::size_t i = 0; i < value.size(); ++i) {
        ValidateAgainstSchema(schema["items"], value[i],
                              path + "[" + std::to_string(i) + "]", problems,
                              depth + 1);
      }
    }
  }

  if (value.is_number()) {
    // nlohmann compares two integers exactly, so an integer bound above 2^53
    // is not rounded through double.
    if (schema.contains("minimum") && schema["minimum"].is_number() &&
        value < schema["minimum"]) {
      AddProblem(problems, ProblemKind::Arguments,
                 path + " must be >= " + schema["minimum"].dump());
    }
    if (schema.contains("maximum") && schema["maximum"].is_number() &&
        value > schema["maximum"]) {
      AddProblem(problems, ProblemKind::Arguments,
                 path + " must be <= " + schema["maximum"].dump());
    }
  }

  if (value.is_string()) {
    const std::string text = value.get<std::string>();
    const std::size_t length = detail::Utf8Length(text);
    if (schema.contains("minLength") &&
        schema["minLength"].is_number_unsigned() &&
        length < schema["minLength"].get<std::size_t>()) {
      AddProblem(problems, ProblemKind::Arguments,
                 path + " is shorter than " +
                     std::to_string(schema["minLength"].get<std::size_t>()) +
                     " characters");
    }
    if (schema.contains("maxLength") &&
        schema["maxLength"].is_number_unsigned() &&
        length > schema["maxLength"].get<std::size_t>()) {
      AddProblem(problems, ProblemKind::Arguments,
                 path + " is longer than " +
                     std::to_string(schema["maxLength"].get<std::size_t>()) +
                     " characters");
    }
  }
}

}  // namespace

std::string ValidationIssue::ToString() const {
  std::string where = "command call #" + std::to_string(index);
  if (!command_name.empty()) {
    where += " (" + command_name + ")";
  }
  return where + ": " + message;
}

bool ValidationReport::HasSchemaIssues() const {
  for (const auto& issue : issues) {
    if (issue.kind == ValidationIssueKind::Schema) {
      return true;
    }
  }
  return false;
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

JsonSchemaCommandValidator::JsonSchemaCommandValidator(ValidationLimits limits)
    : limits_(std::move(limits)) {}

ValidationReport JsonSchemaCommandValidator::ValidateCommandSchemas(
    const ModelRequest& request) const {
  ValidationReport report;
  if (!limits_.validate_schema) {
    report.ok = true;
    return report;
  }

  std::vector<std::string> seen_names;
  for (const auto& command : request.commands) {
    bool duplicate_name = false;
    for (const std::string& seen : seen_names) {
      if (!command.name.empty() && seen == command.name) {
        duplicate_name = true;
        break;
      }
    }
    if (!command.name.empty()) {
      seen_names.push_back(command.name);
    }
    if (duplicate_name) {
      ValidationIssue issue;
      issue.command_name = command.name;
      issue.kind = ValidationIssueKind::Schema;
      issue.message = "is offered more than once in this request";
      report.issues.push_back(std::move(issue));
      continue;
    }
    const Json schema =
        Json::parse(command.parameters_schema_json, nullptr, false);
    if (schema.is_discarded()) {
      ValidationIssue issue;
      issue.command_name = command.name;
      issue.kind = ValidationIssueKind::Schema;
      issue.message = "the command's parameters schema is not valid JSON";
      report.issues.push_back(std::move(issue));
      continue;
    }
    if (!schema.is_object()) {
      ValidationIssue issue;
      issue.command_name = command.name;
      issue.kind = ValidationIssueKind::Schema;
      issue.message = "the command's parameters schema must be a JSON object";
      report.issues.push_back(std::move(issue));
      continue;
    }

    Problems problems;
    CheckSchemaSupported(schema, "", problems, 0);
    for (auto& problem : problems) {
      ValidationIssue issue;
      issue.command_name = command.name;
      issue.kind = ValidationIssueKind::Schema;
      std::string message = std::move(problem.message);
      if (!message.empty() && message.front() == '.') {
        message.erase(message.begin());
      }
      issue.message = std::move(message);
      report.issues.push_back(std::move(issue));
    }
  }

  report.ok = report.issues.empty();
  return report;
}

ValidationReport JsonSchemaCommandValidator::Validate(
    const ModelRequest& request,
    const std::vector<CommandCall>& command_calls) const {
  ValidationReport report;

  const std::size_t effective_max = limits_.max_command_calls.has_value()
                                        ? limits_.max_command_calls.value()
                                        : request.max_command_calls.value_or(0);

  if (command_calls.size() < limits_.min_command_calls) {
    ValidationIssue issue;
    issue.message =
        "expected at least " + std::to_string(limits_.min_command_calls) +
        " command call(s), got " + std::to_string(command_calls.size());
    report.issues.push_back(std::move(issue));
  }
  if (effective_max > 0 && command_calls.size() > effective_max) {
    ValidationIssue issue;
    issue.message = "expected at most " + std::to_string(effective_max) +
                    " command call(s), got " +
                    std::to_string(command_calls.size());
    report.issues.push_back(std::move(issue));
  }

  std::vector<std::string> seen_ids;
  for (std::size_t i = 0; i < command_calls.size(); ++i) {
    const CommandCall& call = command_calls[i];
    const std::size_t index = i;
    const std::string name = call.name;

    auto add_issue = [&report, index, &name](ValidationIssueKind kind,
                                             std::string message) {
      ValidationIssue issue;
      issue.index = index;
      issue.command_name = name;
      issue.kind = kind;
      issue.message = std::move(message);
      report.issues.push_back(std::move(issue));
    };

    if (limits_.require_ids && call.id.empty()) {
      add_issue(ValidationIssueKind::Arguments,
                "has no id, so its result could not be sent back");
    }
    if (limits_.require_unique_ids && !call.id.empty()) {
      for (const auto& seen : seen_ids) {
        if (seen == call.id) {
          add_issue(ValidationIssueKind::Arguments,
                    "reuses command call id '" + call.id +
                        "', so the results could not be told apart");
          break;
        }
      }
      seen_ids.push_back(call.id);
    }
    if (call.name.empty()) {
      add_issue(ValidationIssueKind::Arguments, "has no command name");
      continue;
    }

    const CommandDefinition* command = limits_.require_known_commands
                                           ? request.FindCommand(call.name)
                                           : nullptr;
    if (limits_.require_known_commands && command == nullptr) {
      add_issue(ValidationIssueKind::Arguments,
                "is not one of the commands offered in this request");
      continue;
    }

    const Json arguments = Json::parse(call.arguments_json, nullptr, false);
    if (arguments.is_discarded()) {
      add_issue(ValidationIssueKind::Arguments, "arguments are not valid JSON");
      continue;
    }
    if (limits_.require_arguments_object && !arguments.is_object()) {
      add_issue(ValidationIssueKind::Arguments,
                "arguments are '" + DescribeType(arguments) +
                    "' but must be a JSON object");
      continue;
    }

    if (!limits_.validate_schema || command == nullptr) {
      continue;
    }
    const Json schema =
        Json::parse(command->parameters_schema_json, nullptr, false);
    if (schema.is_discarded()) {
      add_issue(ValidationIssueKind::Schema,
                "the command's parameters schema is not valid JSON");
      continue;
    }
    Problems problems;
    ValidateAgainstSchema(schema, arguments, "", problems, 0);
    for (auto& problem : problems) {
      std::string message = std::move(problem.message);
      if (!message.empty() && message.front() == '.') {
        message.erase(message.begin());
      }
      add_issue(problem.kind == ProblemKind::Schema
                    ? ValidationIssueKind::Schema
                    : ValidationIssueKind::Arguments,
                std::move(message));
    }
  }

  report.ok = report.issues.empty();
  return report;
}

std::shared_ptr<const CommandValidator> DefaultValidator() {
  static const auto validator =
      std::make_shared<const JsonSchemaCommandValidator>();
  return validator;
}

}  // namespace llm
