#include "llm/commands/registry.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace llm {
namespace {

bool ValidateParameter(const std::string& command_name,
                       const ParameterDescriptor& parameter,
                       std::string& error) {
  const auto fail = [&](std::string message) {
    error = "command '" + command_name + "', parameter '" + parameter.name +
            "': " + std::move(message);
    return false;
  };

  // An optional parameter must say what it means when the model leaves it
  // out; otherwise binding would silently value-initialize it.
  if (!parameter.required && !parameter.default_json.has_value()) {
    return fail(
        "an optional parameter needs a default value or must be "
        "required");
  }
  if (parameter.minimum.has_value() && parameter.maximum.has_value() &&
      *parameter.minimum > *parameter.maximum) {
    return fail("minimum is greater than maximum");
  }
  if (parameter.min_length.has_value() && parameter.max_length.has_value() &&
      *parameter.min_length > *parameter.max_length) {
    return fail("minLength is greater than maxLength");
  }
  if ((parameter.min_length.has_value() && *parameter.min_length < 0) ||
      (parameter.max_length.has_value() && *parameter.max_length < 0)) {
    return fail("string length bounds must not be negative");
  }
  if (parameter.min_items.has_value() && parameter.max_items.has_value() &&
      *parameter.min_items > *parameter.max_items) {
    return fail("minItems is greater than maxItems");
  }
  if ((parameter.min_items.has_value() && *parameter.min_items < 0) ||
      (parameter.max_items.has_value() && *parameter.max_items < 0)) {
    return fail("item count bounds must not be negative");
  }

  if (parameter.default_json.has_value()) {
    const std::string wrapped = "{\"" + EscapeJsonString(parameter.name) +
                                "\":" + *parameter.default_json + "}";
    const CommandArguments args = ParseCommandArguments(wrapped);
    std::string detail;
    if (!args.valid() || !CheckArgumentConstraints(args, parameter, detail)) {
      return fail("the default value is invalid: " + detail);
    }
  }
  return true;
}

}  // namespace

CommandRegistrationResult CommandRegistry::Add(RegisteredCommand command) {
  CommandRegistrationResult result;

  if (command.definition.name.empty()) {
    result.status = RegistrationStatus::EmptyName;
    result.error = "a command needs a non-empty name";
    return result;
  }
  for (const RegisteredCommand& existing : commands_) {
    if (existing.definition.name == command.definition.name) {
      result.status = RegistrationStatus::DuplicateName;
      result.error =
          "command '" + command.definition.name + "' is already registered";
      return result;
    }
  }
  for (std::size_t i = 0; i < command.parameters.size(); ++i) {
    const ParameterDescriptor& parameter = command.parameters[i];
    if (parameter.name.empty()) {
      result.status = RegistrationStatus::InvalidParameter;
      result.error = "command '" + command.definition.name +
                     "' has a parameter without a name";
      return result;
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (command.parameters[j].name == parameter.name) {
        result.status = RegistrationStatus::InvalidParameter;
        result.error = "command '" + command.definition.name +
                       "' declares parameter '" + parameter.name +
                       "' more than once";
        return result;
      }
    }
  }
  for (const ParameterDescriptor& parameter : command.parameters) {
    if (!ValidateParameter(command.definition.name, parameter, result.error)) {
      result.status = RegistrationStatus::InvalidParameter;
      return result;
    }
  }
  if (!command.bind) {
    result.status = RegistrationStatus::InvalidParameter;
    result.error =
        "command '" + command.definition.name + "' has no argument decoder";
    return result;
  }

  commands_.push_back(std::move(command));
  ++version_;
  result.status = RegistrationStatus::Registered;
  return result;
}

bool CommandRegistry::Remove(std::string_view name) {
  for (auto it = commands_.begin(); it != commands_.end(); ++it) {
    if (std::string_view(it->definition.name) == name) {
      commands_.erase(it);
      ++version_;
      return true;
    }
  }
  return false;
}

void CommandRegistry::Clear() {
  if (commands_.empty()) {
    return;
  }
  commands_.clear();
  ++version_;
}

bool CommandRegistry::Contains(std::string_view name) const {
  return Find(name) != nullptr;
}

std::vector<CommandDefinition> CommandRegistry::Definitions() const {
  std::vector<CommandDefinition> definitions;
  definitions.reserve(commands_.size());
  for (const RegisteredCommand& command : commands_) {
    definitions.push_back(command.definition);
  }
  return definitions;
}

const RegisteredCommand* CommandRegistry::Find(std::string_view name) const {
  for (const RegisteredCommand& command : commands_) {
    if (std::string_view(command.definition.name) == name) {
      return &command;
    }
  }
  return nullptr;
}

}  // namespace llm
