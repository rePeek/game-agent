#include "llm/commands/registry.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace llm {

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
      result.error = "command '" + command.definition.name +
                     "' is already registered";
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
  if (!command.bind) {
    result.status = RegistrationStatus::InvalidParameter;
    result.error = "command '" + command.definition.name +
                   "' has no argument decoder";
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
