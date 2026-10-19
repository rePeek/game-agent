#pragma once

#include "llm/core/logger.h"
#include "llm/core/message.h"
#include "llm/core/types.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace llm {

/// How the provider should pick commands for a request.
enum class CommandChoiceMode {
  Auto,      // the model decides whether to call a command
  Required,  // the model must call at least one command
  None,      // commands are not offered for this request
  Specific,  // the model must call `command_name`
};

struct CommandChoice {
  CommandChoiceMode mode = CommandChoiceMode::Required;
  std::string command_name;  // only meaningful for CommandChoiceMode::Specific

  static CommandChoice Auto();
  static CommandChoice Required();
  static CommandChoice None();
  static CommandChoice Specific(std::string command_name);

  /// Provider-independent label: "auto", "required", "none" or the name.
  std::string ToString() const;
};

/// A command the model may call.
///
/// `parameters_schema_json` is a JSON Schema object serialized as text, so the
/// core does not need a JSON library to carry it. `strict` asks the provider to
/// enforce that schema itself; providers that cannot must report it through
/// ProviderCapabilities instead of pretending the schema is enforced.
struct CommandDefinition {
  std::string name;
  std::string description;
  std::string parameters_schema_json;
  bool strict = false;

  bool IsValid() const { return !name.empty(); }
};

/// One completion request: the caller's context plus the commands it may use.
struct ModelRequest {
  std::string model;
  std::vector<Message> messages;
  std::vector<CommandDefinition> commands;

  /// Forced command calling is the SDK's contract, so `required` is the
  /// default; callers that also accept plain answers set CommandChoice::Auto().
  CommandChoice command_choice = CommandChoice::Required();

  std::optional<std::int32_t> max_tokens;
  std::optional<double> temperature;
  std::optional<double> top_p;
  std::optional<std::int64_t> seed;

  /// Upper bound on the number of command calls a single completion may return.
  /// Enforced by the SDK validator, not by the provider.
  std::optional<std::size_t> max_command_calls;

  /// Diagnostics only: lets a provider attribute its log events to the agent
  /// and decision that caused them. Request builders ignore it.
  LogContext log;

  bool HasCommands() const { return !commands.empty(); }
  const CommandDefinition* FindCommand(const std::string& name) const;
};

}  // namespace llm
