#include "request_builder.h"

#include <string>

#include "llm/commands/parameter.hpp"

namespace llm {
namespace openai {
namespace detail {
namespace {

bool ParseSchema(const std::string& text,
                 const std::string& command_name,
                 nlohmann::json& schema,
                 std::string& error) {
  if (text.empty()) {
    schema = nlohmann::json::object();
    return true;
  }
  schema = nlohmann::json::parse(text, nullptr, false);
  if (schema.is_discarded()) {
    error = "command '" + command_name +
            "' has a parameters schema that is not valid JSON";
    return false;
  }
  if (!schema.is_object()) {
    error = "command '" + command_name +
            "' parameters schema must be a JSON object";
    return false;
  }
  return true;
}

bool AppendMessage(const Message& message,
                   nlohmann::json& out,
                   std::string& error) {
  nlohmann::json wire{{"role", WireRole(message.role)}};

  switch (message.role) {
    case Role::System:
    case Role::User:
      wire["content"] = message.content;
      if (!message.name.empty()) {
        wire["name"] = message.name;
      }
      break;

    case Role::Assistant:
      if (message.command_calls.empty()) {
        wire["content"] = message.content;
      } else {
        // A tool-only assistant turn must send null content, not "".
        wire["content"] = message.content.empty()
                              ? nlohmann::json(nullptr)
                              : nlohmann::json(message.content);
        nlohmann::json tool_calls = nlohmann::json::array();
        for (const auto& call : message.command_calls) {
          if (call.name.empty()) {
            error = "assistant message has a tool call without a name";
            return false;
          }
          tool_calls.push_back(
              {{"id", call.id},
               {"type", "function"},
               {"function",
                {{"name", call.name}, {"arguments", call.arguments_json}}}});
        }
        wire["tool_calls"] = std::move(tool_calls);
      }
      if (!message.name.empty()) {
        wire["name"] = message.name;
      }
      break;

    case Role::Tool:
      if (message.command_call_id.empty()) {
        error = "tool message is missing tool_call_id";
        return false;
      }
      // OpenAI's tool message carries the id and the payload as text; `name`
      // is not part of it.
      wire["tool_call_id"] = message.command_call_id;
      wire["content"] = message.content;
      break;
  }

  out.push_back(std::move(wire));
  return true;
}

nlohmann::json BuildCommandChoice(const CommandChoice& choice,
                                  std::string& error) {
  switch (choice.mode) {
    case CommandChoiceMode::Auto:
      return "auto";
    case CommandChoiceMode::Required:
      return "required";
    case CommandChoiceMode::None:
      return "none";
    case CommandChoiceMode::Specific:
      if (choice.command_name.empty()) {
        error = "command choice is Specific but names no command";
        return nullptr;
      }
      return nlohmann::json{{"type", "function"},
                            {"function", {{"name", choice.command_name}}}};
  }
  error = "unknown tool_choice";
  return nullptr;
}

ParameterDialect ResolveDialect(const std::string& model,
                                ParameterDialect requested) {
  if (requested != ParameterDialect::Auto) {
    return requested;
  }
  return DialectForModel(model);
}

}  // namespace

const char* WireRole(Role role) {
  switch (role) {
    case Role::System:
      return "system";
    case Role::User:
      return "user";
    case Role::Assistant:
      return "assistant";
    case Role::Tool:
      return "tool";
  }
  return "user";
}

BuildOutcome BuildChatCompletionRequest(const ModelRequest& request,
                                        ParameterDialect dialect,
                                        bool enforce_strict) {
  BuildOutcome outcome;

  if (request.model.empty()) {
    outcome.error = "request.model must not be empty";
    return outcome;
  }
  if (request.messages.empty()) {
    outcome.error = "request.messages must not be empty";
    return outcome;
  }

  nlohmann::json body;
  body["model"] = request.model;
  body["messages"] = nlohmann::json::array();
  for (const auto& message : request.messages) {
    if (!AppendMessage(message, body["messages"], outcome.error)) {
      return outcome;
    }
  }

  if (!request.commands.empty()) {
    nlohmann::json tools = nlohmann::json::array();
    for (const auto& command : request.commands) {
      if (command.name.empty()) {
        outcome.error = "every tool definition needs a name";
        return outcome;
      }
      const bool strict = command.strict && enforce_strict;
      const std::string schema_text =
          strict ? ToStrictSchema(command.parameters_schema_json)
                 : command.parameters_schema_json;
      nlohmann::json parameters;
      if (!ParseSchema(schema_text, command.name, parameters, outcome.error)) {
        return outcome;
      }
      nlohmann::json function{{"name", command.name},
                              {"parameters", parameters}};
      if (!command.description.empty()) {
        function["description"] = command.description;
      }
      if (strict) {
        function["strict"] = true;
      }
      tools.push_back(
          {{"type", "function"}, {"function", std::move(function)}});
    }
    body["tools"] = std::move(tools);
  }

  std::string choice_error;
  nlohmann::json tool_choice =
      BuildCommandChoice(request.command_choice, choice_error);
  if (!choice_error.empty()) {
    outcome.error = choice_error;
    return outcome;
  }
  body["tool_choice"] = std::move(tool_choice);

  const ParameterDialect effective = ResolveDialect(request.model, dialect);
  if (request.max_tokens.has_value()) {
    if (effective == ParameterDialect::ReasoningTokens) {
      body["max_completion_tokens"] = request.max_tokens.value();
    } else {
      body["max_tokens"] = request.max_tokens.value();
    }
  }
  if (effective == ParameterDialect::LegacyTokens) {
    if (request.seed.has_value()) {
      body["seed"] = request.seed.value();
    }
    if (request.temperature.has_value()) {
      body["temperature"] = request.temperature.value();
    }
    if (request.top_p.has_value()) {
      body["top_p"] = request.top_p.value();
    }
  }

  outcome.ok = true;
  outcome.body = std::move(body);
  return outcome;
}

}  // namespace detail

ParameterDialect DialectForModel(const std::string& model) {
  // Reasoning-first families: on Chat Completions they use
  // max_completion_tokens and reject sampling controls.
  static const char* const kReasoningPrefixes[] = {
      "gpt-5", "o1", "o3", "o4", "o5",
  };
  for (const char* prefix : kReasoningPrefixes) {
    if (model.starts_with(prefix)) {
      return ParameterDialect::ReasoningTokens;
    }
  }
  return ParameterDialect::LegacyTokens;
}

}  // namespace openai
}  // namespace llm
