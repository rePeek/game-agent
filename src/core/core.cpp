#include "llm/core/logger.h"
#include "llm/core/message.h"
#include "llm/core/request.h"
#include "llm/core/result.h"
#include "llm/core/types.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace llm {
namespace {

/// Escapes a string for embedding in JSON text (RFC 8259). The core has no
/// JSON library, so the one place it must emit text does the escaping itself.
std::string EscapeJson(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 2);
  for (const char raw : value) {
    const unsigned char byte = static_cast<unsigned char>(raw);
    switch (byte) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (byte < 0x20) {
          static const char kHex[] = "0123456789abcdef";
          out += "\\u00";
          out += kHex[(byte >> 4) & 0x0f];
          out += kHex[byte & 0x0f];
        } else {
          out += static_cast<char>(byte);
        }
        break;
    }
  }
  return out;
}

/// The one model-visible rendering of a command result: success and failure
/// share a shape, so the flag is actually transmitted instead of being implied
/// by the payload.
std::string FormatCommandResult(const CommandOutcome& outcome) {
  std::string out;
  if (outcome.success) {
    out = "{\"ok\":true";
    if (!outcome.content_json.empty()) {
      out += ",\"result\":";
      out += outcome.content_json;
    }
    out += "}";
    return out;
  }
  out = "{\"ok\":false";
  if (!outcome.error_message.empty()) {
    out += ",\"error\":\"";
    out += EscapeJson(outcome.error_message);
    out += "\"";
  }
  if (!outcome.content_json.empty()) {
    out += ",\"detail\":";
    out += outcome.content_json;
  }
  out += "}";
  return out;
}

}  // namespace


std::string CommandCall::ToString() const {
  return "CommandCall{id='" + id + "', name='" + name +
         "', arguments=" + arguments_json + "}";
}

Message Message::System(std::string message_content) {
  Message message;
  message.role = Role::System;
  message.content = std::move(message_content);
  return message;
}

Message Message::User(std::string message_content) {
  Message message;
  message.role = Role::User;
  message.content = std::move(message_content);
  return message;
}

Message Message::Assistant(std::string message_content) {
  Message message;
  message.role = Role::Assistant;
  message.content = std::move(message_content);
  return message;
}

Message Message::AssistantCommandCalls(std::vector<CommandCall> calls) {
  Message message;
  message.role = Role::Assistant;
  message.command_calls = std::move(calls);
  return message;
}

Message Message::CommandResultMessage(const llm::CommandResult& result) {
  Message message;
  message.role = Role::Tool;
  message.content = FormatCommandResult(result.outcome);
  message.command_call_id = result.command_call_id;
  message.name = result.name;
  return message;
}

std::string ToString(Role role) {
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
  return "unknown";
}

std::string ToString(FinishReason reason) {
  switch (reason) {
    case FinishReason::Stop:
      return "stop";
    case FinishReason::Length:
      return "length";
    case FinishReason::ContentFilter:
      return "content_filter";
    case FinishReason::CommandCalls:
      return "tool_calls";
    case FinishReason::Error:
      return "error";
    case FinishReason::Unknown:
      break;
  }
  return "unknown";
}

CommandChoice CommandChoice::Auto() {
  return CommandChoice{CommandChoiceMode::Auto, {}};
}

CommandChoice CommandChoice::Required() {
  return CommandChoice{CommandChoiceMode::Required, {}};
}

CommandChoice CommandChoice::None() {
  return CommandChoice{CommandChoiceMode::None, {}};
}

CommandChoice CommandChoice::Specific(std::string name) {
  return CommandChoice{CommandChoiceMode::Specific, std::move(name)};
}

std::string CommandChoice::ToString() const {
  switch (mode) {
    case CommandChoiceMode::Auto:
      return "auto";
    case CommandChoiceMode::Required:
      return "required";
    case CommandChoiceMode::None:
      return "none";
    case CommandChoiceMode::Specific:
      return command_name.empty() ? "specific" : command_name;
  }
  return "unknown";
}

const CommandDefinition* ModelRequest::FindCommand(const std::string& name) const {
  for (const auto& command : commands) {
    if (command.name == name) {
      return &command;
    }
  }
  return nullptr;
}

std::string CompletionResult::ErrorMessage() const {
  if (error.has_value() && !error->message.empty()) {
    return error->message;
  }
  return ToString(status);
}

Message CompletionResult::ToAssistantMessage() const {
  Message message;
  message.role = Role::Assistant;
  message.content = assistant_text;
  message.command_calls = command_calls;
  return message;
}

std::string ToString(CompletionStatus status) {
  switch (status) {
    case CompletionStatus::Success:
      return "success";
    case CompletionStatus::NoCommandCalls:
      return "no command calls";
    case CompletionStatus::InvalidCommandCalls:
      return "invalid command calls";
    case CompletionStatus::Truncated:
      return "truncated";
    case CompletionStatus::Refused:
      return "refused";
    case CompletionStatus::Unsupported:
      return "unsupported";
    case CompletionStatus::ProviderError:
      return "provider error";
  }
  return "unknown";
}

std::string ToString(ErrorKind kind) {
  switch (kind) {
    case ErrorKind::Network:
      return "network";
    case ErrorKind::Timeout:
      return "timeout";
    case ErrorKind::Auth:
      return "auth";
    case ErrorKind::RateLimit:
      return "rate limit";
    case ErrorKind::InvalidRequest:
      return "invalid request";
    case ErrorKind::Server:
      return "server";
    case ErrorKind::Parse:
      return "parse";
    case ErrorKind::Unsupported:
      return "unsupported";
    case ErrorKind::Unknown:
      break;
  }
  return "unknown";
}
std::size_t EstimateTokens(std::string_view text) {
  if (text.empty()) {
    return 0;
  }
  // Four characters per token is the usual rule of thumb, plus one so even a
  // tiny non-empty message is not counted as free.
  return text.size() / 4 + 1;
}


void EmitLog(const std::shared_ptr<ILogger>& logger,
             LogLevel level,
             std::string_view event,
             std::string message,
             const LogContext& context) {
  if (!logger) {
    return;  // no logger configured: the SDK stays silent
  }

  LogRecord record;
  record.level = level;
  record.event = std::string(event);
  record.message = std::move(message);
  record.session_id = context.session_id;
  record.request_id = context.request_id;

  try {
    logger->Log(record);
  } catch (...) {
    // A logger must never be able to fail a decision. Swallow and carry on.
  }
}

}  // namespace llm
