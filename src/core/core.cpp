#include "llm/core/logger.h"
#include "llm/core/message.h"
#include "llm/core/request.h"
#include "llm/core/result.h"
#include "llm/core/types.h"

#include <memory>
#include <string_view>
#include <utility>

namespace llm {

std::string ToolCall::ToString() const {
  return "ToolCall{id='" + id + "', name='" + name +
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

Message Message::AssistantToolCalls(std::vector<ToolCall> calls) {
  Message message;
  message.role = Role::Assistant;
  message.tool_calls = std::move(calls);
  return message;
}

Message Message::ToolResult(const llm::ToolResult& result) {
  Message message;
  message.role = Role::Tool;
  message.content = result.content_json;
  message.tool_call_id = result.tool_call_id;
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
    case FinishReason::ToolCalls:
      return "tool_calls";
    case FinishReason::Error:
      return "error";
    case FinishReason::Unknown:
      break;
  }
  return "unknown";
}

ToolChoice ToolChoice::Auto() {
  return ToolChoice{ToolChoiceMode::Auto, {}};
}

ToolChoice ToolChoice::Required() {
  return ToolChoice{ToolChoiceMode::Required, {}};
}

ToolChoice ToolChoice::None() {
  return ToolChoice{ToolChoiceMode::None, {}};
}

ToolChoice ToolChoice::Specific(std::string name) {
  return ToolChoice{ToolChoiceMode::Specific, std::move(name)};
}

std::string ToolChoice::ToString() const {
  switch (mode) {
    case ToolChoiceMode::Auto:
      return "auto";
    case ToolChoiceMode::Required:
      return "required";
    case ToolChoiceMode::None:
      return "none";
    case ToolChoiceMode::Specific:
      return tool_name.empty() ? "specific" : tool_name;
  }
  return "unknown";
}

const ToolDefinition* ModelRequest::FindTool(const std::string& name) const {
  for (const auto& tool : tools) {
    if (tool.name == name) {
      return &tool;
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
  message.tool_calls = tool_calls;
  return message;
}

std::string ToString(CompletionStatus status) {
  switch (status) {
    case CompletionStatus::Success:
      return "success";
    case CompletionStatus::NoToolCalls:
      return "no tool calls";
    case CompletionStatus::InvalidToolCalls:
      return "invalid tool calls";
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
