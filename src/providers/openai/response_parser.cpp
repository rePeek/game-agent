#include "response_parser.h"

#include <string>

#include <nlohmann/json.hpp>

namespace llm {
namespace openai {
namespace detail {
namespace {

std::string OptionalString(const nlohmann::json& value,
                           const char* key,
                           bool& present) {
  present = false;
  if (!value.contains(key)) {
    return {};
  }
  const auto& field = value[key];
  if (field.is_null()) {
    return {};
  }
  if (!field.is_string()) {
    return {};
  }
  present = true;
  return field.get<std::string>();
}

/// Reads an optional string field strictly: a present field of the wrong type
/// is a protocol error, not something to ignore.
bool ReadStringField(const nlohmann::json& object,
                     const char* key,
                     std::string& out,
                     std::string& error) {
  const auto it = object.find(key);
  if (it == object.end() || it->is_null()) {
    out.clear();
    return true;
  }
  if (!it->is_string()) {
    error = std::string("field '") + key + "' must be a string";
    return false;
  }
  out = it->get<std::string>();
  return true;
}

/// Reads an optional integer field strictly, rejecting wrong types and values
/// that do not fit in std::int64_t.
bool ReadIntField(const nlohmann::json& object,
                  const char* key,
                  std::int64_t& out,
                  std::string& error) {
  const auto it = object.find(key);
  if (it == object.end() || it->is_null()) {
    out = 0;
    return true;
  }
  if (!it->is_number_integer()) {
    error = std::string("field '") + key + "' must be an integer";
    return false;
  }
  try {
    out = it->get<std::int64_t>();
  } catch (const nlohmann::json::exception&) {
    error = std::string("field '") + key + "' is out of range";
    return false;
  }
  return true;
}

/// Arguments travel as text. A conforming server sends a JSON string; an object
/// (seen from some compatible servers) is re-serialized so it can still be
/// carried, and anything else — including a missing or null arguments object —
/// is rejected: the SDK never invents arguments the model did not produce.
bool ReadArguments(const nlohmann::json& function,
                   std::string& arguments,
                   std::string& error) {
  const auto it = function.find("arguments");
  if (it == function.end()) {
    error = "tool call has no arguments object";
    return false;
  }
  if (it->is_null()) {
    error = "tool call has null arguments";
    return false;
  }
  if (it->is_string()) {
    arguments = it->get<std::string>();
    return true;
  }
  if (it->is_object() || it->is_array()) {
    arguments = it->dump();
    return true;
  }
  error = "tool call arguments are neither a string nor a JSON value";
  return false;
}

}  // namespace

FinishReason MapFinishReason(const std::string& wire_value) {
  if (wire_value == "stop") {
    return FinishReason::Stop;
  }
  if (wire_value == "length") {
    return FinishReason::Length;
  }
  if (wire_value == "content_filter") {
    return FinishReason::ContentFilter;
  }
  if (wire_value == "tool_calls" || wire_value == "function_call") {
    return FinishReason::CommandCalls;
  }
  if (wire_value == "error") {
    return FinishReason::Error;
  }
  return FinishReason::Unknown;
}

std::string ExtractErrorMessage(const std::string& body) {
  const nlohmann::json parsed = nlohmann::json::parse(body, nullptr, false);
  if (!parsed.is_discarded() && parsed.is_object() &&
      parsed.contains("error")) {
    const auto& error = parsed["error"];
    if (error.is_object() && error.contains("message") &&
        error["message"].is_string()) {
      return error["message"].get<std::string>();
    }
    if (error.is_string()) {
      return error.get<std::string>();
    }
  }

  // Keep the payload readable but bounded: error bodies can be large HTML.
  constexpr std::size_t kMaxLength = 512;
  if (body.size() <= kMaxLength) {
    return body;
  }
  return body.substr(0, kMaxLength) + "...";
}

ParsedCompletion ParseChatCompletion(const std::string& body) {
  ParsedCompletion parsed;
  // Last-resort guard: a malformed response must become a Parse failure, never
  // an exception escaping the provider.
  try {
    const nlohmann::json response = nlohmann::json::parse(body, nullptr, false);
    if (response.is_discarded()) {
      parsed.error = "response body is not valid JSON";
      return parsed;
    }
    if (!response.is_object() || !response.contains("choices") ||
        !response["choices"].is_array() || response["choices"].empty()) {
      parsed.error = "response has no choices";
      return parsed;
    }

    if (!ReadStringField(response, "model", parsed.model, parsed.error)) {
      return parsed;
    }

    if (response.contains("usage") && response["usage"].is_object()) {
      const auto& usage = response["usage"];
      if (!ReadIntField(usage, "prompt_tokens", parsed.usage.prompt_tokens,
                        parsed.error) ||
          !ReadIntField(usage, "completion_tokens",
                        parsed.usage.completion_tokens, parsed.error) ||
          !ReadIntField(usage, "total_tokens", parsed.usage.total_tokens,
                        parsed.error)) {
        return parsed;
      }
    }

    const auto& choice = response["choices"][0];
    if (!choice.is_object()) {
      parsed.error = "first choice is not an object";
      return parsed;
    }
    if (!ReadStringField(choice, "finish_reason", parsed.finish_reason,
                         parsed.error)) {
      return parsed;
    }

    if (!choice.contains("message") || !choice["message"].is_object()) {
      parsed.error = "first choice has no message";
      return parsed;
    }
    const auto& message = choice["message"];

    bool has_text = false;
    parsed.assistant_text = OptionalString(message, "content", has_text);

    bool has_refusal = false;
    const std::string refusal_text =
        OptionalString(message, "refusal", has_refusal);
    if (has_refusal && !refusal_text.empty()) {
      parsed.refusal = refusal_text;
    }
    // Some servers report a refusal through finish_reason only.
    if (parsed.finish_reason == "refusal") {
      parsed.refusal =
          parsed.assistant_text.empty() ? "refused" : parsed.assistant_text;
    }

    if (message.contains("tool_calls") && !message["tool_calls"].is_null()) {
      if (!message["tool_calls"].is_array()) {
        parsed.error = "message.tool_calls is not an array";
        return parsed;
      }
      for (const auto& wire_call : message["tool_calls"]) {
        if (!wire_call.is_object() || !wire_call.contains("function") ||
            !wire_call["function"].is_object()) {
          parsed.error = "tool call is missing its function object";
          return parsed;
        }
        const auto& function = wire_call["function"];

        CommandCall call;
        if (!ReadStringField(wire_call, "id", call.id, parsed.error) ||
            !ReadStringField(function, "name", call.name, parsed.error)) {
          return parsed;
        }
        if (!ReadArguments(function, call.arguments_json, parsed.error)) {
          return parsed;
        }
        parsed.command_calls.push_back(std::move(call));
      }
    }

    parsed.ok = true;
    return parsed;
  } catch (const std::exception& error) {
    parsed.ok = false;
    parsed.error =
        std::string("could not parse the completion: ") + error.what();
  } catch (...) {
    parsed.ok = false;
    parsed.error = "could not parse the completion";
  }
  return parsed;
}

}  // namespace detail
}  // namespace openai
}  // namespace llm
