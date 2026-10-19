#include "llm/http.h"
#include "llm/openai.h"
#include "openai_request_builder.h"
#include "openai_response_parser.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>
#include <utility>

#include <nlohmann/json.hpp>

namespace llm {
namespace openai {
namespace {

constexpr char kCompletionsPath[] = "/v1/chat/completions";

ModelError MakeError(ErrorKind kind,
                     std::string message,
                     int http_status = 0,
                     bool retryable = false,
                     std::string raw_body = {}) {
  ModelError error;
  error.kind = kind;
  error.message = std::move(message);
  error.http_status = http_status;
  error.retryable = retryable;
  error.raw_body = std::move(raw_body);
  return error;
}

bool IsRetryableStatus(int status) {
  switch (status) {
    case 408:  // request timeout
    case 409:  // conflict
    case 429:  // rate limited
    case 500:
    case 502:
    case 503:
    case 504:
      return true;
    default:
      return false;
  }
}

ErrorKind ClassifyStatus(int status) {
  if (status == 401 || status == 403) {
    return ErrorKind::Auth;
  }
  if (status == 429) {
    return ErrorKind::RateLimit;
  }
  if (status >= 500) {
    return ErrorKind::Server;
  }
  if (status >= 400) {
    return ErrorKind::InvalidRequest;
  }
  return ErrorKind::Unknown;
}

ErrorKind ClassifyTransportError(TransportError error) {
  switch (error) {
    case TransportError::Timeout:
      return ErrorKind::Timeout;
    case TransportError::Connection:
    case TransportError::Tls:
    case TransportError::Io:
      return ErrorKind::Network;
    case TransportError::None:
    case TransportError::Other:
      return ErrorKind::Unknown;
  }
  return ErrorKind::Unknown;
}

/// Requirements of this request that the provider cannot guarantee. A provider
/// that cannot force tool calling or enforce a schema must say so instead of
/// being assumed to have done it.
std::string CheckCapabilities(const ModelRequest& request,
                              const ProviderCapabilities& capabilities) {
  std::string problems;

  const bool wants_forced_choice =
      request.tool_choice.mode == ToolChoiceMode::Required ||
      request.tool_choice.mode == ToolChoiceMode::Specific;
  if (wants_forced_choice && !capabilities.forced_tool_choice) {
    problems += "forced tool_choice (" + request.tool_choice.ToString() +
                ") is not supported by this provider";
  }

  for (const auto& tool : request.tools) {
    if (tool.strict && !capabilities.strict_tool_schema) {
      if (!problems.empty()) {
        problems += "; ";
      }
      problems += "strict schema for tool '" + tool.name +
                  "' cannot be enforced by this provider";
      break;
    }
  }

  return problems;
}

std::string BuildUrl(const std::string& base_url) {
  std::string base = base_url;
  while (!base.empty() && base.back() == '/') {
    base.pop_back();
  }
  return base + kCompletionsPath;
}

CompletionResult ProviderErrorResult(const ModelRequest& request,
                                     ModelError error) {
  CompletionResult result;
  result.status = CompletionStatus::ProviderError;
  result.model = request.model;
  result.error = std::move(error);
  return result;
}

}  // namespace

OpenAIProvider::OpenAIProvider(ProviderOptions options)
    : options_(std::move(options)) {
  if (!options_.transport) {
    options_.transport = CreateHttpsTransport();
  }
  if (!options_.validator) {
    options_.validator = DefaultValidator();
  }
}

CompletionResult OpenAIProvider::Complete(const ModelRequest& request) {
  CompletionResult result;
  result.model = request.model;

  if (options_.api_key.empty()) {
    return ProviderErrorResult(
        request, MakeError(ErrorKind::Auth,
                           "no API key configured for the OpenAI provider"));
  }

  // 1. Refuse what this provider cannot guarantee, unless the caller explicitly
  //    accepts SDK-side validation instead.
  const std::string unenforceable =
      CheckCapabilities(request, options_.capabilities);
  if (!unenforceable.empty()) {
    if (!options_.allow_validation_downgrade) {
      result.status = CompletionStatus::Unsupported;
      result.error = MakeError(ErrorKind::Unsupported, unenforceable);
      return result;
    }
    result.warnings.push_back(
        "provider-side guarantee downgraded to SDK-side "
        "validation: " +
        unenforceable);
  }

  // 2. Build the request.
  detail::BuildOutcome built = detail::BuildChatCompletionRequest(request);
  if (!built.ok) {
    return ProviderErrorResult(
        request, MakeError(ErrorKind::InvalidRequest, built.error));
  }

  HttpRequest http;
  http.url = BuildUrl(options_.base_url);
  http.body = built.body.dump();
  http.timeout_seconds = options_.timeout_seconds;
  http.headers.push_back({"Authorization", "Bearer " + options_.api_key});
  http.headers.push_back({"Content-Type", "application/json"});
  http.headers.push_back({"Accept", "application/json"});
  if (!options_.organization.empty()) {
    http.headers.push_back({"OpenAI-Organization", options_.organization});
  }
  for (const auto& header : options_.extra_headers) {
    http.headers.push_back(header);
  }

  // 3. One completion, one (possibly retried) HTTP request.
  const int max_attempts = std::max(1, options_.retry.max_attempts);
  int delay_ms = std::max(0, options_.retry.initial_delay_ms);
  HttpResponse response;
  for (int attempt = 1;; ++attempt) {
    response = options_.transport->Send(http);

    const bool succeeded = response.IsSuccess();
    const bool retryable =
        response.transport_ok ? IsRetryableStatus(response.status) : true;
    if (succeeded || !retryable || attempt >= max_attempts) {
      break;
    }
    if (delay_ms > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
      delay_ms = static_cast<int>(std::min(
          static_cast<double>(options_.retry.max_delay_ms),
          static_cast<double>(delay_ms) * options_.retry.backoff_factor));
    }
  }

  result.provider_metadata = response.body;

  if (!response.transport_ok) {
    return ProviderErrorResult(
        request,
        MakeError(ClassifyTransportError(response.error_kind),
                  "transport failure: " + (response.error.empty()
                                               ? std::string("unknown error")
                                               : response.error),
                  0, true, response.body));
  }
  if (!response.IsSuccess()) {
    return ProviderErrorResult(
        request,
        MakeError(ClassifyStatus(response.status),
                  detail::ExtractErrorMessage(response.body), response.status,
                  IsRetryableStatus(response.status), response.body));
  }

  // 4. Parse the completion.
  detail::ParsedCompletion parsed = detail::ParseChatCompletion(response.body);
  if (!parsed.ok) {
    return ProviderErrorResult(
        request, MakeError(ErrorKind::Parse, parsed.error, response.status,
                           false, response.body));
  }

  if (!parsed.model.empty()) {
    result.model = parsed.model;
  }
  result.usage = parsed.usage;
  result.finish_reason = detail::MapFinishReason(parsed.finish_reason);
  result.assistant_text = parsed.assistant_text;
  result.tool_calls = parsed.tool_calls;

  // 5. Decide the status the caller branches on.
  if (!parsed.refusal.empty()) {
    result.status = CompletionStatus::Refused;
    result.error =
        MakeError(ErrorKind::InvalidRequest, parsed.refusal, response.status);
    return result;
  }
  if (result.finish_reason == FinishReason::Length) {
    result.status = CompletionStatus::Truncated;
    result.error = MakeError(
        ErrorKind::InvalidRequest,
        "completion was cut off by the token limit before the tool calls were "
        "complete",
        response.status);
    return result;
  }
  if (result.finish_reason == FinishReason::ContentFilter) {
    result.status = CompletionStatus::Refused;
    result.error = MakeError(ErrorKind::InvalidRequest,
                             "completion was blocked by the provider's content "
                             "filter",
                             response.status);
    return result;
  }
  if (result.tool_calls.empty()) {
    result.status = CompletionStatus::NoToolCalls;
    result.error = MakeError(
        ErrorKind::InvalidRequest,
        parsed.assistant_text.empty()
            ? "completion contained no tool calls"
            : "completion contained no tool calls: " + parsed.assistant_text,
        response.status);
    return result;
  }

  // The batch must pass every check the SDK promises: known tools, ids,
  // parseable arguments and a matching parameter schema.
  const ValidationReport report =
      options_.validator->Validate(request, result.tool_calls);
  if (!report.ok) {
    result.status = CompletionStatus::InvalidToolCalls;
    result.error =
        MakeError(ErrorKind::InvalidRequest, report.Summary(), response.status);
    return result;
  }

  result.status = CompletionStatus::Success;
  return result;
}

std::shared_ptr<IModelProvider> CreateProvider(ProviderOptions options) {
  return std::make_shared<OpenAIProvider>(std::move(options));
}

std::shared_ptr<IModelProvider> CreateProviderFromEnvironment() {
  ProviderOptions options;
  if (const char* api_key = std::getenv("OPENAI_API_KEY"); api_key != nullptr) {
    options.api_key = api_key;
  }
  if (const char* base_url = std::getenv("OPENAI_BASE_URL");
      base_url != nullptr && *base_url != '\0') {
    options.base_url = base_url;
  }
  return CreateProvider(std::move(options));
}

}  // namespace openai
}  // namespace llm
