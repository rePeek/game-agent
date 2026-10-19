#include "http_transport.h"
#include "llm/providers/openai/openai.h"
#include "request_builder.h"
#include "response_parser.h"

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

  // Events are emitted with the request's diagnostics context, so a shared
  // provider can be attributed to the session that caused the call.
  const std::shared_ptr<ILogger> logger = options_.logger;
  const LogContext log_context = request.log;
  const auto log = [&logger, &log_context](LogLevel level,
                                           std::string_view event,
                                           std::string message) {
    EmitLog(logger, level, event, std::move(message), log_context);
  };

  if (options_.api_key.empty()) {
    log(LogLevel::Error, log_event::kRequestFailed,
        "kind=auth, attempts=0, message=no API key configured");
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
      log(LogLevel::Error, log_event::kRequestFailed,
          "kind=unsupported, attempts=0, message=" + unenforceable);
      result.status = CompletionStatus::Unsupported;
      result.error = MakeError(ErrorKind::Unsupported, unenforceable);
      return result;
    }
    result.warnings.push_back(
        "provider-side guarantee downgraded to SDK-side "
        "validation: " +
        unenforceable);
    log(LogLevel::Warning, log_event::kRequirementDowngraded, unenforceable);
  }

  // 2. Build the request. A malformed request is the caller's error and is
  //    reported as such before anything else is checked.
  detail::BuildOutcome built =
      detail::BuildChatCompletionRequest(request, options_.parameter_dialect);
  if (!built.ok) {
    log(LogLevel::Error, log_event::kRequestFailed,
        "kind=invalid request, attempts=0, message=" + built.error);
    return ProviderErrorResult(
        request, MakeError(ErrorKind::InvalidRequest, built.error));
  }

  // 3. A schema this validator cannot enforce must not be sent either: Success
  //    would otherwise claim a check that never happened. There is no
  //    downgrade here — the SDK simply cannot verify those constraints.
  if (options_.validator) {
    const ValidationReport schemas =
        options_.validator->ValidateToolSchemas(request);
    if (!schemas.ok) {
      log(LogLevel::Warning, log_event::kValidationFailed,
          "the request uses a tool schema that cannot be enforced: " +
              schemas.Summary());
      result.status = CompletionStatus::Unsupported;
      result.error =
          MakeError(ErrorKind::Unsupported,
                    "the request uses a tool schema that cannot be enforced: " +
                        schemas.Summary());
      return result;
    }
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

  log(LogLevel::Debug, log_event::kRequestStarted,
      "model=" + request.model +
          ", tools=" + std::to_string(request.tools.size()) +
          ", tool_choice=" + request.tool_choice.ToString() +
          ", messages=" + std::to_string(request.messages.size()));

  // 4. One completion, one (possibly retried) HTTP request.
  const int max_attempts = std::max(1, options_.retry.max_attempts);
  int delay_ms = std::max(0, options_.retry.initial_delay_ms);
  int attempts_made = 0;
  HttpResponse response;
  for (int attempt = 1;; ++attempt) {
    ++attempts_made;
    response = options_.transport->Send(http);

    const bool succeeded = response.IsSuccess();
    const bool retryable =
        response.transport_ok ? IsRetryableStatus(response.status) : true;
    if (succeeded || !retryable || attempt >= max_attempts) {
      break;
    }
    log(LogLevel::Warning, log_event::kRequestRetry,
        "attempt=" + std::to_string(attempt) +
            ", http=" + std::to_string(response.status) +
            ", kind=" + ToString(ClassifyStatus(response.status)) +
            ", next_delay_ms=" + std::to_string(delay_ms));
    if (delay_ms > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
      delay_ms = static_cast<int>(std::min(
          static_cast<double>(options_.retry.max_delay_ms),
          static_cast<double>(delay_ms) * options_.retry.backoff_factor));
    }
  }

  // Retries belong to this one logical completion; they are never extra
  // reasoning rounds.
  result.http_attempts = attempts_made;
  result.provider_metadata = response.body;

  // The error paths below keep what was already recorded on `result`
  // (attempt count, raw body), so they must not build a fresh result.
  if (!response.transport_ok) {
    result.status = CompletionStatus::ProviderError;
    result.error =
        MakeError(ClassifyTransportError(response.error_kind),
                  "transport failure: " + (response.error.empty()
                                               ? std::string("unknown error")
                                               : response.error),
                  0, true, response.body);
    // Classification only: the provider's or the network's own text stays in
    // CompletionResult, not in the log.
    log(LogLevel::Error, log_event::kRequestFailed,
        "kind=" + ToString(result.error->kind) +
            ", http=0, attempts=" + std::to_string(attempts_made));
    return result;
  }
  if (!response.IsSuccess()) {
    result.status = CompletionStatus::ProviderError;
    result.error =
        MakeError(ClassifyStatus(response.status),
                  detail::ExtractErrorMessage(response.body), response.status,
                  IsRetryableStatus(response.status), response.body);
    log(LogLevel::Error, log_event::kRequestFailed,
        "kind=" + ToString(result.error->kind) +
            ", http=" + std::to_string(response.status) +
            ", attempts=" + std::to_string(attempts_made));
    return result;
  }

  // 5. Parse the completion.
  detail::ParsedCompletion parsed = detail::ParseChatCompletion(response.body);
  if (!parsed.ok) {
    result.status = CompletionStatus::ProviderError;
    result.error = MakeError(ErrorKind::Parse, parsed.error, response.status,
                             false, response.body);
    // parsed.error is the parser's own description (never model output).
    log(LogLevel::Error, log_event::kRequestFailed,
        "kind=parse, http=" + std::to_string(response.status) + ", attempts=" +
            std::to_string(attempts_made) + ", message=" + parsed.error);
    return result;
  }

  if (!parsed.model.empty()) {
    result.model = parsed.model;
  }
  result.usage = parsed.usage;
  result.finish_reason = detail::MapFinishReason(parsed.finish_reason);
  result.assistant_text = parsed.assistant_text;
  result.tool_calls = parsed.tool_calls;

  // 6. Decide the status the caller branches on. One exit, so the completion
  //    is logged once with the status the caller actually receives.
  if (!parsed.refusal.empty()) {
    result.status = CompletionStatus::Refused;
    result.error =
        MakeError(ErrorKind::InvalidRequest, parsed.refusal, response.status);
  } else if (result.finish_reason == FinishReason::Length) {
    result.status = CompletionStatus::Truncated;
    result.error = MakeError(
        ErrorKind::InvalidRequest,
        "completion was cut off by the token limit before the tool calls were "
        "complete",
        response.status);
  } else if (result.finish_reason == FinishReason::ContentFilter) {
    result.status = CompletionStatus::Refused;
    result.error = MakeError(ErrorKind::InvalidRequest,
                             "completion was blocked by the provider's content "
                             "filter",
                             response.status);
  } else if (result.tool_calls.empty()) {
    result.status = CompletionStatus::NoToolCalls;
    result.error = MakeError(
        ErrorKind::InvalidRequest,
        parsed.assistant_text.empty()
            ? "completion contained no tool calls"
            : "completion contained no tool calls: " + parsed.assistant_text,
        response.status);
  } else {
    // The batch must pass every check the SDK promises: known tools, ids,
    // parseable arguments and a matching parameter schema.
    const ValidationReport report =
        options_.validator->Validate(request, result.tool_calls);
    if (!report.ok) {
      log(LogLevel::Warning, log_event::kValidationFailed, report.Summary());
      result.status = report.HasSchemaIssues()
                          ? CompletionStatus::Unsupported
                          : CompletionStatus::InvalidToolCalls;
      result.error = MakeError(ErrorKind::InvalidRequest, report.Summary(),
                               response.status);
    } else {
      result.status = CompletionStatus::Success;
    }
  }

  log(result.IsSuccess() ? LogLevel::Info : LogLevel::Warning,
      log_event::kRequestCompleted,
      "status=" + ToString(result.status) +
          ", http=" + std::to_string(response.status) +
          ", attempts=" + std::to_string(attempts_made) +
          ", tokens=" + std::to_string(result.usage.total_tokens) +
          ", tools=" + std::to_string(result.tool_calls.size()) +
          ", finish_reason=" + ToString(result.finish_reason));
  return result;
}

std::shared_ptr<IModelProvider> CreateProvider(ProviderOptions options) {
  return std::make_shared<OpenAIProvider>(std::move(options));
}

std::shared_ptr<IModelProvider> CreateProviderFromEnvironment(
    std::shared_ptr<ILogger> logger) {
  ProviderOptions options;
  options.logger = std::move(logger);
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
