#pragma once

#include "llm/core/message.h"
#include "llm/core/types.h"

#include <optional>
#include <string>
#include <vector>

namespace llm {

/// Outcome of a single Complete() call.
///
/// `Success` is reserved for a completion whose command calls all passed the
/// SDK's validation (command name known, count within the request limit, valid
/// JSON and matching parameter schema). Any unvalidated batch must be reported
/// as one of the other states, so callers can treat Success as "safe to
/// prepare".
enum class CompletionStatus {
  Success,
  NoCommandCalls,
  InvalidCommandCalls,
  Truncated,
  Refused,
  Unsupported,
  ProviderError,
};

enum class ErrorKind {
  Unknown,
  Network,
  Timeout,
  Auth,
  RateLimit,
  InvalidRequest,
  Server,
  Parse,
  Unsupported,
};

struct ModelError {
  ErrorKind kind = ErrorKind::Unknown;
  std::string message;
  int http_status = 0;  // 0 when the failure was not an HTTP response
  bool retryable = false;
  std::string raw_body;  // provider payload, diagnostics only
};

/// Everything one completion produced: the status a caller branches on, the
/// validated command calls, and the protocol data around them.
struct CompletionResult {
  CompletionStatus status = CompletionStatus::ProviderError;
  std::vector<CommandCall> command_calls;
  Usage usage;
  std::optional<ModelError> error;

  // Protocol data kept for diagnostics and caller-side accounting.
  FinishReason finish_reason = FinishReason::Unknown;
  std::string model;
  std::string provider_metadata;  // raw provider response (JSON text)

  /// Natural-language output, when the provider produced any. Diagnostic
  /// only: it is never the success signal and is never validated.
  std::string assistant_text;

  /// Non-fatal notes about how this completion was handled, e.g. that a
  /// provider-side guarantee was downgraded to SDK-side validation.
  std::vector<std::string> warnings;

  /// HTTP attempts this *single* logical completion needed (1 = no retry,
  /// 0 = nothing was sent because the request was refused before sending).
  /// Retries never turn into extra reasoning rounds.
  int http_attempts = 0;

  bool IsSuccess() const { return status == CompletionStatus::Success; }
  explicit operator bool() const { return IsSuccess(); }
  bool HasCommandCalls() const { return !command_calls.empty(); }

  /// Error message if one is set, otherwise a description of the status.
  std::string ErrorMessage() const;

  /// Assistant message for the next request's context (text + command calls).
  /// Building it is the caller's decision; the SDK never appends it itself.
  Message ToAssistantMessage() const;
};

std::string ToString(CompletionStatus status);
std::string ToString(ErrorKind kind);

}  // namespace llm
