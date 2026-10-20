#pragma once

/// Logging seam of the llm:: core.
///
/// The SDK defines only this interface and produces events; it ships no logger
/// implementation, no formatting, no file handling and no third-party logging
/// dependency. An application injects its own (a console printer in tests, its
/// own logging system in production) through AgentConfig or ProviderOptions.
///
/// Rules the SDK follows when it emits:
///  * a null logger means no logging at all;
///  * a logger that throws cannot change a decision's outcome (dispatch
///    swallows the exception);
///  * the logger is never called while a session lock is held, so an
///    implementation may call back into the SDK;
///  * only metadata is logged (status, counts, tokens, attempts, error
///    classification) — never prompts, API keys or raw responses;
///  * logging is not state management: CompletionResult and the session
///    history stay authoritative.

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace llm {

enum class LogLevel {
  Debug,
  Info,
  Warning,
  Error,
};

/// One log event.
///
/// `event` is a stable name (see llm::log_event), while `message` carries the
/// details. `session_id` and `request_id` make it possible to attribute an
/// event to the decision that caused it, which matters as soon as several
/// agents share one provider.
struct LogRecord {
  LogLevel level = LogLevel::Debug;
  std::string event;
  std::string message;
  std::string session_id;
  std::uint64_t request_id = 0;
};

class ILogger {
 public:
  virtual ~ILogger() = default;

  /// Called from whatever thread produced the event; implementations that write
  /// to a shared sink must synchronize.
  virtual void Log(const LogRecord& record) = 0;
};

/// Diagnostics identity that travels with a request, so provider events can be
/// attributed to the session and decision that caused them. Never part of the
/// provider protocol: request builders ignore it.
struct LogContext {
  std::string session_id;
  std::uint64_t request_id = 0;
};

/// Stable event names. Tests and downstream dashboards match on these.
namespace log_event {

// Session lifecycle.
inline constexpr std::string_view kSessionCreated = "session.created";
inline constexpr std::string_view kSessionClosed = "session.closed";
inline constexpr std::string_view kDecisionStarted = "session.decision_started";
inline constexpr std::string_view kDecisionRejected =
    "session.decision_rejected";
inline constexpr std::string_view kDecisionCompleted =
    "session.decision_completed";
inline constexpr std::string_view kDecisionRecorded =
    "session.decision_recorded";
inline constexpr std::string_view kDecisionDiscarded =
    "session.decision_discarded";
/// Tool results were reported for a decision that is not awaiting them, or
/// they did not cover every tool call.
inline constexpr std::string_view kDecisionRecordRejected =
    "session.decision_record_rejected";

// Provider lifecycle.
inline constexpr std::string_view kRequestStarted = "provider.request_started";
inline constexpr std::string_view kRequestRetry = "provider.retry";
inline constexpr std::string_view kRequestCompleted =
    "provider.request_completed";
inline constexpr std::string_view kRequestFailed = "provider.request_failed";
/// The request needed a guarantee the provider cannot make; the caller allowed
/// the SDK to validate instead.
inline constexpr std::string_view kRequirementDowngraded =
    "provider.requirement_downgraded";

// Command registration and preparation.
inline constexpr std::string_view kCommandRegistered = "commands.registered";
inline constexpr std::string_view kCommandRegistrationRejected =
    "commands.registration_rejected";
inline constexpr std::string_view kCommandsPrepared = "commands.prepared";
inline constexpr std::string_view kCommandsPrepareRejected =
    "commands.prepare_rejected";

// Validation.
inline constexpr std::string_view kValidationFailed =
    "commands.validation_failed";

}  // namespace log_event

/// Sends one record if a logger is configured.
///
/// Safe with a null logger, and it never throws: a failing logger must not turn
/// a working decision into a failure.
void EmitLog(const std::shared_ptr<ILogger>& logger,
             LogLevel level,
             std::string_view event,
             std::string message,
             const LogContext& context = {});

}  // namespace llm
