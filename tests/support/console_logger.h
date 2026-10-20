#pragma once

/// Console logger for tests and manual debugging: a thin printer, nothing more.
///
/// It lives in the test tree on purpose — the SDK ships no logger
/// implementation. Production code (Football Sim) provides its own adapter.

#include <iostream>
#include <mutex>

#include <llm/core/logger.h>

class ConsoleLogger final : public llm::ILogger {
 public:
  explicit ConsoleLogger(std::ostream& out = std::clog) : out_(out) {}

  void Log(const llm::LogRecord& record) override {
    // Agent workers can log concurrently, so the stream is guarded.
    std::lock_guard<std::mutex> lock(mutex_);
    // One statement per field: this formatting is stable across clang-format
    // versions, unlike a single wrapped stream chain.
    out_ << "[" << LevelName(record.level) << "] ";
    out_ << "[" << record.session_id << ":" << record.request_id << "] ";
    out_ << record.event << ": " << record.message << '\n';
  }

 private:
  static const char* LevelName(llm::LogLevel level) {
    switch (level) {
      case llm::LogLevel::Debug:
        return "DEBUG";
      case llm::LogLevel::Info:
        return "INFO";
      case llm::LogLevel::Warning:
        return "WARN";
      case llm::LogLevel::Error:
        return "ERROR";
    }
    return "UNKNOWN";
  }

  std::ostream& out_;
  std::mutex mutex_;
};
