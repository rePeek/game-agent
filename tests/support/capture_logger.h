#pragma once

/// Log collector for assertions.
///
/// Tests should not depend on someone reading console output: this logger keeps
/// every record so a test can assert that a lifecycle event happened, and in
/// which order, without relying on the timing of different workers.

#include <algorithm>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <llm/core/logger.h>

class CaptureLogger final : public llm::ILogger {
 public:
  void Log(const llm::LogRecord& record) override {
    std::lock_guard<std::mutex> lock(mutex_);
    records_.push_back(record);
  }

  std::vector<llm::LogRecord> Records() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return records_;
  }

  /// Number of records with this event name.
  std::size_t Count(std::string_view event) const {
    const std::vector<llm::LogRecord> records = Records();
    return static_cast<std::size_t>(std::count_if(
        records.begin(), records.end(), [event](const llm::LogRecord& record) {
          return record.event == event;
        }));
  }

  bool Has(std::string_view event) const { return Count(event) > 0; }

  /// First record with this event name; `found` reports whether there was one.
  llm::LogRecord First(std::string_view event, bool& found) const {
    for (const llm::LogRecord& record : Records()) {
      if (record.event == event) {
        found = true;
        return record;
      }
    }
    found = false;
    return {};
  }

  /// Messages of every record with this event name, in order.
  std::vector<std::string> Messages(std::string_view event) const {
    std::vector<std::string> messages;
    for (const llm::LogRecord& record : Records()) {
      if (record.event == event) {
        messages.push_back(record.message);
      }
    }
    return messages;
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    records_.clear();
  }

 private:
  mutable std::mutex mutex_;
  std::vector<llm::LogRecord> records_;
};
