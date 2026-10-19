// Unit tests for the logging seam: dispatch behaviour only. The SDK ships no
// logger, so the record collector here is local to the test.

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <llm/core/logger.h>

namespace llm {
namespace {

class CollectingLogger final : public ILogger {
 public:
  void Log(const LogRecord& record) override { records.push_back(record); }

  std::vector<LogRecord> records;
};

class ThrowingLogger final : public ILogger {
 public:
  void Log(const LogRecord&) override {
    throw std::runtime_error("sink is down");
  }
};

TEST(LlmLoggerTest, NoLoggerIsSilentAndSafe) {
  const std::shared_ptr<ILogger> none;
  EmitLog(none, LogLevel::Error, log_event::kRequestFailed, "ignored");
  SUCCEED();
}

TEST(LlmLoggerTest, DeliversTheRecordWithItsContext) {
  auto logger = std::make_shared<CollectingLogger>();
  LogContext context;
  context.session_id = "home";
  context.request_id = 7;

  EmitLog(logger, LogLevel::Info, log_event::kDecisionCompleted,
          "status=success, tools=2", context);

  ASSERT_EQ(logger->records.size(), 1u);
  const LogRecord& record = logger->records[0];
  EXPECT_EQ(record.level, LogLevel::Info);
  EXPECT_EQ(record.event, log_event::kDecisionCompleted);
  EXPECT_EQ(record.message, "status=success, tools=2");
  EXPECT_EQ(record.session_id, "home");
  EXPECT_EQ(record.request_id, 7u);
  EXPECT_EQ(record.event, "session.decision_completed")
      << "event names are part of the SDK's contract";
}

TEST(LlmLoggerTest, AThrowingLoggerCannotFailTheCaller) {
  auto logger = std::make_shared<ThrowingLogger>();
  EXPECT_NO_THROW(
      EmitLog(logger, LogLevel::Error, log_event::kRequestFailed, "boom"));
}

TEST(LlmLoggerTest, MissingContextIsReportedAsEmpty) {
  auto logger = std::make_shared<CollectingLogger>();
  EmitLog(logger, LogLevel::Debug, log_event::kDecisionStarted, "rev=1");

  ASSERT_EQ(logger->records.size(), 1u);
  EXPECT_TRUE(logger->records[0].session_id.empty());
  EXPECT_EQ(logger->records[0].request_id, 0u);
}

}  // namespace
}  // namespace llm
