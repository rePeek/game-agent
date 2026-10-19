// Tests for the session's log events: correlation, rejections, re-entrancy and
// the promise that logging can never change a decision.

#include "capture_logger.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <llm/context/session.h>
#include <llm/core/logger.h>
#include <llm/core/provider.h>

namespace llm {
namespace {

constexpr char kContextText[] = "Minute 62, opponent presses high.";

ToolDefinition PressingTool() {
  ToolDefinition tool;
  tool.name = "set_pressing_intensity";
  tool.description = "pressing";
  tool.parameters_schema_json =
      R"({"type":"object","properties":{"level":{"type":"integer"}},"required":["level"]})";
  return tool;
}

CompletionResult SuccessfulBatch() {
  CompletionResult result;
  result.status = CompletionStatus::Success;
  result.tool_calls = {
      ToolCall{"call_a", "set_pressing_intensity", R"({"level":4})"}};
  result.usage = Usage{10, 5, 15};
  result.http_attempts = 1;
  result.model = "gpt-5.6";
  return result;
}

class ScriptedProvider : public IModelProvider {
 public:
  CompletionResult next;
  std::string Name() const override { return "scripted"; }
  CompletionResult Complete(const ModelRequest&) override { return next; }
};

/// Logs by asking the session about itself: only safe because the SDK never
/// calls the logger while holding a session lock.
class ReentrantLogger final : public ILogger {
 public:
  explicit ReentrantLogger(std::shared_ptr<Session>* session)
      : session_(session) {}
  int calls = 0;

  void Log(const LogRecord&) override {
    ++calls;
    if (*session_) {
      (*session_)->pending_decision_count();
      (*session_)->in_flight_decision_count();
      (*session_)->current_revision();
      (*session_)->busy();
    }
  }

 private:
  std::shared_ptr<Session>* session_;
};

class ThrowingLogger final : public ILogger {
 public:
  void Log(const LogRecord&) override { throw std::runtime_error("sink down"); }
};

ContextUpdate Update(std::string revision) {
  ContextUpdate update;
  update.current_context = kContextText;
  update.recent_history = "possession 42%";
  update.context_revision = std::move(revision);
  return update;
}

ToolResult ResultFor(const std::string& call_id) {
  ToolResult result;
  result.tool_call_id = call_id;
  result.name = "set_pressing_intensity";
  result.content_json = R"({"applied":true})";
  return result;
}

struct Fixture {
  std::shared_ptr<ScriptedProvider> provider =
      std::make_shared<ScriptedProvider>();
  std::shared_ptr<CaptureLogger> logger = std::make_shared<CaptureLogger>();
  std::shared_ptr<Session> session;

  explicit Fixture(std::shared_ptr<ILogger> injected = nullptr) {
    SessionConfig config;
    config.model = "gpt-5.6";
    config.system_prompt = "You are the head coach.";
    config.session_id = "home";
    config.logger = injected != nullptr ? std::move(injected) : logger;
    session = CreateSession(provider, std::move(config));
    session->SetTools({PressingTool()});
  }
};

TEST(LlmSessionLoggingTest, LifecycleEventsCarryTheSessionAndRequestId) {
  Fixture fixture;
  fixture.provider->next = SuccessfulBatch();

  const DecisionResult decision = fixture.session->Complete(Update("rev-62"));
  ASSERT_TRUE(decision.completion.IsSuccess());
  fixture.session->RecordToolResults(decision.decision_id,
                                     {ResultFor("call_a")});

  const std::vector<LogRecord> records = fixture.logger->Records();
  ASSERT_EQ(records.size(), 4u);
  EXPECT_EQ(records[0].event, log_event::kSessionCreated);
  EXPECT_EQ(records[1].event, log_event::kDecisionStarted);
  EXPECT_EQ(records[2].event, log_event::kDecisionCompleted);
  EXPECT_EQ(records[3].event, log_event::kDecisionRecorded);

  for (const LogRecord& record : records) {
    EXPECT_EQ(record.session_id, "home");
  }
  EXPECT_EQ(records[0].request_id, 0u) << "session creation is not a decision";
  EXPECT_EQ(records[1].request_id, 1u);
  EXPECT_EQ(records[2].request_id, records[1].request_id)
      << "started and completed describe the same attempt";
  EXPECT_NE(records[1].message.find("rev=rev-62"), std::string::npos);
  EXPECT_NE(records[2].message.find("status=success"), std::string::npos);
  EXPECT_NE(records[2].message.find("tools=1"), std::string::npos);
  EXPECT_NE(records[2].message.find("tokens=15"), std::string::npos);
  EXPECT_NE(records[3].message.find("decision_id=1"), std::string::npos);
  EXPECT_NE(records[0].message.find("home"), std::string::npos);
}

TEST(LlmSessionLoggingTest, RejectionsAndDiscardsExplainThemselves) {
  Fixture fixture;
  fixture.provider->next = SuccessfulBatch();
  const DecisionResult first = fixture.session->Complete(Update("rev-62"));

  // Still awaiting results: the next decision is refused, and the record says
  // why (this is the "why was it Busy?" answer).
  const DecisionResult busy = fixture.session->Complete(Update("rev-63"));
  EXPECT_EQ(busy.status, DecisionStatus::Busy);
  const bool found_rejection =
      fixture.logger->Has(log_event::kDecisionRejected);
  ASSERT_TRUE(found_rejection);
  const std::vector<std::string> rejections =
      fixture.logger->Messages(log_event::kDecisionRejected);
  ASSERT_EQ(rejections.size(), 1u);
  EXPECT_NE(rejections[0].find("awaiting tool results"), std::string::npos);
  EXPECT_EQ(fixture.session->current_revision(), "rev-62")
      << "the refused update was not applied";

  // Discarding is a different event: it is a deliberate decision, not a
  // refusal.
  EXPECT_TRUE(fixture.session->DiscardDecision(first.decision_id));
  EXPECT_TRUE(fixture.logger->Has(log_event::kDecisionDiscarded));
  EXPECT_EQ(fixture.logger->Count(log_event::kDecisionRejected), 1u);

  // The next accepted decision gets the next request id.
  fixture.provider->next = SuccessfulBatch();
  const DecisionResult second = fixture.session->Complete(Update("rev-64"));
  const std::vector<std::string> started =
      fixture.logger->Messages(log_event::kDecisionStarted);
  ASSERT_EQ(started.size(), 2u);
  bool found = false;
  const LogRecord completed =
      fixture.logger->First(log_event::kDecisionCompleted, found);
  ASSERT_TRUE(found);
  EXPECT_EQ(completed.request_id, 1u);
  EXPECT_EQ(second.status, DecisionStatus::Completed);
}

TEST(LlmSessionLoggingTest, RecordRejectionsAreVisible) {
  Fixture fixture;
  fixture.provider->next = SuccessfulBatch();
  const DecisionResult decision = fixture.session->Complete(Update("rev-62"));

  EXPECT_EQ(fixture.session->RecordToolResults(999, {ResultFor("call_a")}),
            RecordOutcome::UnknownDecision);
  EXPECT_EQ(fixture.session->RecordToolResults(decision.decision_id, {}),
            RecordOutcome::IncompleteResults);

  const std::vector<std::string> rejections =
      fixture.logger->Messages(log_event::kDecisionRecordRejected);
  ASSERT_EQ(rejections.size(), 2u);
  EXPECT_NE(rejections[0].find("no decision with that id"), std::string::npos);
  EXPECT_NE(rejections[1].find("do not cover every tool call"),
            std::string::npos);
  EXPECT_FALSE(fixture.logger->Has(log_event::kDecisionRecorded));
}

TEST(LlmSessionLoggingTest, ClosedSessionsAreVisibleAndStillRefuse) {
  Fixture fixture;
  fixture.provider->next = SuccessfulBatch();
  fixture.session->Complete(Update("rev-62"));
  fixture.session->Close();

  EXPECT_TRUE(fixture.logger->Has(log_event::kSessionClosed));

  const DecisionResult refused = fixture.session->Complete(Update("rev-63"));
  EXPECT_EQ(refused.status, DecisionStatus::Closed);
  const std::vector<std::string> rejections =
      fixture.logger->Messages(log_event::kDecisionRejected);
  ASSERT_EQ(rejections.size(), 1u);
  EXPECT_NE(rejections[0].find("closed"), std::string::npos);
}

TEST(LlmSessionLoggingTest, AsynchronousDecisionsAreCorrelated) {
  Fixture fixture;
  fixture.provider->next = SuccessfulBatch();

  for (const char* revision : {"rev-62", "rev-63"}) {
    const DecisionResult decision =
        fixture.session->CompleteAsync(Update(revision)).get();
    ASSERT_TRUE(decision.completion.IsSuccess());
    ASSERT_EQ(fixture.session->RecordToolResults(decision.decision_id,
                                                 {ResultFor("call_a")}),
              RecordOutcome::Recorded);
  }

  const std::vector<LogRecord> records = fixture.logger->Records();
  std::vector<std::uint64_t> started;
  std::vector<std::uint64_t> completed;
  for (const LogRecord& record : records) {
    if (record.event == log_event::kDecisionStarted) {
      started.push_back(record.request_id);
    }
    if (record.event == log_event::kDecisionCompleted) {
      completed.push_back(record.request_id);
    }
  }
  EXPECT_EQ(started, (std::vector<std::uint64_t>{1, 2}));
  EXPECT_EQ(completed, started)
      << "each completion belongs to its own attempt, not to whichever "
         "worker finished first";
}

TEST(LlmSessionLoggingTest, OnlyMetadataIsLogged) {
  Fixture fixture;
  fixture.provider->next = SuccessfulBatch();
  const DecisionResult decision = fixture.session->Complete(Update("rev-62"));
  fixture.session->RecordToolResults(decision.decision_id,
                                     {ResultFor("call_a")});

  for (const LogRecord& record : fixture.logger->Records()) {
    EXPECT_EQ(record.message.find(kContextText), std::string::npos)
        << "the context text must not be logged: " << record.message;
    EXPECT_EQ(record.message.find("possession"), std::string::npos);
    EXPECT_EQ(record.message.find("level"), std::string::npos)
        << "tool arguments must not be logged";
  }
}

TEST(LlmSessionLoggingTest, NoLoggerIsTheDefaultAndChangesNothing) {
  Fixture fixture(std::make_shared<CaptureLogger>());
  fixture.provider->next = SuccessfulBatch();

  // The same flow without a logger: no records anywhere and no side effects.
  SessionConfig config;
  config.model = "gpt-5.6";
  auto silent = CreateSession(fixture.provider, std::move(config));
  silent->SetTools({PressingTool()});

  const DecisionResult decision = silent->Complete(Update("rev-62"));
  ASSERT_TRUE(decision.completion.IsSuccess());
  EXPECT_EQ(
      silent->RecordToolResults(decision.decision_id, {ResultFor("call_a")}),
      RecordOutcome::Recorded);
  EXPECT_TRUE(fixture.logger->Records().empty());
}

TEST(LlmSessionLoggingTest, LoggerMayCallBackIntoTheSession) {
  std::shared_ptr<Session> session;
  auto logger = std::make_shared<ReentrantLogger>(&session);

  SessionConfig config;
  config.model = "gpt-5.6";
  config.session_id = "home";
  config.logger = logger;

  auto provider = std::make_shared<ScriptedProvider>();
  provider->next = SuccessfulBatch();
  session = CreateSession(provider, std::move(config));
  session->SetTools({PressingTool()});

  const DecisionResult decision = session->Complete(Update("rev-62"));

  EXPECT_TRUE(decision.completion.IsSuccess())
      << "a logger that queries the session must not deadlock it";
  EXPECT_GT(logger->calls, 0);
  EXPECT_EQ(
      session->RecordToolResults(decision.decision_id, {ResultFor("call_a")}),
      RecordOutcome::Recorded);
  session.reset();  // the reentrant logger must not outlive the session
}

TEST(LlmSessionLoggingTest, AThrowingLoggerDoesNotChangeTheDecision) {
  Fixture fixture(std::make_shared<ThrowingLogger>());
  fixture.provider->next = SuccessfulBatch();

  const DecisionResult decision = fixture.session->Complete(Update("rev-62"));

  ASSERT_TRUE(decision.completion.IsSuccess());
  EXPECT_EQ(decision.decision_id, 1u);
  EXPECT_EQ(fixture.session->RecordToolResults(decision.decision_id,
                                               {ResultFor("call_a")}),
            RecordOutcome::Recorded);
  EXPECT_EQ(fixture.session->ContextSnapshot().decision_count(), 1u);
}

}  // namespace
}  // namespace llm
