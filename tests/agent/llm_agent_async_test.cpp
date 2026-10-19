// Asynchronous Agent tests: submission, in-flight behaviour, closing,
// exception safety and worker lifetimes. Written against the gated provider so
// nothing depends on timing.

#include "capture_logger.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <llm/agent.hpp>
#include <llm/core/logger.h>
#include <llm/core/provider.h>

namespace llm {
namespace {

constexpr char kRawArguments[] = R"({"level":4})";

ToolDefinition PressingTool() {
  ToolDefinition tool;
  tool.name = "set_pressing_intensity";
  tool.description = "Pressing intensity";
  tool.parameters_schema_json =
      R"({"type":"object","properties":{"level":{"type":"integer"}},"required":["level"]})";
  tool.strict = true;
  return tool;
}

CompletionResult SuccessfulBatch() {
  CompletionResult result;
  result.status = CompletionStatus::Success;
  result.tool_calls = {
      ToolCall{"call_a", "set_pressing_intensity", kRawArguments}};
  result.usage = Usage{10, 5, 15};
  result.http_attempts = 1;
  result.model = "scripted-model";
  return result;
}

/// Blocks inside Complete() until the test releases it.
class GatedProvider : public IModelProvider {
 public:
  CompletionResult next = SuccessfulBatch();
  int calls = 0;

  std::string Name() const override { return "gated"; }
  CompletionResult Complete(const ModelRequest&) override {
    std::unique_lock<std::mutex> lock(mutex_);
    ++calls;
    entered_cv_.notify_all();
    release_cv_.wait(lock, [this] { return released_; });
    return next;
  }

  bool WaitUntilEntered(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return entered_cv_.wait_for(lock, timeout, [this] { return calls > 0; });
  }
  void Release() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      released_ = true;
    }
    release_cv_.notify_all();
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable entered_cv_;
  std::condition_variable release_cv_;
  bool released_ = false;
};

class ThrowingProvider : public IModelProvider {
 public:
  std::string Name() const override { return "throwing"; }
  CompletionResult Complete(const ModelRequest&) override {
    throw std::runtime_error("provider exploded");
  }
};

AgentConfig DefaultConfig() {
  AgentConfig config;
  config.id = "home";
  config.model = "scripted-model";
  config.system_prompt = "You are the head coach.";
  return config;
}

ContextUpdate Update(const std::string& revision) {
  ContextUpdate update;
  update.current_context = "Minute 62.";
  update.context_revision = revision;
  return update;
}

ToolResult ResultFor(const std::string& call_id) {
  ToolResult result;
  result.tool_call_id = call_id;
  result.name = "set_pressing_intensity";
  result.content_json = R"({"applied":true})";
  return result;
}

TEST(LlmAgentAsyncTest, SubmittingDoesNotBlockAndResolvesOnce) {
  auto provider = std::make_shared<GatedProvider>();
  Agent agent(provider, DefaultConfig());
  agent.SetTools({PressingTool()});

  agent.UpdateContext(Update("rev-62"));
  std::future<DecisionResult> future = agent.DecideAsync();

  // The caller keeps running while the provider works.
  ASSERT_TRUE(provider->WaitUntilEntered(std::chrono::seconds(5)));
  EXPECT_EQ(agent.InFlightDecisionCount(), 1u);
  EXPECT_TRUE(agent.IsBusy());
  EXPECT_FALSE(agent.WaitForIdle(std::chrono::milliseconds(0)));

  provider->Release();
  const DecisionResult decision = future.get();

  EXPECT_EQ(decision.status, DecisionStatus::Completed);
  ASSERT_TRUE(decision.completion.IsSuccess());
  EXPECT_EQ(decision.decision_id, 1u);
  EXPECT_EQ(decision.context_revision, "rev-62");
  EXPECT_EQ(agent.InFlightDecisionCount(), 0u);
  EXPECT_EQ(agent.PendingDecisionCount(), 1u);
  EXPECT_TRUE(agent.WaitForIdle(std::chrono::milliseconds(0)));
}

TEST(LlmAgentAsyncTest, OnlyOneDecisionRunsAtATime) {
  auto provider = std::make_shared<GatedProvider>();
  Agent agent(provider, DefaultConfig());

  agent.UpdateContext(Update("rev-62"));
  std::future<DecisionResult> first = agent.DecideAsync();
  ASSERT_TRUE(provider->WaitUntilEntered(std::chrono::seconds(5)));

  // A second submission while the first is in flight is refused immediately.
  agent.UpdateContext(Update("rev-63"));
  const DecisionResult busy = agent.DecideAsync().get();
  EXPECT_EQ(busy.status, DecisionStatus::Busy);
  EXPECT_EQ(busy.decision_id, 0u);
  EXPECT_EQ(provider->calls, 1);

  provider->Release();
  EXPECT_TRUE(first.get().completion.IsSuccess());
}

TEST(LlmAgentAsyncTest, CloseDuringFlightDiscardsTheResult) {
  auto provider = std::make_shared<GatedProvider>();
  Agent agent(provider, DefaultConfig());

  agent.UpdateContext(Update("rev-62"));
  std::future<DecisionResult> future = agent.DecideAsync();
  ASSERT_TRUE(provider->WaitUntilEntered(std::chrono::seconds(5)));

  agent.Close();
  provider->Release();

  const DecisionResult decision = future.get();  // always satisfied
  EXPECT_EQ(decision.status, DecisionStatus::Closed);
  EXPECT_EQ(decision.decision_id, 0u);
  EXPECT_EQ(agent.PendingDecisionCount(), 0u);
  EXPECT_EQ(agent.ContextSnapshot().decision_count(), 0u);
}

TEST(LlmAgentAsyncTest, SubmittingAfterCloseResolvesImmediately) {
  auto provider = std::make_shared<GatedProvider>();
  Agent agent(provider, DefaultConfig());

  agent.UpdateContext(Update("rev-62"));
  std::future<DecisionResult> future = agent.DecideAsync();
  agent.Close();
  EXPECT_EQ(agent.DecideAsync().get().status, DecisionStatus::Closed);

  provider->Release();
  EXPECT_EQ(future.get().status, DecisionStatus::Closed);
}

TEST(LlmAgentAsyncTest, AsynchronousDecisionsFeedTheNextRequest) {
  auto provider = std::make_shared<GatedProvider>();
  provider
      ->Release();  // the gate stays open: each decision resolves on its own

  Agent agent(provider, DefaultConfig());
  agent.SetTools({PressingTool()});

  agent.UpdateContext(Update("rev-62"));
  const DecisionResult first = agent.DecideAsync().get();
  ASSERT_TRUE(first.completion.IsSuccess());
  ASSERT_EQ(agent.RecordToolResults(first.decision_id, {ResultFor("call_a")}),
            RecordOutcome::Recorded);

  agent.UpdateContext(Update("rev-70"));
  const DecisionResult second = agent.DecideAsync().get();

  ASSERT_TRUE(second.completion.IsSuccess());
  EXPECT_EQ(second.decision_id, 2u);
  EXPECT_EQ(second.context_revision, "rev-70");
  EXPECT_EQ(agent.ContextSnapshot().decision_count(), 1u);
}

TEST(LlmAgentAsyncTest, WorkerSurvivesRepeatedAgentLifetimes) {
  for (int i = 0; i < 5; ++i) {
    auto provider = std::make_shared<GatedProvider>();
    provider->Release();
    Agent agent(provider, DefaultConfig());
    agent.UpdateContext(Update("rev-1"));
    EXPECT_TRUE(agent.DecideAsync().get().completion.IsSuccess());
    // The destructor joins the worker; no thread is left behind.
  }
  SUCCEED();
}

TEST(LlmAgentAsyncTest, ThrowingProviderIsReportedNotThrown) {
  Agent agent(std::make_shared<ThrowingProvider>(), DefaultConfig());

  agent.UpdateContext(Update("rev-1"));
  const DecisionResult decision = agent.DecideAsync().get();

  EXPECT_EQ(decision.status, DecisionStatus::Completed);
  EXPECT_EQ(decision.completion.status, CompletionStatus::ProviderError);
  ASSERT_TRUE(decision.completion.error.has_value());
  EXPECT_NE(decision.completion.ErrorMessage().find("provider exploded"),
            std::string::npos);
  EXPECT_EQ(agent.PendingDecisionCount(), 0u);
}

TEST(LlmAgentAsyncTest, ThrowingProviderLeavesTheAgentUsable) {
  // A provider that throws must not poison the agent: the next round works.
  class FlakyProvider final : public IModelProvider {
   public:
    int calls = 0;
    std::string Name() const override { return "flaky"; }
    CompletionResult Complete(const ModelRequest&) override {
      if (++calls == 1) {
        throw std::runtime_error("first call fails");
      }
      return SuccessfulBatch();
    }
  };

  auto provider = std::make_shared<FlakyProvider>();
  Agent agent(provider, DefaultConfig());
  agent.SetTools({PressingTool()});

  agent.UpdateContext(Update("rev-1"));
  EXPECT_EQ(agent.DecideAsync().get().completion.status,
            CompletionStatus::ProviderError);
  EXPECT_FALSE(agent.IsBusy());

  agent.UpdateContext(Update("rev-2"));
  const DecisionResult second = agent.DecideAsync().get();
  ASSERT_TRUE(second.completion.IsSuccess());
  EXPECT_EQ(second.decision_id, 1u);
}

TEST(LlmAgentAsyncTest, TwoAgentsRunIndependentlyOnOneProvider) {
  auto provider = std::make_shared<GatedProvider>();
  provider->Release();

  AgentConfig home_config = DefaultConfig();
  Agent home(provider, std::move(home_config));
  AgentConfig away_config = DefaultConfig();
  away_config.id = "away";
  Agent away(provider, std::move(away_config));
  home.SetTools({PressingTool()});
  away.SetTools({PressingTool()});

  home.UpdateContext(Update("rev-h"));
  away.UpdateContext(Update("rev-a"));

  // Both may be in flight at the same time: the "one at a time" rule is per
  // agent, not per provider.
  std::future<DecisionResult> home_future = home.DecideAsync();
  std::future<DecisionResult> away_future = away.DecideAsync();
  const DecisionResult home_decision = home_future.get();
  const DecisionResult away_decision = away_future.get();

  ASSERT_TRUE(home_decision.completion.IsSuccess());
  ASSERT_TRUE(away_decision.completion.IsSuccess());
  EXPECT_EQ(home_decision.context_revision, "rev-h");
  EXPECT_EQ(away_decision.context_revision, "rev-a");
  EXPECT_EQ(home_decision.decision_id, 1u);
  EXPECT_EQ(away_decision.decision_id, 1u);
  EXPECT_EQ(provider->calls, 2);
}

TEST(LlmAgentAsyncTest, DisablingTheLoggerCannotBreakADecision) {
  // A logger that throws on every event, on both the agent and the provider.
  class ThrowingLogger final : public ILogger {
   public:
    void Log(const LogRecord&) override {
      throw std::runtime_error("sink down");
    }
  };

  auto logger = std::make_shared<ThrowingLogger>();
  auto provider = std::make_shared<GatedProvider>();
  provider->Release();

  AgentConfig config = DefaultConfig();
  config.logger = logger;
  Agent agent(provider, std::move(config));
  agent.SetTools({PressingTool()});

  agent.UpdateContext(Update("rev-62"));
  const DecisionResult decision = agent.DecideAsync().get();

  ASSERT_EQ(decision.status, DecisionStatus::Completed);
  ASSERT_TRUE(decision.completion.IsSuccess());
  EXPECT_EQ(
      agent.RecordToolResults(decision.decision_id, {ResultFor("call_a")}),
      RecordOutcome::Recorded);
  EXPECT_EQ(agent.ContextSnapshot().decision_count(), 1u);
}

}  // namespace
}  // namespace llm
