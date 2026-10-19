// Unit tests for llm::Session: provider injection, staging, atomic
// confirmation and per-session isolation.

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <llm/context/session.h>
#include <llm/core/message.h>
#include <llm/core/provider.h>
#include <llm/core/request.h>
#include <llm/core/result.h>
#include <llm/core/types.h>

namespace llm {
namespace {

constexpr char kRawArguments[] = R"({"level":3})";

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
      ToolCall{"call_a", "set_pressing_intensity", kRawArguments},
      ToolCall{"call_b", "set_defensive_line", R"({"height":"high"})"},
  };
  result.assistant_text = "Adjusting.";
  result.usage = Usage{10, 5, 15};
  return result;
}

/// Records what it was asked and answers with a scripted completion.
class ScriptedProvider : public IModelProvider {
 public:
  CompletionResult next;
  std::vector<ModelRequest> requests;

  std::string Name() const override { return "scripted"; }
  ProviderCapabilities Capabilities() const override {
    ProviderCapabilities capabilities;
    capabilities.forced_tool_choice = true;
    return capabilities;
  }
  CompletionResult Complete(const ModelRequest& request) override {
    requests.push_back(request);
    return next;
  }
};

/// Blocks inside Complete() until the test releases it.
class GatedProvider : public IModelProvider {
 public:
  CompletionResult next;

  std::string Name() const override { return "gated"; }
  CompletionResult Complete(const ModelRequest&) override {
    std::unique_lock<std::mutex> lock(mutex_);
    ++calls_;
    entered_cv_.notify_all();
    release_cv_.wait(lock, [this] { return released_; });
    return next;
  }

  /// Waits until the provider is inside Complete().
  bool WaitUntilEntered(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return entered_cv_.wait_for(lock, timeout, [this] { return calls_ > 0; });
  }
  void Release() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      released_ = true;
    }
    release_cv_.notify_all();
  }
  int calls() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return calls_;
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable entered_cv_;
  std::condition_variable release_cv_;
  int calls_ = 0;
  bool released_ = false;
};

ContextUpdate Update(std::string context, std::string revision) {
  ContextUpdate update;
  update.current_context = std::move(context);
  update.context_revision = std::move(revision);
  return update;
}

ToolResult ResultFor(const std::string& call_id) {
  ToolResult result;
  result.tool_call_id = call_id;
  result.name = "tool";
  result.content_json = R"({"applied":true})";
  return result;
}

struct Fixture {
  std::shared_ptr<ScriptedProvider> provider =
      std::make_shared<ScriptedProvider>();
  std::shared_ptr<Session> session;

  Fixture() {
    SessionConfig config;
    config.model = "gpt-5.6";
    config.system_prompt = "You are the head coach.";
    config.context.max_messages = 64;
    session = CreateSession(provider, std::move(config));
    session->SetTools({PressingTool()});
  }
};

TEST(LlmSessionTest, CompleteUsesTheContextAndStagesTheDecision) {
  Fixture fixture;
  fixture.provider->next = SuccessfulBatch();

  const DecisionResult decision =
      fixture.session->Complete(Update("Minute 62.", "rev-62"));

  EXPECT_EQ(decision.status, DecisionStatus::Completed);
  EXPECT_TRUE(decision.completion.IsSuccess());
  EXPECT_EQ(decision.decision_id, 1u);
  EXPECT_EQ(decision.context_revision, "rev-62");
  EXPECT_EQ(fixture.session->current_revision(), "rev-62");

  // The provider saw a request built from the session: system prompt, current
  // context, tools, forced tool calling.
  ASSERT_EQ(fixture.provider->requests.size(), 1u);
  const ModelRequest& request = fixture.provider->requests[0];
  EXPECT_EQ(request.model, "gpt-5.6");
  ASSERT_EQ(request.tools.size(), 1u);
  EXPECT_EQ(request.tool_choice.mode, ToolChoiceMode::Required);
  ASSERT_EQ(request.messages.size(), 2u);
  EXPECT_EQ(request.messages[0].role, Role::System);
  EXPECT_EQ(request.messages[0].content, "You are the head coach.");
  EXPECT_NE(request.messages[1].content.find("Minute 62."), std::string::npos);

  // Staged, not recorded: the history changes only when results arrive.
  EXPECT_EQ(fixture.session->pending_decision_count(), 1u);
  EXPECT_EQ(fixture.session->ContextSnapshot().decision_count(), 0u);
}

TEST(LlmSessionTest, RecordToolResultsCommitsAtomically) {
  Fixture fixture;
  fixture.provider->next = SuccessfulBatch();
  const DecisionResult decision =
      fixture.session->Complete(Update("Minute 62.", "rev-62"));

  const RecordOutcome outcome = fixture.session->RecordToolResults(
      decision.decision_id, {ResultFor("call_a"), ResultFor("call_b")});

  EXPECT_EQ(outcome, RecordOutcome::Recorded);
  EXPECT_EQ(fixture.session->pending_decision_count(), 0u);

  const Context context = fixture.session->ContextSnapshot();
  ASSERT_EQ(context.decision_count(), 1u);
  const DecisionRecord& record = context.decisions()[0];
  EXPECT_EQ(record.decision_id, decision.decision_id);
  EXPECT_EQ(record.context_revision, "rev-62");
  EXPECT_EQ(record.assistant.tool_calls.size(), 2u);
  ASSERT_EQ(record.tool_results.size(), 2u);
  EXPECT_EQ(record.tool_results[0].tool_call_id, "call_a");
  EXPECT_EQ(record.tool_results[1].tool_call_id, "call_b");
  EXPECT_EQ(record.tool_results[0].content, R"({"applied":true})");
}

TEST(LlmSessionTest, IncompleteResultsRecordNothing) {
  Fixture fixture;
  fixture.provider->next = SuccessfulBatch();
  const DecisionResult decision =
      fixture.session->Complete(Update("Minute 62.", "rev-62"));

  // Only one of two calls answered.
  const RecordOutcome missing = fixture.session->RecordToolResults(
      decision.decision_id, {ResultFor("call_a")});
  EXPECT_EQ(missing, RecordOutcome::IncompleteResults);

  // An unknown call id answers nothing.
  const RecordOutcome unknown = fixture.session->RecordToolResults(
      decision.decision_id, {ResultFor("call_a"), ResultFor("call_zzz")});
  EXPECT_EQ(unknown, RecordOutcome::IncompleteResults);

  // A duplicate result for the same call is not a match either.
  const RecordOutcome duplicate = fixture.session->RecordToolResults(
      decision.decision_id, {ResultFor("call_a"), ResultFor("call_a")});
  EXPECT_EQ(duplicate, RecordOutcome::IncompleteResults);

  // Nothing was recorded and the decision is still pending.
  EXPECT_EQ(fixture.session->ContextSnapshot().decision_count(), 0u);
  EXPECT_EQ(fixture.session->pending_decision_count(), 1u);

  // The complete set goes through afterwards.
  EXPECT_EQ(
      fixture.session->RecordToolResults(
          decision.decision_id, {ResultFor("call_a"), ResultFor("call_b")}),
      RecordOutcome::Recorded);
}

TEST(LlmSessionTest, ResultsAreStoredInCallOrderWhateverOrderTheyArrive) {
  Fixture fixture;
  fixture.provider->next = SuccessfulBatch();
  const DecisionResult decision =
      fixture.session->Complete(Update("Minute 62.", "rev-62"));

  EXPECT_EQ(
      fixture.session->RecordToolResults(
          decision.decision_id, {ResultFor("call_b"), ResultFor("call_a")}),
      RecordOutcome::Recorded);

  const DecisionRecord& record =
      fixture.session->ContextSnapshot().decisions()[0];
  EXPECT_EQ(record.tool_results[0].tool_call_id, "call_a");
  EXPECT_EQ(record.tool_results[1].tool_call_id, "call_b");
}

TEST(LlmSessionTest, UnknownDecisionIsReported) {
  Fixture fixture;
  EXPECT_EQ(fixture.session->RecordToolResults(42, {ResultFor("call_a")}),
            RecordOutcome::UnknownDecision);
  EXPECT_FALSE(fixture.session->DiscardDecision(42));
}

TEST(LlmSessionTest, OneDecisionAtATime) {
  Fixture fixture;
  fixture.provider->next = SuccessfulBatch();

  ASSERT_EQ(fixture.session->Complete(Update("Minute 62.", "rev-62")).status,
            DecisionStatus::Completed);

  // A second decision while the first awaits results would be taken with an
  // incomplete transcript, so it is refused.
  const DecisionResult busy =
      fixture.session->Complete(Update("Minute 63.", "rev-63"));
  EXPECT_EQ(busy.status, DecisionStatus::Busy);
  EXPECT_EQ(busy.decision_id, 0u);
  EXPECT_FALSE(busy.completion.IsSuccess());
  EXPECT_EQ(fixture.provider->requests.size(), 1u)
      << "a refused decision must not reach the provider";
  // The refused update was not applied either.
  EXPECT_EQ(fixture.session->current_revision(), "rev-62");

  // Discarding unblocks the session.
  EXPECT_TRUE(fixture.session->DiscardDecision(1));
  EXPECT_EQ(fixture.session->pending_decision_count(), 0u);
  const DecisionResult after =
      fixture.session->Complete(Update("Minute 63.", "rev-63"));
  EXPECT_EQ(after.status, DecisionStatus::Completed);
  EXPECT_EQ(after.decision_id, 2u);
}

TEST(LlmSessionTest, FailedDecisionsAreNotStaged) {
  Fixture fixture;
  fixture.provider->next.status = CompletionStatus::NoToolCalls;

  const DecisionResult decision =
      fixture.session->Complete(Update("Minute 62.", "rev-62"));

  EXPECT_EQ(decision.status, DecisionStatus::Completed);
  EXPECT_EQ(decision.completion.status, CompletionStatus::NoToolCalls);
  EXPECT_EQ(decision.decision_id, 0u);
  EXPECT_EQ(fixture.session->pending_decision_count(), 0u);
  EXPECT_EQ(fixture.session->ContextSnapshot().decision_count(), 0u);

  // The next round works without a discard, because nothing is pending.
  fixture.provider->next = SuccessfulBatch();
  EXPECT_EQ(fixture.session->Complete(Update("Minute 63.", "rev-63")).status,
            DecisionStatus::Completed);
}

TEST(LlmSessionTest, RecordedDecisionsAppearInTheNextRequest) {
  Fixture fixture;
  fixture.provider->next = SuccessfulBatch();
  const DecisionResult first =
      fixture.session->Complete(Update("Minute 62.", "rev-62"));
  fixture.session->RecordToolResults(
      first.decision_id, {ResultFor("call_a"), ResultFor("call_b")});

  fixture.provider->next = SuccessfulBatch();
  fixture.session->Complete(Update("Minute 70.", "rev-70"));

  ASSERT_EQ(fixture.provider->requests.size(), 2u);
  const std::vector<Message>& messages = fixture.provider->requests[1].messages;
  // system, assistant(tool calls), tool, tool, current context
  ASSERT_EQ(messages.size(), 5u);
  EXPECT_EQ(messages[1].role, Role::Assistant);
  ASSERT_EQ(messages[1].tool_calls.size(), 2u);
  EXPECT_EQ(messages[1].tool_calls[0].arguments_json, kRawArguments);
  EXPECT_EQ(messages[2].role, Role::Tool);
  EXPECT_EQ(messages[2].tool_call_id, "call_a");
  EXPECT_EQ(messages[3].tool_call_id, "call_b");
  EXPECT_NE(messages[4].content.find("Minute 70."), std::string::npos);
}

TEST(LlmSessionTest, SessionsAreIsolated) {
  Fixture first;
  Fixture second;
  first.provider->next = SuccessfulBatch();
  second.provider->next = SuccessfulBatch();

  const DecisionResult first_decision =
      first.session->Complete(Update("Minute 10.", "rev-a"));
  first.session->RecordToolResults(first_decision.decision_id,
                                   {ResultFor("call_a"), ResultFor("call_b")});

  EXPECT_EQ(second.session->Complete(Update("Minute 80.", "rev-b")).decision_id,
            1u);
  EXPECT_EQ(second.session->ContextSnapshot().decision_count(), 0u);
  EXPECT_EQ(second.session->current_revision(), "rev-b");

  EXPECT_EQ(first.session->ContextSnapshot().decision_count(), 1u);
  EXPECT_EQ(first.session->current_revision(), "rev-a");

  const std::vector<Message>& second_messages =
      second.provider->requests[0].messages;
  ASSERT_EQ(second_messages.size(), 2u) << "no history leaked between sessions";
  EXPECT_NE(second_messages[1].content.find("Minute 80."), std::string::npos);
}

TEST(LlmSessionTest, ClosedSessionsRefuseDecisions) {
  Fixture fixture;
  fixture.provider->next = SuccessfulBatch();
  const DecisionResult decision =
      fixture.session->Complete(Update("Minute 62.", "rev-62"));

  fixture.session->Close();
  EXPECT_TRUE(fixture.session->closed());

  const DecisionResult refused =
      fixture.session->Complete(Update("Minute 63.", "rev-63"));
  EXPECT_EQ(refused.status, DecisionStatus::Closed);
  EXPECT_EQ(fixture.provider->requests.size(), 1u);

  // A pending decision cannot be confirmed through a closed session either.
  EXPECT_EQ(
      fixture.session->RecordToolResults(
          decision.decision_id, {ResultFor("call_a"), ResultFor("call_b")}),
      RecordOutcome::Closed);
  EXPECT_EQ(fixture.session->ContextSnapshot().decision_count(), 0u);
}

TEST(LlmSessionTest, PendingDecisionIdsAreVisible) {
  Fixture fixture;
  fixture.provider->next = SuccessfulBatch();
  const DecisionResult decision =
      fixture.session->Complete(Update("Minute 62.", "rev-62"));

  const std::vector<std::uint64_t> ids =
      fixture.session->pending_decision_ids();
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0], decision.decision_id);

  fixture.session->DiscardDecision(decision.decision_id);
  EXPECT_TRUE(fixture.session->pending_decision_ids().empty());
}

TEST(LlmSessionTest, ConfigurationIsMutable) {
  Fixture fixture;
  fixture.provider->next = SuccessfulBatch();

  fixture.session->SetSystemPrompt("You are the assistant coach.");
  fixture.session->SetModel("gpt-4.1");
  fixture.session->SetTools({});
  EXPECT_EQ(fixture.session->model(), "gpt-4.1");
  EXPECT_TRUE(fixture.session->tools().empty());

  fixture.session->Complete(Update("Minute 5.", "rev-5"));

  ASSERT_EQ(fixture.provider->requests.size(), 1u);
  const ModelRequest& request = fixture.provider->requests[0];
  EXPECT_EQ(request.model, "gpt-4.1");
  EXPECT_TRUE(request.tools.empty());
  EXPECT_EQ(request.messages[0].content, "You are the assistant coach.");
}

// ---------------------------------------------------------------------------
// CTX-3: asynchronous decisions
// ---------------------------------------------------------------------------
TEST(LlmSessionAsyncTest, SubmittingDoesNotBlockAndResolvesOnce) {
  auto provider = std::make_shared<GatedProvider>();
  provider->next = SuccessfulBatch();
  SessionConfig config;
  config.model = "gpt-5.6";
  auto session = CreateSession(provider, std::move(config));
  session->SetTools({PressingTool()});

  std::future<DecisionResult> future =
      session->CompleteAsync(Update("Minute 62.", "rev-62"));

  // The caller keeps running while the provider works.
  EXPECT_TRUE(provider->WaitUntilEntered(std::chrono::seconds(5)));
  EXPECT_EQ(session->in_flight_decision_count(), 1u);
  EXPECT_TRUE(session->busy());
  EXPECT_FALSE(session->WaitForIdle(std::chrono::milliseconds(0)));

  provider->Release();
  const DecisionResult decision = future.get();

  EXPECT_EQ(decision.status, DecisionStatus::Completed);
  EXPECT_TRUE(decision.completion.IsSuccess());
  EXPECT_EQ(decision.decision_id, 1u);
  EXPECT_EQ(decision.context_revision, "rev-62");
  EXPECT_EQ(session->in_flight_decision_count(), 0u);
  EXPECT_EQ(session->pending_decision_count(), 1u);
  EXPECT_TRUE(session->WaitForIdle(std::chrono::milliseconds(0)));
}

TEST(LlmSessionAsyncTest, OnlyOneDecisionRunsAtATime) {
  auto provider = std::make_shared<GatedProvider>();
  provider->next = SuccessfulBatch();
  SessionConfig config;
  config.model = "gpt-5.6";
  auto session = CreateSession(provider, std::move(config));

  std::future<DecisionResult> first =
      session->CompleteAsync(Update("Minute 62.", "rev-62"));
  ASSERT_TRUE(provider->WaitUntilEntered(std::chrono::seconds(5)));

  // A second asynchronous decision is refused immediately.
  const DecisionResult busy_async =
      session->CompleteAsync(Update("Minute 63.", "rev-63")).get();
  EXPECT_EQ(busy_async.status, DecisionStatus::Busy);
  EXPECT_EQ(busy_async.decision_id, 0u);

  // A synchronous decision is refused too: same rule, same bookkeeping.
  const DecisionResult busy_sync =
      session->Complete(Update("Minute 64.", "rev-64"));
  EXPECT_EQ(busy_sync.status, DecisionStatus::Busy);

  EXPECT_EQ(provider->calls(), 1);
  // The refused updates were not applied.
  EXPECT_EQ(session->current_revision(), "rev-62");

  provider->Release();
  EXPECT_TRUE(first.get().completion.IsSuccess());
}

TEST(LlmSessionAsyncTest, CloseDuringFlightDiscardsTheResult) {
  auto provider = std::make_shared<GatedProvider>();
  provider->next = SuccessfulBatch();
  SessionConfig config;
  config.model = "gpt-5.6";
  auto session = CreateSession(provider, std::move(config));

  std::future<DecisionResult> future =
      session->CompleteAsync(Update("Minute 62.", "rev-62"));
  ASSERT_TRUE(provider->WaitUntilEntered(std::chrono::seconds(5)));

  session->Close();
  provider->Release();

  const DecisionResult decision = future.get();  // always satisfied
  EXPECT_EQ(decision.status, DecisionStatus::Closed);
  EXPECT_EQ(decision.decision_id, 0u);
  EXPECT_EQ(session->pending_decision_count(), 0u);
  EXPECT_EQ(session->ContextSnapshot().decision_count(), 0u);
}

TEST(LlmSessionAsyncTest, SynchronousCloseStillAnswersTheFuture) {
  auto provider = std::make_shared<GatedProvider>();
  provider->next = SuccessfulBatch();
  SessionConfig config;
  config.model = "gpt-5.6";
  auto session = CreateSession(provider, std::move(config));

  std::future<DecisionResult> future =
      session->CompleteAsync(Update("Minute 62.", "rev-62"));
  session->Close();
  // Submitting after Close resolves immediately instead of hanging.
  const DecisionResult refused =
      session->CompleteAsync(Update("Minute 63.", "rev-63")).get();
  EXPECT_EQ(refused.status, DecisionStatus::Closed);

  provider->Release();
  EXPECT_EQ(future.get().status, DecisionStatus::Closed);
}

TEST(LlmSessionAsyncTest, AsynchronousDecisionsFeedTheNextRequest) {
  auto provider = std::make_shared<GatedProvider>();
  provider->next = SuccessfulBatch();
  SessionConfig config;
  config.model = "gpt-5.6";
  config.system_prompt = "You are the head coach.";
  auto session = CreateSession(provider, std::move(config));
  session->SetTools({PressingTool()});

  // Thread the gate open for the whole test: each decision resolves on its own.
  provider->Release();
  std::future<DecisionResult> first =
      session->CompleteAsync(Update("Minute 62.", "rev-62"));
  const DecisionResult first_decision = first.get();

  ASSERT_EQ(
      session->RecordToolResults(first_decision.decision_id,
                                 {ResultFor("call_a"), ResultFor("call_b")}),
      RecordOutcome::Recorded);

  std::future<DecisionResult> second =
      session->CompleteAsync(Update("Minute 70.", "rev-70"));
  const DecisionResult second_decision = second.get();

  EXPECT_EQ(second_decision.decision_id, 2u);
  EXPECT_TRUE(second_decision.completion.IsSuccess());
  EXPECT_EQ(session->ContextSnapshot().decision_count(), 1u);
}

TEST(LlmSessionAsyncTest, WorkerSurvivesRepeatedSessionLifetimes) {
  for (int i = 0; i < 5; ++i) {
    auto provider = std::make_shared<GatedProvider>();
    provider->next = SuccessfulBatch();
    SessionConfig config;
    config.model = "gpt-5.6";
    auto session = CreateSession(provider, std::move(config));
    provider->Release();
    EXPECT_TRUE(session->CompleteAsync(Update("Minute 1.", "rev-1"))
                    .get()
                    .completion.IsSuccess());
    // The destructor joins the worker; no thread is left behind.
  }
  SUCCEED();
}

TEST(LlmSessionAsyncTest, ProviderErrorsAreReportedNotThrown) {
  class ThrowingProvider : public IModelProvider {
   public:
    std::string Name() const override { return "throwing"; }
    CompletionResult Complete(const ModelRequest&) override {
      throw std::runtime_error("boom");
    }
  };

  SessionConfig config;
  config.model = "gpt-5.6";
  auto session = CreateSession(std::make_shared<ThrowingProvider>(), config);

  const DecisionResult decision =
      session->Complete(Update("Minute 1.", "rev-1"));

  EXPECT_EQ(decision.status, DecisionStatus::Completed);
  EXPECT_EQ(decision.completion.status, CompletionStatus::ProviderError);
  ASSERT_TRUE(decision.completion.error.has_value());
  EXPECT_NE(decision.completion.ErrorMessage().find("boom"), std::string::npos);
  EXPECT_EQ(session->pending_decision_count(), 0u);
}

}  // namespace
}  // namespace llm
