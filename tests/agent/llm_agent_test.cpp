// Unit tests for llm::Agent: configuration, decisions, staging and isolation.
// The provider is scripted, so nothing here needs a network.

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <llm/agent.hpp>
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

/// Records what it was asked and answers with a scripted completion.
class ScriptedProvider : public IModelProvider {
 public:
  CompletionResult next = SuccessfulBatch();
  std::vector<ModelRequest> requests;

  std::string Name() const override { return "scripted"; }
  CompletionResult Complete(const ModelRequest& request) override {
    requests.push_back(request);
    return next;
  }
};

ContextUpdate Update(std::string revision, std::string context = "Minute 62.") {
  ContextUpdate update;
  update.current_context = std::move(context);
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

AgentConfig DefaultConfig() {
  AgentConfig config;
  config.id = "home";
  config.model = "scripted-model";
  config.system_prompt = "You are the head coach.";
  config.max_tokens = 300;
  config.max_tool_calls = 3;
  return config;
}

TEST(LlmAgentTest, DecisionIsAsynchronousAndStaged) {
  auto provider = std::make_shared<ScriptedProvider>();
  Agent agent(provider, DefaultConfig());
  agent.SetTools({PressingTool()});

  std::future<DecisionResult> future = agent.DecideAsync(Update("rev-62"));
  const DecisionResult decision = future.get();

  EXPECT_EQ(decision.status, DecisionStatus::Completed);
  ASSERT_TRUE(decision.completion.IsSuccess())
      << decision.completion.ErrorMessage();
  EXPECT_EQ(decision.decision_id, 1u);
  EXPECT_EQ(decision.context_revision, "rev-62");
  EXPECT_EQ(decision.completion.tool_calls[0].arguments_json, kRawArguments);

  // Staged, not recorded: the history moves only when results arrive.
  EXPECT_EQ(agent.PendingDecisionCount(), 1u);
  EXPECT_TRUE(agent.IsBusy());
  EXPECT_EQ(agent.ContextSnapshot().decision_count(), 0u);

  // The provider saw a request built from the agent: system prompt first,
  // then the rendered context, with the agent's tools and bounds.
  ASSERT_EQ(provider->requests.size(), 1u);
  const ModelRequest& request = provider->requests[0];
  EXPECT_EQ(request.model, "scripted-model");
  // system prompt (agent), recent history and current context (context)
  ASSERT_EQ(request.messages.size(), 3u);
  EXPECT_EQ(request.messages[0].role, Role::System);
  EXPECT_EQ(request.messages[0].content, "You are the head coach.");
  EXPECT_NE(request.messages[1].content.find("possession 42%"),
            std::string::npos);
  EXPECT_EQ(request.messages[2].role, Role::User);
  EXPECT_NE(request.messages[2].content.find("Minute 62."), std::string::npos);
  ASSERT_EQ(request.tools.size(), 1u);
  EXPECT_EQ(request.tools[0].name, "set_pressing_intensity");
  EXPECT_EQ(request.tool_choice.mode, ToolChoiceMode::Required);
  EXPECT_EQ(request.max_tokens, 300);
  EXPECT_EQ(request.max_tool_calls, 3u);
  EXPECT_EQ(request.log.session_id, "home");
  EXPECT_EQ(request.log.request_id, 1u);
}

TEST(LlmAgentTest, RecordingMovesTheHistoryForward) {
  auto provider = std::make_shared<ScriptedProvider>();
  Agent agent(provider, DefaultConfig());
  agent.SetTools({PressingTool()});

  const DecisionResult first = agent.DecideAsync(Update("rev-62")).get();
  ASSERT_TRUE(first.completion.IsSuccess());

  EXPECT_EQ(agent.RecordToolResults(first.decision_id, {ResultFor("call_a")}),
            RecordOutcome::Recorded);
  EXPECT_EQ(agent.PendingDecisionCount(), 0u);
  EXPECT_FALSE(agent.IsBusy());

  const Context context = agent.ContextSnapshot();
  ASSERT_EQ(context.decision_count(), 1u);
  EXPECT_EQ(context.decisions()[0].decision_id, first.decision_id);
  EXPECT_EQ(context.decisions()[0].tool_results.size(), 1u);

  // The next decision carries the confirmed round.
  const DecisionResult second = agent.DecideAsync(Update("rev-70")).get();
  ASSERT_TRUE(second.completion.IsSuccess());
  ASSERT_EQ(provider->requests.size(), 2u);
  const std::vector<Message>& messages = provider->requests[1].messages;
  // system, assistant(tool calls), tool result, recent history, current context
  ASSERT_EQ(messages.size(), 5u);
  EXPECT_EQ(messages[1].role, Role::Assistant);
  ASSERT_EQ(messages[1].tool_calls.size(), 1u);
  EXPECT_EQ(messages[1].tool_calls[0].arguments_json, kRawArguments);
  EXPECT_EQ(messages[2].role, Role::Tool);
  EXPECT_EQ(messages[2].tool_call_id, "call_a");
}

TEST(LlmAgentTest, OneDecisionAtATime) {
  auto provider = std::make_shared<ScriptedProvider>();
  Agent agent(provider, DefaultConfig());

  const DecisionResult first = agent.DecideAsync(Update("rev-62")).get();
  ASSERT_TRUE(first.completion.IsSuccess());

  const DecisionResult busy = agent.DecideAsync(Update("rev-63")).get();
  EXPECT_EQ(busy.status, DecisionStatus::Busy);
  EXPECT_EQ(busy.decision_id, 0u);
  EXPECT_EQ(provider->requests.size(), 1u)
      << "a refused decision must not reach the provider";

  EXPECT_TRUE(agent.DiscardDecision(first.decision_id));
  EXPECT_FALSE(agent.IsBusy());
  const DecisionResult after = agent.DecideAsync(Update("rev-63")).get();
  EXPECT_EQ(after.status, DecisionStatus::Completed);
  EXPECT_EQ(after.decision_id, 2u);
}

TEST(LlmAgentTest, IncompleteResultsRecordNothing) {
  auto provider = std::make_shared<ScriptedProvider>();
  Agent agent(provider, DefaultConfig());
  const DecisionResult decision = agent.DecideAsync(Update("rev-62")).get();
  ASSERT_TRUE(decision.completion.IsSuccess());

  EXPECT_EQ(agent.RecordToolResults(decision.decision_id, {}),
            RecordOutcome::IncompleteResults);
  EXPECT_EQ(agent.RecordToolResults(decision.decision_id,
                                    {ResultFor("call_unknown")}),
            RecordOutcome::IncompleteResults);
  EXPECT_EQ(agent.RecordToolResults(999, {ResultFor("call_a")}),
            RecordOutcome::UnknownDecision);

  EXPECT_EQ(agent.ContextSnapshot().decision_count(), 0u);
  EXPECT_EQ(agent.PendingDecisionCount(), 1u);
  EXPECT_EQ(
      agent.RecordToolResults(decision.decision_id, {ResultFor("call_a")}),
      RecordOutcome::Recorded);
}

TEST(LlmAgentTest, FailedDecisionsAreNotStaged) {
  auto provider = std::make_shared<ScriptedProvider>();
  provider->next.status = CompletionStatus::NoToolCalls;
  Agent agent(provider, DefaultConfig());

  const DecisionResult decision = agent.DecideAsync(Update("rev-62")).get();

  EXPECT_EQ(decision.status, DecisionStatus::Completed);
  EXPECT_EQ(decision.completion.status, CompletionStatus::NoToolCalls);
  EXPECT_EQ(decision.decision_id, 0u);
  EXPECT_EQ(agent.PendingDecisionCount(), 0u);
  EXPECT_FALSE(agent.IsBusy());
}

TEST(LlmAgentTest, AgentsShareAProviderButNotAContext) {
  auto provider = std::make_shared<ScriptedProvider>();

  AgentConfig home_config = DefaultConfig();
  Agent home(provider, std::move(home_config));
  AgentConfig away_config = DefaultConfig();
  away_config.id = "away";
  Agent away(provider, std::move(away_config));

  const DecisionResult home_decision = home.DecideAsync(Update("rev-h")).get();
  ASSERT_TRUE(home_decision.completion.IsSuccess());
  EXPECT_EQ(
      home.RecordToolResults(home_decision.decision_id, {ResultFor("call_a")}),
      RecordOutcome::Recorded);

  const DecisionResult away_decision = away.DecideAsync(Update("rev-a")).get();
  ASSERT_TRUE(away_decision.completion.IsSuccess());

  EXPECT_EQ(home.ContextSnapshot().decision_count(), 1u);
  EXPECT_EQ(away.ContextSnapshot().decision_count(), 0u);
  EXPECT_EQ(home.current_revision(), "rev-h");
  EXPECT_EQ(away.current_revision(), "rev-a");

  // Each agent's request carries its own identity and history.
  ASSERT_EQ(provider->requests.size(), 2u);
  EXPECT_EQ(provider->requests[0].log.session_id, "home");
  EXPECT_EQ(provider->requests[1].log.session_id, "away");
  EXPECT_EQ(provider->requests[1].messages.size(), 3u)
      << "no history leaked between agents (system + history + context)";
}

TEST(LlmAgentTest, ConfigurationIsMutable) {
  auto provider = std::make_shared<ScriptedProvider>();
  Agent agent(provider, DefaultConfig());

  agent.SetSystemPrompt("You are the assistant coach.");
  agent.SetModel("another-model");
  agent.SetTools({});
  EXPECT_EQ(agent.model(), "another-model");
  EXPECT_TRUE(agent.tools().empty());

  const DecisionResult decision = agent.DecideAsync(Update("rev-5")).get();
  ASSERT_TRUE(decision.completion.IsSuccess());

  const ModelRequest& request = provider->requests[0];
  EXPECT_EQ(request.model, "another-model");
  EXPECT_TRUE(request.tools.empty());
  EXPECT_EQ(request.messages[0].content, "You are the assistant coach.");
}

TEST(LlmAgentTest, CloseRefusesFurtherDecisions) {
  auto provider = std::make_shared<ScriptedProvider>();
  Agent agent(provider, DefaultConfig());
  const DecisionResult first = agent.DecideAsync(Update("rev-62")).get();

  agent.Close();
  EXPECT_TRUE(agent.IsClosed());

  const DecisionResult refused = agent.DecideAsync(Update("rev-63")).get();
  EXPECT_EQ(refused.status, DecisionStatus::Closed);
  EXPECT_EQ(provider->requests.size(), 1u);
  EXPECT_EQ(agent.RecordToolResults(first.decision_id, {ResultFor("call_a")}),
            RecordOutcome::Closed);
  EXPECT_EQ(agent.ContextSnapshot().decision_count(), 0u);
}

TEST(LlmAgentTest, WaitForIdleReportsWhenTheWorkerIsDone) {
  auto provider = std::make_shared<ScriptedProvider>();
  Agent agent(provider, DefaultConfig());

  EXPECT_TRUE(agent.WaitForIdle(std::chrono::milliseconds(0)));
  const DecisionResult decision = agent.DecideAsync(Update("rev-62")).get();
  EXPECT_EQ(agent.InFlightDecisionCount(), 0u);
  EXPECT_TRUE(agent.WaitForIdle(std::chrono::milliseconds(0)));
  EXPECT_EQ(decision.status, DecisionStatus::Completed);
}

}  // namespace
}  // namespace llm
