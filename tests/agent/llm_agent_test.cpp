// Unit tests for llm::Agent: configuration, decisions, staging and isolation.
// The provider is scripted, so nothing here needs a network.

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <llm/agent.hpp>
#include <llm/core/provider.h>

namespace llm {
namespace {

constexpr char kRawArguments[] = R"({"level":4})";

CommandDefinition PressingTool() {
  CommandDefinition tool;
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
  result.command_calls = {
      CommandCall{"call_a", "set_pressing_intensity", kRawArguments}};
  result.assistant_text = "Adjusting.";
  result.usage = Usage{10, 5, 15};
  result.http_attempts = 1;
  result.model = "scripted-model";
  return result;
}

/// A batch with two calls, used to check result bookkeeping.
CompletionResult TwoCallBatch() {
  CompletionResult result;
  result.status = CompletionStatus::Success;
  result.command_calls = {
      CommandCall{"call_a", "set_pressing_intensity", kRawArguments}};
  result.command_calls.push_back(
      CommandCall{"call_b", "set_defensive_line", R"({"height":"high"})"});
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

/// Blocks inside Complete() until the test releases it, so in-flight behaviour
/// can be tested without sleeping.
class GatedProvider : public IModelProvider {
 public:
  CompletionResult next = SuccessfulBatch();
  std::vector<ModelRequest> requests;

  std::string Name() const override { return "gated"; }
  CompletionResult Complete(const ModelRequest& request) override {
    std::unique_lock<std::mutex> lock(mutex_);
    requests.push_back(request);
    ++calls_;
    entered_cv_.notify_all();
    release_cv_.wait(lock, [this] { return released_; });
    return next;
  }

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

 private:
  mutable std::mutex mutex_;
  std::condition_variable entered_cv_;
  std::condition_variable release_cv_;
  int calls_ = 0;
  bool released_ = false;
};

ContextUpdate Update(std::string revision, std::string context = "Minute 62.") {
  ContextUpdate update;
  update.current_context = std::move(context);
  update.recent_history = "possession 42%";
  update.context_revision = std::move(revision);
  return update;
}

CommandResult ResultFor(const std::string& call_id,
                      const std::string& name = "set_pressing_intensity") {
  CommandResult result;
  result.command_call_id = call_id;
  result.name = name;
  result.outcome.success = true;
  result.outcome.content_json = R"({"applied":true})";
  return result;
}

/// One decision round: update the context, then ask.
DecisionResult Decide(Agent& agent, ContextUpdate update) {
  agent.UpdateContext(std::move(update));
  return agent.DecideAsync().get();
}

AgentConfig DefaultConfig() {
  AgentConfig config;
  config.id = "home";
  config.model = "scripted-model";
  config.system_prompt = "You are the head coach.";
  config.max_output_tokens = 300;
  config.max_command_calls = 3;
  return config;
}

TEST(LlmAgentTest, DecisionIsAsynchronousAndStaged) {
  auto provider = std::make_shared<ScriptedProvider>();
  Agent agent(provider, DefaultConfig());
  agent.SetCommands({PressingTool()});

  agent.UpdateContext(Update("rev-62"));
  std::future<DecisionResult> future = agent.DecideAsync();
  const DecisionResult decision = future.get();

  EXPECT_EQ(decision.status, DecisionStatus::Completed);
  ASSERT_TRUE(decision.completion.IsSuccess())
      << decision.completion.ErrorMessage();
  EXPECT_EQ(decision.decision_id, 1u);
  EXPECT_EQ(decision.context_revision, "rev-62");
  EXPECT_EQ(decision.completion.command_calls[0].arguments_json, kRawArguments);

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
  ASSERT_EQ(request.commands.size(), 1u);
  EXPECT_EQ(request.commands[0].name, "set_pressing_intensity");
  EXPECT_EQ(request.command_choice.mode, CommandChoiceMode::Required);
  EXPECT_EQ(request.max_tokens, 300);
  EXPECT_EQ(request.max_command_calls, 3u);
  EXPECT_EQ(request.log.session_id, "home");
  EXPECT_EQ(request.log.request_id, 1u);
}

TEST(LlmAgentTest, RecordingMovesTheHistoryForward) {
  auto provider = std::make_shared<ScriptedProvider>();
  Agent agent(provider, DefaultConfig());
  agent.SetCommands({PressingTool()});

  const DecisionResult first = Decide(agent, Update("rev-62"));
  ASSERT_TRUE(first.completion.IsSuccess());

  EXPECT_EQ(agent.RecordCommandResults(first.decision_id, {ResultFor("call_a")}),
            RecordOutcome::Recorded);
  EXPECT_EQ(agent.PendingDecisionCount(), 0u);
  EXPECT_FALSE(agent.IsBusy());

  const Context context = agent.ContextSnapshot();
  ASSERT_EQ(context.decision_count(), 1u);
  EXPECT_EQ(context.decisions()[0].decision_id, first.decision_id);
  EXPECT_EQ(context.decisions()[0].command_results.size(), 1u);

  // The next decision carries the confirmed round.
  const DecisionResult second = Decide(agent, Update("rev-70"));
  ASSERT_TRUE(second.completion.IsSuccess());
  ASSERT_EQ(provider->requests.size(), 2u);
  const std::vector<Message>& messages = provider->requests[1].messages;
  // system, assistant(tool calls), tool result, recent history, current context
  ASSERT_EQ(messages.size(), 5u);
  EXPECT_EQ(messages[1].role, Role::Assistant);
  ASSERT_EQ(messages[1].command_calls.size(), 1u);
  EXPECT_EQ(messages[1].command_calls[0].arguments_json, kRawArguments);
  EXPECT_EQ(messages[2].role, Role::Tool);
  EXPECT_EQ(messages[2].command_call_id, "call_a");
}

TEST(LlmAgentTest, OneDecisionAtATime) {
  auto provider = std::make_shared<ScriptedProvider>();
  Agent agent(provider, DefaultConfig());

  const DecisionResult first = Decide(agent, Update("rev-62"));
  ASSERT_TRUE(first.completion.IsSuccess());

  const DecisionResult busy = Decide(agent, Update("rev-63"));
  EXPECT_EQ(busy.status, DecisionStatus::Busy);
  EXPECT_EQ(busy.decision_id, 0u);
  EXPECT_EQ(provider->requests.size(), 1u)
      << "a refused decision must not reach the provider";

  EXPECT_TRUE(agent.DiscardDecision(first.decision_id));
  EXPECT_FALSE(agent.IsBusy());
  const DecisionResult after = Decide(agent, Update("rev-63"));
  EXPECT_EQ(after.status, DecisionStatus::Completed);
  EXPECT_EQ(after.decision_id, 2u);
}

TEST(LlmAgentTest, IncompleteResultsRecordNothing) {
  auto provider = std::make_shared<ScriptedProvider>();
  Agent agent(provider, DefaultConfig());
  const DecisionResult decision = Decide(agent, Update("rev-62"));
  ASSERT_TRUE(decision.completion.IsSuccess());

  EXPECT_EQ(agent.RecordCommandResults(decision.decision_id, {}),
            RecordOutcome::IncompleteResults);
  EXPECT_EQ(agent.RecordCommandResults(decision.decision_id,
                                    {ResultFor("call_unknown")}),
            RecordOutcome::IncompleteResults);
  EXPECT_EQ(agent.RecordCommandResults(999, {ResultFor("call_a")}),
            RecordOutcome::UnknownDecision);

  EXPECT_EQ(agent.ContextSnapshot().decision_count(), 0u);
  EXPECT_EQ(agent.PendingDecisionCount(), 1u);
  EXPECT_EQ(
      agent.RecordCommandResults(decision.decision_id, {ResultFor("call_a")}),
      RecordOutcome::Recorded);
}

TEST(LlmAgentTest, FailedDecisionsAreNotStaged) {
  auto provider = std::make_shared<ScriptedProvider>();
  provider->next.status = CompletionStatus::NoCommandCalls;
  Agent agent(provider, DefaultConfig());

  const DecisionResult decision = Decide(agent, Update("rev-62"));

  EXPECT_EQ(decision.status, DecisionStatus::Completed);
  EXPECT_EQ(decision.completion.status, CompletionStatus::NoCommandCalls);
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

  const DecisionResult home_decision = Decide(home, Update("rev-h"));
  ASSERT_TRUE(home_decision.completion.IsSuccess());
  EXPECT_EQ(
      home.RecordCommandResults(home_decision.decision_id, {ResultFor("call_a")}),
      RecordOutcome::Recorded);

  const DecisionResult away_decision = Decide(away, Update("rev-a"));
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
  agent.SetCommands({});
  EXPECT_EQ(agent.model(), "another-model");
  EXPECT_TRUE(agent.Commands().empty());

  const DecisionResult decision = Decide(agent, Update("rev-5"));
  ASSERT_TRUE(decision.completion.IsSuccess());

  const ModelRequest& request = provider->requests[0];
  EXPECT_EQ(request.model, "another-model");
  EXPECT_TRUE(request.commands.empty());
  EXPECT_EQ(request.messages[0].content, "You are the assistant coach.");
}

TEST(LlmAgentTest, CloseRefusesFurtherDecisions) {
  auto provider = std::make_shared<ScriptedProvider>();
  Agent agent(provider, DefaultConfig());
  const DecisionResult first = Decide(agent, Update("rev-62"));

  agent.Close();
  EXPECT_TRUE(agent.IsClosed());

  const DecisionResult refused = Decide(agent, Update("rev-63"));
  EXPECT_EQ(refused.status, DecisionStatus::Closed);
  EXPECT_EQ(provider->requests.size(), 1u);
  EXPECT_EQ(agent.RecordCommandResults(first.decision_id, {ResultFor("call_a")}),
            RecordOutcome::Closed);
  EXPECT_EQ(agent.ContextSnapshot().decision_count(), 0u);
}

TEST(LlmAgentTest, WaitForIdleReportsWhenTheWorkerIsDone) {
  auto provider = std::make_shared<ScriptedProvider>();
  Agent agent(provider, DefaultConfig());

  EXPECT_TRUE(agent.WaitForIdle(std::chrono::milliseconds(0)));
  const DecisionResult decision = Decide(agent, Update("rev-62"));
  EXPECT_EQ(agent.InFlightDecisionCount(), 0u);
  EXPECT_TRUE(agent.WaitForIdle(std::chrono::milliseconds(0)));
  EXPECT_EQ(decision.status, DecisionStatus::Completed);
}

TEST(LlmAgentTest, ContextUpdatesDuringFlightApplyToTheNextDecision) {
  auto provider = std::make_shared<GatedProvider>();
  Agent agent(provider, DefaultConfig());
  agent.SetCommands({PressingTool()});

  agent.UpdateContext(Update("rev-62", "Minute 62."));
  std::future<DecisionResult> future = agent.DecideAsync();
  ASSERT_TRUE(provider->WaitUntilEntered(std::chrono::seconds(5)));

  // The game keeps updating the world while the model is thinking. That is
  // allowed, and it cannot change the request already sent.
  agent.UpdateContext(Update("rev-70", "Minute 70."));
  EXPECT_EQ(agent.current_revision(), "rev-70");
  EXPECT_TRUE(agent.IsBusy());

  provider->Release();
  const DecisionResult first = future.get();
  ASSERT_TRUE(first.completion.IsSuccess());
  EXPECT_EQ(first.context_revision, "rev-62")
      << "the decision reports the snapshot it was taken from";

  // ... and the next decision sees the newer state.
  ASSERT_EQ(agent.RecordCommandResults(first.decision_id, {ResultFor("call_a")}),
            RecordOutcome::Recorded);
  const DecisionResult second = agent.DecideAsync().get();
  ASSERT_TRUE(second.completion.IsSuccess());
  EXPECT_EQ(second.context_revision, "rev-70");

  ASSERT_EQ(provider->requests.size(), 2u);
  const std::vector<Message>& messages = provider->requests[1].messages;
  ASSERT_FALSE(messages.empty());
  EXPECT_NE(messages.back().content.find("Minute 70."), std::string::npos)
      << "the next request carries the updated context";
}

TEST(LlmAgentTest, ARefusedDecisionStillUsesTheLatestUpdate) {
  auto provider = std::make_shared<ScriptedProvider>();
  Agent agent(provider, DefaultConfig());

  const DecisionResult first = Decide(agent, Update("rev-62", "Minute 62."));
  ASSERT_TRUE(first.completion.IsSuccess());

  // Refused as Busy: the update is recorded either way, because UpdateContext
  // is about the world state, not about the request.
  const DecisionResult busy = Decide(agent, Update("rev-63", "Minute 63."));
  EXPECT_EQ(busy.status, DecisionStatus::Busy);
  EXPECT_EQ(agent.current_revision(), "rev-63");

  ASSERT_EQ(agent.DiscardDecision(first.decision_id), true);
  const DecisionResult after = agent.DecideAsync().get();
  ASSERT_TRUE(after.completion.IsSuccess());
  EXPECT_EQ(after.context_revision, "rev-63")
      << "a refused decision does not swallow the update";
}

TEST(LlmAgentTest, ResultsAreStoredInCallOrderWhateverOrderTheyArrive) {
  auto provider = std::make_shared<ScriptedProvider>();
  provider->next = TwoCallBatch();
  Agent agent(provider, DefaultConfig());

  const DecisionResult decision = Decide(agent, Update("rev-62"));
  ASSERT_TRUE(decision.completion.IsSuccess());
  ASSERT_EQ(decision.completion.command_calls.size(), 2u);

  // The game may execute in whatever order it likes (here reversed); the
  // transcript is normalized to the order the model asked in, so every tool
  // turn sits next to its call in a deterministic position.
  EXPECT_EQ(agent.RecordCommandResults(
                decision.decision_id,
                {ResultFor("call_b", "set_defensive_line"),
                 ResultFor("call_a")}),
            RecordOutcome::Recorded);

  // Keep the snapshot alive: ContextSnapshot() returns a copy.
  const Context context = agent.ContextSnapshot();
  const DecisionRecord& record = context.decisions()[0];
  ASSERT_EQ(record.assistant.command_calls.size(), 2u);
  EXPECT_EQ(record.assistant.command_calls[0].id, "call_a");
  EXPECT_EQ(record.assistant.command_calls[1].id, "call_b");
  ASSERT_EQ(record.command_results.size(), 2u);
  EXPECT_EQ(record.command_results[0].command_call_id, "call_a");
  EXPECT_EQ(record.command_results[1].command_call_id, "call_b");
}

TEST(LlmAgentTest, ResultsThatNameTheWrongCommandAreRejected) {
  auto provider = std::make_shared<ScriptedProvider>();
  provider->next = TwoCallBatch();
  Agent agent(provider, DefaultConfig());
  ASSERT_TRUE(agent
                  .RegisterCommand(
                      "set_pressing_intensity", "",
                      Params(Param<int>("level").Required()),
                      [](int) { return CommandOutcome{true, {}, {}}; })
                  .ok());
  ASSERT_TRUE(agent
                  .RegisterCommand(
                      "set_defensive_line", "",
                      Params(Param<std::string>("height").Required()),
                      [](const std::string&) {
                        return CommandOutcome{true, {}, {}};
                      })
                  .ok());

  const DecisionResult decision = Decide(agent, Update("rev-62"));
  ASSERT_TRUE(decision.completion.IsSuccess());

  // call_b is reported under the wrong command name.
  EXPECT_EQ(agent.RecordCommandResults(
                decision.decision_id,
                {ResultFor("call_a"), ResultFor("call_b")}),
            RecordOutcome::NameMismatch);
  EXPECT_EQ(agent.PendingDecisionCount(), 1u) << "nothing was recorded";
  EXPECT_TRUE(agent.DiscardDecision(decision.decision_id));
}

TEST(LlmAgentTest, FailedResultsAreRecordedAsExecutedFacts) {
  auto provider = std::make_shared<ScriptedProvider>();
  provider->next = SuccessfulBatch();
  Agent agent(provider, DefaultConfig());
  ASSERT_TRUE(agent
                  .RegisterCommand(
                      "set_pressing_intensity", "",
                      Params(Param<int>("level").Required()),
                      [](int) { return CommandOutcome{true, {}, {}}; })
                  .ok());

  const DecisionResult decision = Decide(agent, Update("rev-62"));
  ASSERT_TRUE(decision.completion.IsSuccess());

  CommandResult failed;
  failed.command_call_id = "call_a";
  failed.name = "set_pressing_intensity";
  failed.outcome.success = false;
  failed.outcome.error_message = "rejected_by_referee";
  EXPECT_EQ(agent.RecordCommandResults(decision.decision_id, {failed}),
            RecordOutcome::Recorded);

  const Context context = agent.ContextSnapshot();
  ASSERT_EQ(context.decisions().size(), 1u);
  const Message& message = context.decisions()[0].command_results[0];
  EXPECT_EQ(message.content, R"({"ok":false,"error":"rejected_by_referee"})");
}

TEST(LlmAgentTest, ContextTooLargeIsRefusedBeforeSendingAnything) {
  auto provider = std::make_shared<ScriptedProvider>();
  AgentConfig config = DefaultConfig();
  config.context.max_input_tokens = 8;
  Agent agent(provider, config);
  agent.SetCommands({PressingTool()});

  ContextUpdate update = Update("rev-62");
  update.current_context =
      "A very long match description that cannot possibly fit eight tokens.";
  agent.UpdateContext(std::move(update));

  const DecisionResult decision = agent.DecideAsync().get();
  EXPECT_EQ(decision.status, DecisionStatus::ContextTooLarge);
  EXPECT_TRUE(provider->requests.empty())
      << "an over-budget request is never sent";
  EXPECT_FALSE(agent.IsBusy());
}

TEST(LlmAgentTest, DecisionContextTravelsFromDecideToHistory) {
  auto provider = std::make_shared<ScriptedProvider>();
  provider->next = SuccessfulBatch();
  Agent agent(provider, DefaultConfig());
  agent.SetCommands({PressingTool()});

  ContextUpdate update = Update("rev-62");
  update.decision_context = "Minute 62, 1-1, they press high.";
  agent.UpdateContext(std::move(update));
  const DecisionResult first = agent.DecideAsync().get();
  ASSERT_TRUE(first.completion.IsSuccess());
  EXPECT_EQ(agent.RecordCommandResults(first.decision_id,
                                       {ResultFor("call_a")}),
            RecordOutcome::Recorded);

  const Context context = agent.ContextSnapshot();
  ASSERT_EQ(context.decisions().size(), 1u);
  EXPECT_EQ(context.decisions()[0].decision_context,
            "Minute 62, 1-1, they press high.");

  // The next request shows the framing that decision answered.
  agent.UpdateContext(Update("rev-70"));
  const DecisionResult second = agent.DecideAsync().get();
  ASSERT_TRUE(second.completion.IsSuccess());
  ASSERT_GE(provider->requests.size(), 2u);
  const std::vector<Message>& messages = provider->requests[1].messages;
  ASSERT_GE(messages.size(), 3u);
  EXPECT_EQ(messages[1].role, Role::User);
  EXPECT_NE(messages[1].content.find("Minute 62, 1-1"), std::string::npos);
  EXPECT_EQ(messages[2].role, Role::Assistant);
}

}  // namespace
}  // namespace llm
