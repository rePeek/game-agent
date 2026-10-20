// PrepareCommands tests: binding, batch atomicity, one-shot invocation and
// lifecycle invalidation. The provider is scripted, so nothing here needs a
// network.

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <llm/agent.hpp>
#include <llm/core/provider.h>

namespace llm {
namespace {

CompletionResult Batch(std::vector<CommandCall> calls) {
  CompletionResult result;
  result.status = CompletionStatus::Success;
  result.command_calls = std::move(calls);
  result.usage = Usage{10, 5, 15};
  result.http_attempts = 1;
  result.model = "scripted-model";
  return result;
}

class ScriptedProvider : public IModelProvider {
 public:
  CompletionResult next;
  std::string Name() const override { return "scripted"; }
  CompletionResult Complete(const ModelRequest&) override { return next; }
};

AgentConfig Config() {
  AgentConfig config;
  config.id = "home";
  config.model = "scripted-model";
  return config;
}

ContextUpdate Update(const std::string& revision) {
  ContextUpdate update;
  update.current_context = "Minute 62.";
  update.context_revision = revision;
  return update;
}

struct Counters {
  int pressing = 0;
  int formation = 0;
};

void RegisterPressing(Agent& agent, Counters& counters) {
  ASSERT_TRUE(
      agent
          .RegisterCommand(
              "set_pressing", "Pressing",
              Params(Param<int>("level").Range(1, 5).Required()),
              [&counters](int level) -> CommandOutcome {
                ++counters.pressing;
                return CommandOutcome{
                    true, "{\"level\":" + std::to_string(level) + "}", {}};
              })
          .ok());
}

void RegisterFormation(Agent& agent, Counters& counters) {
  ASSERT_TRUE(
      agent
          .RegisterCommand(
              "set_formation", "Formation",
              Params(Param<std::string>("formation").Required()),
              [&counters](const std::string& formation) -> CommandOutcome {
                ++counters.formation;
                return CommandOutcome{true, "\"" + formation + "\"", {}};
              })
          .ok());
}

DecisionResult Decide(Agent& agent, const std::string& revision) {
  agent.UpdateContext(Update(revision));
  return agent.DecideAsync().get();
}

TEST(LlmAgentPrepareTest, BindsEveryCallWithoutRunningIt) {
  auto provider = std::make_shared<ScriptedProvider>();
  provider->next = Batch(
      {CommandCall{"call_a", "set_pressing", R"({"level":4})"},
       CommandCall{"call_b", "set_formation", R"({"formation":"4-4-2"})"}});
  Agent agent(provider, Config());
  Counters counters;
  RegisterPressing(agent, counters);
  RegisterFormation(agent, counters);

  const DecisionResult decision = Decide(agent, "rev-1");
  ASSERT_TRUE(decision.completion.IsSuccess());

  PrepareResult prepared = agent.PrepareCommands(decision.decision_id);
  ASSERT_TRUE(prepared.ok()) << prepared.error;
  ASSERT_EQ(prepared.commands.size(), 2u);
  EXPECT_EQ(prepared.commands[0].Id(), "call_a");
  EXPECT_EQ(prepared.commands[0].Name(), "set_pressing");
  EXPECT_EQ(prepared.commands[1].Id(), "call_b");
  EXPECT_EQ(prepared.commands[1].Name(), "set_formation");
  EXPECT_EQ(counters.pressing, 0) << "preparing must have no side effects";
  EXPECT_EQ(counters.formation, 0);

  std::vector<CommandResult> results;
  results.reserve(prepared.commands.size());
  for (PreparedCommand& command : prepared.commands) {
    results.push_back(std::move(command).Invoke());
  }
  EXPECT_EQ(counters.pressing, 1);
  EXPECT_EQ(counters.formation, 1);
  EXPECT_EQ(results[0].command_call_id, "call_a");
  EXPECT_TRUE(results[0].outcome.success);
  EXPECT_EQ(results[0].outcome.content_json, R"({"level":4})");

  EXPECT_EQ(
      agent.RecordCommandResults(decision.decision_id, std::move(results)),
      RecordOutcome::Recorded);
}

TEST(LlmAgentPrepareTest, InvokingIsOneShot) {
  auto provider = std::make_shared<ScriptedProvider>();
  provider->next =
      Batch({CommandCall{"call_a", "set_pressing", R"({"level":4})"}});
  Agent agent(provider, Config());
  Counters counters;
  RegisterPressing(agent, counters);

  const DecisionResult decision = Decide(agent, "rev-1");
  PrepareResult prepared = agent.PrepareCommands(decision.decision_id);
  ASSERT_TRUE(prepared.ok()) << prepared.error;

  PreparedCommand command = std::move(prepared.commands[0]);
  const CommandResult first = std::move(command).Invoke();
  EXPECT_TRUE(first.outcome.success);
  EXPECT_EQ(counters.pressing, 1);

  const CommandResult second = std::move(command).Invoke();
  EXPECT_FALSE(second.outcome.success);
  EXPECT_NE(second.outcome.error_message.find("already invoked"),
            std::string::npos);
  EXPECT_EQ(counters.pressing, 1) << "the callback ran at most once";

  agent.DiscardDecision(decision.decision_id);
}

TEST(LlmAgentPrepareTest, ABatchWithOneUnbindableCallPreparesNone) {
  auto provider = std::make_shared<ScriptedProvider>();
  // set_formation is not registered.
  provider->next = Batch(
      {CommandCall{"call_a", "set_pressing", R"({"level":4})"},
       CommandCall{"call_b", "set_formation", R"({"formation":"4-4-2"})"}});
  Agent agent(provider, Config());
  Counters counters;
  RegisterPressing(agent, counters);

  const DecisionResult decision = Decide(agent, "rev-1");
  const PrepareResult prepared = agent.PrepareCommands(decision.decision_id);

  EXPECT_FALSE(prepared.ok());
  EXPECT_EQ(prepared.status, PrepareStatus::InvalidArguments);
  EXPECT_TRUE(prepared.commands.empty()) << "all or nothing";
  EXPECT_EQ(counters.pressing, 0) << "nothing may run when the batch fails";
  EXPECT_NE(prepared.error.find("set_formation"), std::string::npos);
}

TEST(LlmAgentPrepareTest, InvalidArgumentsFailTheWholeBatch) {
  auto provider = std::make_shared<ScriptedProvider>();
  provider->next =
      Batch({CommandCall{"call_a", "set_pressing", R"({"level":4})"},
             CommandCall{"call_b", "set_pressing", R"({"level":"high"})"}});
  Agent agent(provider, Config());
  Counters counters;
  RegisterPressing(agent, counters);

  const DecisionResult decision = Decide(agent, "rev-1");
  const PrepareResult prepared = agent.PrepareCommands(decision.decision_id);

  EXPECT_EQ(prepared.status, PrepareStatus::InvalidArguments);
  EXPECT_TRUE(prepared.commands.empty());
  EXPECT_NE(prepared.error.find("must be an integer"), std::string::npos);
  EXPECT_EQ(counters.pressing, 0);
}

TEST(LlmAgentPrepareTest, DiscardingOrClosingInvalidatesPreparedCommands) {
  auto provider = std::make_shared<ScriptedProvider>();
  provider->next =
      Batch({CommandCall{"call_a", "set_pressing", R"({"level":4})"}});
  Agent agent(provider, Config());
  Counters counters;
  RegisterPressing(agent, counters);

  const DecisionResult decision = Decide(agent, "rev-1");
  PrepareResult prepared = agent.PrepareCommands(decision.decision_id);
  ASSERT_TRUE(prepared.ok()) << prepared.error;
  ASSERT_TRUE(agent.DiscardDecision(decision.decision_id));

  const CommandResult refused = std::move(prepared.commands[0]).Invoke();
  EXPECT_FALSE(refused.outcome.success);
  EXPECT_EQ(counters.pressing, 0)
      << "a discarded decision must not run its prepared commands";

  // Now with Close(): the decision is still pending when the agent closes.
  const DecisionResult second = Decide(agent, "rev-2");
  PrepareResult prepared_second = agent.PrepareCommands(second.decision_id);
  ASSERT_TRUE(prepared_second.ok()) << prepared_second.error;
  agent.Close();
  const CommandResult after_close =
      std::move(prepared_second.commands[0]).Invoke();
  EXPECT_FALSE(after_close.outcome.success);
  EXPECT_EQ(counters.pressing, 0);
}

TEST(LlmAgentPrepareTest, PrepareRefusesUnknownDecisionsAndClosedAgents) {
  auto provider = std::make_shared<ScriptedProvider>();
  provider->next =
      Batch({CommandCall{"call_a", "set_pressing", R"({"level":4})"}});
  Agent agent(provider, Config());
  Counters counters;
  RegisterPressing(agent, counters);

  EXPECT_EQ(agent.PrepareCommands(999).status, PrepareStatus::UnknownDecision);

  const DecisionResult decision = Decide(agent, "rev-1");
  agent.Close();
  EXPECT_EQ(agent.PrepareCommands(decision.decision_id).status,
            PrepareStatus::Closed);
}

TEST(LlmAgentPrepareTest, AThrowingCallbackIsReportedAsAFailedOutcome) {
  auto provider = std::make_shared<ScriptedProvider>();
  provider->next = Batch({CommandCall{"call_a", "explode", R"({"level":1})"}});
  Agent agent(provider, Config());
  ASSERT_TRUE(
      agent
          .RegisterCommand(
              "explode", "", Params(Param<int>("level").Required()),
              [](int) -> CommandOutcome { throw std::runtime_error("boom"); })
          .ok());

  const DecisionResult decision = Decide(agent, "rev-1");
  PrepareResult prepared = agent.PrepareCommands(decision.decision_id);
  ASSERT_TRUE(prepared.ok()) << prepared.error;

  const CommandResult result = std::move(prepared.commands[0]).Invoke();
  EXPECT_FALSE(result.outcome.success);
  EXPECT_NE(result.outcome.error_message.find("boom"), std::string::npos);
  agent.DiscardDecision(decision.decision_id);
}

TEST(LlmAgentPrepareTest, ACommandCallExecutesAtMostOnceAcrossBatches) {
  auto provider = std::make_shared<ScriptedProvider>();
  provider->next =
      Batch({CommandCall{"call_a", "set_pressing", R"({"level":4})"}});
  Agent agent(provider, Config());
  Counters counters;
  RegisterPressing(agent, counters);

  const DecisionResult decision = Decide(agent, "rev-1");
  PrepareResult batch_a = agent.PrepareCommands(decision.decision_id);
  PrepareResult batch_b = agent.PrepareCommands(decision.decision_id);
  ASSERT_TRUE(batch_a.ok()) << batch_a.error;
  ASSERT_TRUE(batch_b.ok()) << batch_b.error;

  const CommandResult first = std::move(batch_a.commands[0]).Invoke();
  EXPECT_TRUE(first.outcome.success);
  EXPECT_EQ(counters.pressing, 1);

  // A second batch prepared from the same model call must not run it again.
  const CommandResult second = std::move(batch_b.commands[0]).Invoke();
  EXPECT_FALSE(second.outcome.success);
  EXPECT_NE(second.outcome.error_message.find("already executed"),
            std::string::npos);
  EXPECT_EQ(counters.pressing, 1)
      << "a non-idempotent command must execute at most once";

  agent.DiscardDecision(decision.decision_id);
}

}  // namespace
}  // namespace llm
