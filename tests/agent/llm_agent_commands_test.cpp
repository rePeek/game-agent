// Agent command-registration and configuration tests. The provider is scripted,
// so nothing here needs a network.

#include "agent_driver.h"

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>

#include <gtest/gtest.h>
#include <llm/agent.hpp>
#include <llm/core/provider.h>

namespace llm {
namespace {

CompletionResult SuccessfulBatch() {
  CompletionResult result;
  result.status = CompletionStatus::Success;
  result.command_calls = {
      CommandCall{"call_a", "set_pressing_intensity", R"({"level":4})"}};
  result.usage = Usage{10, 5, 15};
  result.http_attempts = 1;
  result.model = "scripted-model";
  return result;
}

class StubProvider : public IModelProvider {
 public:
  std::string Name() const override { return "stub"; }
  CompletionResult Complete(const ModelRequest&) override {
    return SuccessfulBatch();
  }
};

/// Blocks inside Complete() until released, so Busy rules can be exercised
/// without depending on timing.
class GatedProvider : public IModelProvider {
 public:
  std::string Name() const override { return "gated"; }
  CompletionResult Complete(const ModelRequest&) override {
    std::unique_lock<std::mutex> lock(mutex_);
    ++calls_;
    entered_cv_.notify_all();
    release_cv_.wait(lock, [this] { return released_; });
    return SuccessfulBatch();
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
  std::mutex mutex_;
  std::condition_variable entered_cv_;
  std::condition_variable release_cv_;
  int calls_ = 0;
  bool released_ = false;
};

AgentConfig Config() {
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

void RegisterPressing(Agent& agent) {
  (void)agent.RegisterCommand("set_pressing_intensity", "Pressing",
                              Params(Param<int>("level").Required()),
                              [](int) { return CommandOutcome{true, {}, {}}; });
}

TEST(LlmAgentCommandsTest, NotConfiguredIsDistinctFromClosed) {
  Agent agent(Config());

  // No provider yet: the refusal names configuration, not closure.
  const StartOutcome undecided = agent.DecideAsync();
  EXPECT_EQ(undecided.status, StartStatus::NotConfigured);
  EXPECT_FALSE(agent.IsClosed());

  ASSERT_TRUE(agent.SetProvider(std::make_shared<StubProvider>()).ok());
  RegisterPressing(agent);
  agent.UpdateContext(Update("rev-1"));
  EXPECT_TRUE(test::DecideOrThrow(agent).completion.IsSuccess());
}

TEST(LlmAgentCommandsTest, RejectsNullProviderAndEmptyModel) {
  Agent agent(Config());
  EXPECT_EQ(agent.SetProvider(nullptr).status, ConfigStatus::InvalidProvider);
  EXPECT_EQ(agent.SetModel("").status, ConfigStatus::InvalidModel);
  EXPECT_TRUE(agent.SetModel("gpt-5.6").ok());
  EXPECT_EQ(agent.model(), "gpt-5.6");
}

TEST(LlmAgentCommandsTest, RegistersTypedCommandsAndRejectsDuplicates) {
  Agent agent(std::make_shared<StubProvider>(), Config());

  int seen = 0;
  const CommandRegistrationResult registered = agent.RegisterCommand(
      "set_pressing_intensity", "Change pressing intensity",
      Params(Param<int>("level").Range(1, 5).Required()),
      [&seen](int level) -> CommandOutcome {
        seen = level;
        return CommandOutcome{true, "{}", {}};
      });
  ASSERT_TRUE(registered.ok()) << registered.error;
  ASSERT_EQ(agent.Commands().size(), 1u);
  EXPECT_EQ(agent.Commands()[0].name, "set_pressing_intensity");
  EXPECT_FALSE(agent.Commands()[0].parameters_schema_json.empty());

  const CommandRegistrationResult duplicate = agent.RegisterCommand(
      "set_pressing_intensity", "again", Params(), []() {});
  EXPECT_EQ(duplicate.status, RegistrationStatus::DuplicateName);

  EXPECT_EQ(agent.UnregisterCommand("set_pressing_intensity").status,
            RegistrationStatus::Registered);
  EXPECT_EQ(agent.UnregisterCommand("set_pressing_intensity").status,
            RegistrationStatus::NotRegistered);
  EXPECT_TRUE(agent.Commands().empty());
}

TEST(LlmAgentCommandsTest, CommandSetIsFrozenWhileADecisionIsOutstanding) {
  auto provider = std::make_shared<GatedProvider>();
  Agent agent(provider, Config());
  RegisterPressing(agent);

  agent.UpdateContext(Update("rev-1"));
  ASSERT_TRUE(agent.DecideAsync().ok());
  ASSERT_TRUE(provider->WaitUntilEntered(std::chrono::seconds(5)));

  // In flight: registering, removing and clearing are all refused.
  EXPECT_EQ(agent.RegisterCommand("late", "", Params(), []() {}).status,
            RegistrationStatus::AgentBusy);
  EXPECT_EQ(agent.UnregisterCommand("set_pressing_intensity").status,
            RegistrationStatus::AgentBusy);
  EXPECT_EQ(agent.ClearCommands().status, RegistrationStatus::AgentBusy);

  provider->Release();
  const DecisionResult decision = test::WaitDecision(agent);
  ASSERT_TRUE(decision.completion.IsSuccess());

  // Awaiting results is Busy too.
  EXPECT_EQ(agent.RegisterCommand("late", "", Params(), []() {}).status,
            RegistrationStatus::AgentBusy);

  ASSERT_TRUE(agent.DiscardDecision(decision.decision_id));
  EXPECT_TRUE(agent.RegisterCommand("late", "", Params(), []() {}).ok());
}

TEST(LlmAgentCommandsTest, ClosedAgentsRefuseConfigurationAndRegistration) {
  Agent agent(std::make_shared<StubProvider>(), Config());
  agent.Close();

  EXPECT_EQ(agent.SetProvider(std::make_shared<StubProvider>()).status,
            ConfigStatus::AgentClosed);
  EXPECT_EQ(agent.RegisterCommand("a", "", Params(), []() {}).status,
            RegistrationStatus::AgentClosed);
  EXPECT_EQ(agent.UnregisterCommand("a").status,
            RegistrationStatus::AgentClosed);
  EXPECT_EQ(agent.ClearCommands().status, RegistrationStatus::AgentClosed);
  EXPECT_EQ(agent.DecideAsync().status, StartStatus::Closed);
}

TEST(LlmAgentCommandsTest, ProviderCannotChangeWhileADecisionIsOutstanding) {
  auto provider = std::make_shared<GatedProvider>();
  Agent agent(provider, Config());
  RegisterPressing(agent);

  agent.UpdateContext(Update("rev-1"));
  ASSERT_TRUE(agent.DecideAsync().ok());
  ASSERT_TRUE(provider->WaitUntilEntered(std::chrono::seconds(5)));
  EXPECT_EQ(agent.SetProvider(std::make_shared<StubProvider>()).status,
            ConfigStatus::AgentBusy)
      << "in flight";

  provider->Release();
  const DecisionResult decision = test::WaitDecision(agent);
  ASSERT_TRUE(decision.completion.IsSuccess());
  EXPECT_EQ(agent.SetProvider(std::make_shared<StubProvider>()).status,
            ConfigStatus::AgentBusy)
      << "awaiting results";

  EXPECT_EQ(agent.PendingDecisionCount(), 1u);
  EXPECT_TRUE(agent.DiscardDecision(decision.decision_id));
  EXPECT_TRUE(agent.SetProvider(std::make_shared<StubProvider>()).ok());
}

TEST(LlmAgentCommandsTest, EmptyRegistryRefusesADecision) {
  Agent agent(std::make_shared<StubProvider>(), Config());
  agent.UpdateContext(Update("rev-1"));

  // The SDK forces a command call, so there is no valid request to send.
  EXPECT_EQ(agent.DecideAsync().status, StartStatus::NoCommands);
  EXPECT_TRUE(
      agent
          .RegisterCommand("set_pressing_intensity", "",
                           Params(Param<int>("level").Required()),
                           [](int) { return CommandOutcome{true, {}, {}}; })
          .ok());
  EXPECT_TRUE(test::DecideOrThrow(agent).completion.IsSuccess());
}

TEST(LlmAgentCommandsTest, InvalidContextBudgetConfigurationIsRefused) {
  AgentConfig config = Config();
  config.context.max_input_tokens = 100;
  config.context.reserved_output_tokens = 100;  // no room left for input
  Agent agent(std::make_shared<StubProvider>(), config);
  RegisterPressing(agent);
  agent.UpdateContext(Update("rev-1"));

  EXPECT_EQ(agent.DecideAsync().status, StartStatus::InvalidConfig);
}

TEST(LlmAgentCommandsTest, MalformedResultJsonIsRejected) {
  Agent agent(std::make_shared<StubProvider>(), Config());
  RegisterPressing(agent);
  agent.UpdateContext(Update("rev-1"));
  const DecisionResult decision = test::DecideOrThrow(agent);
  ASSERT_TRUE(decision.completion.IsSuccess());

  CommandResult bad;
  bad.command_call_id = "call_a";
  bad.name = "set_pressing_intensity";
  bad.outcome.success = true;
  bad.outcome.content_json = "not json";
  EXPECT_EQ(agent.RecordCommandResults(decision.decision_id, {bad}),
            RecordOutcome::InvalidResultJson);
  EXPECT_EQ(agent.PendingDecisionCount(), 1u) << "nothing was recorded";
  EXPECT_TRUE(agent.DiscardDecision(decision.decision_id));
}

}  // namespace
}  // namespace llm
