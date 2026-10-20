// Optional smoke test against a real OpenAI-compatible server (for example a
// local llama-server). It is skipped unless both environment variables are set:
//
//   GAME_AGENT_LLAMA_BASE_URL=http://127.0.0.1:8080
//   GAME_AGENT_LLAMA_MODEL=<model-id-advertised-by-the-server>
//   GAME_AGENT_LLAMA_API_KEY=<optional, defaults to a placeholder>
//
//     cmake --build build --target llama_smoke_tests
//     ctest -R llama_smoke --output-on-failure
//
// Everything else in the suite stays offline; this is the one place a real
// model is involved.

#include "agent_driver.h"

#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <llm/sdk.h>

namespace llm {
namespace {

std::string Env(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr ? std::string(value) : std::string();
}

struct Server {
  std::string base_url = Env("GAME_AGENT_LLAMA_BASE_URL");
  std::string model = Env("GAME_AGENT_LLAMA_MODEL");

  bool configured() const { return !base_url.empty() && !model.empty(); }
};

std::shared_ptr<IModelProvider> MakeProvider(const Server& server) {
  llm::openai::ProviderOptions options;
  options.api_key = Env("GAME_AGENT_LLAMA_API_KEY");
  if (options.api_key.empty()) {
    options.api_key = "sk-local";
  }
  options.base_url = server.base_url;
  options.timeout_seconds = 60;
  // A local server may ignore strict schemas and forced choice; say so, and let
  // SDK-side validation be the guarantee.
  options.capabilities.strict_command_schema = false;
  options.allow_validation_downgrade = true;
  return llm::openai::CreateProvider(std::move(options));
}

void RegisterCommands(Agent& agent) {
  (void)agent.RegisterCommand(
      "set_pressing_intensity",
      "Raise or lower how aggressively the team presses",
      Params(Param<int>("level").Range(1, 5).Required(),
             Param<std::string>("reason_code")
                 .OneOf("chasing", "stable", "protect")
                 .Required()),
      [](int, const std::string&) -> CommandOutcome {
        return CommandOutcome{true, R"({"applied":true})", {}};
      });
  (void)agent.RegisterCommand(
      "set_defensive_line", "Move the defensive line up or down the pitch",
      Params(Param<std::string>("height")
                 .OneOf("high", "medium", "low")
                 .Required()),
      [](const std::string&) -> CommandOutcome {
        return CommandOutcome{true, R"({"applied":true})", {}};
      });
}

AgentConfig Config(const Server& server, const std::string& id) {
  AgentConfig config;
  config.id = id;
  config.model = server.model;
  config.system_prompt =
      "You are the head coach. Answer only with command calls that change the "
      "tactical setup.";
  config.max_command_calls = 3;
  return config;
}

ContextUpdate Update(const std::string& revision) {
  ContextUpdate update;
  update.current_context =
      "minute=62; score=1-1; opponent presses high down our left";
  update.decision_context = "minute 62, score 1-1";
  update.context_revision = revision;
  return update;
}

/// The SDK contract on a real server: a decision either becomes a fully
/// validated, bound batch or is reported as a failure. It never hangs, never
/// stages an invalid batch, and finishing it always leaves the agent idle.
DecisionResult RunOneDecision(Agent& agent, const std::string& revision) {
  agent.UpdateContext(Update(revision));
  const StartOutcome started = agent.DecideAsync();
  if (!started.ok()) {
    DecisionResult result;
    result.context_revision = started.context_revision;
    return result;
  }
  const std::optional<DecisionResult> decision =
      test::PollUntilReady(agent, std::chrono::seconds(120));
  EXPECT_TRUE(decision.has_value()) << "the model call never completed";
  return decision.value_or(DecisionResult{});
}

/// Ends a decision: record a successful batch, discard anything else. Returns
/// true when a decision entered the transcript.
bool FinishOneDecision(Agent& agent, const DecisionResult& decision) {
  if (!decision.completion.IsSuccess()) {
    // A refusal or a server-side error is legitimate; it must be reported and
    // staged nowhere.
    EXPECT_EQ(agent.PendingDecisionCount(), 0u);
    return false;
  }
  PrepareResult prepared = agent.PrepareCommands(decision.decision_id);
  if (!prepared.ok()) {
    // The SDK refused to bind the batch; it must not be left pending.
    EXPECT_TRUE(agent.DiscardDecision(decision.decision_id));
    return false;
  }
  std::vector<CommandResult> results;
  results.reserve(prepared.commands.size());
  for (PreparedCommand& command : prepared.commands) {
    results.push_back(std::move(command).Invoke());
  }
  EXPECT_EQ(
      agent.RecordCommandResults(decision.decision_id, std::move(results)),
      RecordOutcome::Recorded);
  return true;
}

TEST(LlamaSmokeTest, DecisionIsEitherAValidatedBatchOrAReportedFailure) {
  const Server server;
  if (!server.configured()) {
    GTEST_SKIP() << "set GAME_AGENT_LLAMA_BASE_URL and GAME_AGENT_LLAMA_MODEL";
  }

  Agent agent(MakeProvider(server), Config(server, "smoke"));
  RegisterCommands(agent);

  const DecisionResult decision = RunOneDecision(agent, "rev-1");
  // The model call finished either way; a successful batch waits as pending
  // until the game decides what to do with it.
  EXPECT_EQ(agent.InFlightDecisionCount(), 0u);

  const bool recorded = FinishOneDecision(agent, decision);
  EXPECT_FALSE(agent.IsBusy());
  EXPECT_EQ(agent.ContextSnapshot().decision_count(), recorded ? 1u : 0u);
}

TEST(LlamaSmokeTest, TwoAgentsShareTheProviderButNotTheirHistory) {
  const Server server;
  if (!server.configured()) {
    GTEST_SKIP() << "set GAME_AGENT_LLAMA_BASE_URL and GAME_AGENT_LLAMA_MODEL";
  }

  auto provider = MakeProvider(server);
  Agent home(provider, Config(server, "home"));
  Agent away(provider, Config(server, "away"));
  RegisterCommands(home);
  RegisterCommands(away);

  FinishOneDecision(home, RunOneDecision(home, "home-1"));
  FinishOneDecision(away, RunOneDecision(away, "away-1"));

  // Both are idle and their histories are separate objects, each holding only
  // its own rounds.
  EXPECT_FALSE(home.IsBusy());
  EXPECT_FALSE(away.IsBusy());
  EXPECT_LE(home.ContextSnapshot().decision_count(), 1u);
  EXPECT_LE(away.ContextSnapshot().decision_count(), 1u);
}

TEST(LlamaSmokeTest, RepeatedDecisionsKeepTheHistoryBounded) {
  const Server server;
  if (!server.configured()) {
    GTEST_SKIP() << "set GAME_AGENT_LLAMA_BASE_URL and GAME_AGENT_LLAMA_MODEL";
  }

  AgentConfig config = Config(server, "long-run");
  config.context.max_decisions = 4;
  config.context.max_memory_decisions = 8;
  Agent agent(MakeProvider(server), config);
  RegisterCommands(agent);

  for (int round = 1; round <= 6; ++round) {
    FinishOneDecision(agent,
                      RunOneDecision(agent, "rev-" + std::to_string(round)));
  }

  EXPECT_FALSE(agent.IsBusy());
  EXPECT_LE(agent.ContextSnapshot().decision_count(), 8u);
}

}  // namespace
}  // namespace llm
