// Cross-component tests: Agent + Context + the OpenAI provider driven through
// a scripted transport. No network, no API key — but the full stack: the
// request that would go on the wire is built and asserted at every round.

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <llm/agent.hpp>
#include <llm/context/context.h>
#include <llm/providers/openai/openai.h>
#include <nlohmann/json.hpp>

namespace llm {
namespace {

/// The argument text the scripted server sends: preserved byte for byte.
constexpr char kWireArguments[] = R"({"level": 4})";

void RegisterCommands(Agent& agent) {
  (void)agent.RegisterCommand(
      "set_pressing_intensity", "Pressing intensity",
      Params(Param<int>("level").Range(1, 5).Required()),
      [](int) { return CommandOutcome{true, {}, {}}; });
  (void)agent.RegisterCommand(
      "set_defensive_line", "Defensive line height",
      Params(Param<std::string>("height")
                 .OneOf("high", "medium", "low")
                 .Required()),
      [](const std::string&) { return CommandOutcome{true, {}, {}}; });
}

std::string WireResponse(const std::string& tool_calls_json) {
  const nlohmann::json body{
      {"id", "chatcmpl-test"},
      {"model", "gpt-5.6"},
      {"choices",
       nlohmann::json::array({nlohmann::json{
           {"index", 0},
           {"finish_reason", "tool_calls"},
           {"message",
            {{"role", "assistant"},
             {"content", nullptr},
             {"tool_calls", nlohmann::json::parse(tool_calls_json)}}}}})},
      {"usage",
       {{"prompt_tokens", 20},
        {"completion_tokens", 8},
        {"total_tokens", 28}}}};
  return body.dump();
}

std::string TwoCallResponse() {
  return WireResponse(
      R"([{"id":"call_a","type":"function","function":{"name":"set_pressing_intensity","arguments":"{\"level\": 4}"}},)"
      R"({"id":"call_b","type":"function","function":{"name":"set_defensive_line","arguments":"{\"height\":\"high\"}"}}])");
}

std::string OneCallResponse() {
  return WireResponse(
      R"([{"id":"call_c","type":"function","function":{"name":"set_pressing_intensity","arguments":"{\"level\":5}"}}])");
}

/// Records the request bodies it was given and answers with a script.
class ScriptedTransport : public ITransport {
 public:
  std::vector<std::string> bodies;
  std::vector<std::string> responses;

  HttpResponse Send(const HttpRequest& request) override {
    bodies.push_back(request.body);
    HttpResponse response;
    response.transport_ok = true;
    response.status = 200;
    response.body = responses.size() > 1 ? responses.front() : responses.at(0);
    if (responses.size() > 1) {
      responses.erase(responses.begin());
    }
    return response;
  }

  nlohmann::json Body(std::size_t index) const {
    return nlohmann::json::parse(bodies.at(index), nullptr, false);
  }
};

ContextUpdate Update(std::string context, std::string revision) {
  ContextUpdate update;
  update.current_context = std::move(context);
  update.context_revision = std::move(revision);
  return update;
}

CommandResult ResultFor(const std::string& call_id, const std::string& name) {
  CommandResult result;
  result.command_call_id = call_id;
  result.name = name;
  result.outcome.success = true;
  result.outcome.content_json = R"({"applied":true})";
  return result;
}

struct Fixture {
  std::shared_ptr<ScriptedTransport> transport =
      std::make_shared<ScriptedTransport>();
  std::shared_ptr<Agent> agent;

  explicit Fixture(std::size_t max_messages = 32) {
    llm::openai::ProviderOptions options;
    options.api_key = "scripted";
    options.transport = transport;

    AgentConfig config;
    config.id = "home";
    config.model = llm::openai::models::kGpt56;
    config.system_prompt = "You are the head coach.";
    config.context.max_messages = max_messages;
    config.max_output_tokens = 300;
    config.max_command_calls = 3;
    agent = std::make_shared<Agent>(
        llm::openai::CreateProvider(std::move(options)), std::move(config));
    RegisterCommands(*agent);
  }

  /// One decision round: update the context, then ask.
  DecisionResult Decide(const std::string& revision) {
    agent->UpdateContext(Update("Minute window.", revision));
    return agent->DecideAsync().get();
  }
};

TEST(LlmAgentOpenAITest, AsyncDecisionRoundTripsThroughTheWire) {
  Fixture fixture;
  fixture.transport->responses.push_back(TwoCallResponse());

  fixture.agent->UpdateContext(
      Update("Minute 62, 1-1, they press high.", "rev-62"));
  const DecisionResult decision = fixture.agent->DecideAsync().get();

  ASSERT_EQ(decision.status, DecisionStatus::Completed);
  ASSERT_TRUE(decision.completion.IsSuccess())
      << decision.completion.ErrorMessage();
  EXPECT_EQ(decision.context_revision, "rev-62");
  ASSERT_EQ(decision.completion.command_calls.size(), 2u);
  EXPECT_EQ(decision.completion.command_calls[0].name,
            "set_pressing_intensity");
  EXPECT_EQ(decision.completion.command_calls[0].arguments_json, kWireArguments)
      << "the argument text the model produced is preserved verbatim";
  EXPECT_EQ(decision.completion.usage.total_tokens, 28);
  EXPECT_EQ(decision.completion.http_attempts, 1);

  // What went on the wire for the first decision.
  const nlohmann::json body = fixture.transport->Body(0);
  EXPECT_EQ(body["model"], "gpt-5.6");
  EXPECT_EQ(body["tool_choice"], "required");
  EXPECT_EQ(body["max_completion_tokens"], 300)
      << "gpt-5.6 takes max_completion_tokens, and the agent bound it";
  EXPECT_FALSE(body.contains("parallel_tool_calls"));
  ASSERT_EQ(body["tools"].size(), 2u);
  ASSERT_EQ(body["messages"].size(), 2u);
  EXPECT_EQ(body["messages"][0]["role"], "system");
  EXPECT_EQ(body["messages"][1]["role"], "user");
  EXPECT_NE(body["messages"][1]["content"].get<std::string>().find("Minute 62"),
            std::string::npos);
}

TEST(LlmAgentOpenAITest, CommandResultsFeedTheNextRequest) {
  Fixture fixture;
  fixture.transport->responses.push_back(TwoCallResponse());

  const DecisionResult first = fixture.Decide("rev-62");
  ASSERT_TRUE(first.completion.IsSuccess()) << first.completion.ErrorMessage();

  // The game executes the batch and reports what it did.
  EXPECT_EQ(
      fixture.agent->RecordCommandResults(
          first.decision_id, {ResultFor("call_a", "set_pressing_intensity"),
                              ResultFor("call_b", "set_defensive_line")}),
      RecordOutcome::Recorded);

  fixture.transport->responses.push_back(OneCallResponse());
  fixture.agent->UpdateContext(
      Update("Minute 70, they switched to a low block.", "rev-70"));
  const DecisionResult second = fixture.agent->DecideAsync().get();
  ASSERT_TRUE(second.completion.IsSuccess())
      << second.completion.ErrorMessage();

  // The second request carries the confirmed decision: assistant tool calls
  // followed by one tool message per call, then the new state.
  const nlohmann::json body = fixture.transport->Body(1);
  ASSERT_EQ(body["messages"].size(), 5u);
  EXPECT_EQ(body["messages"][0]["role"], "system");
  EXPECT_EQ(body["messages"][1]["role"], "assistant");
  EXPECT_TRUE(body["messages"][1]["content"].is_null());
  ASSERT_EQ(body["messages"][1]["tool_calls"].size(), 2u);
  EXPECT_EQ(body["messages"][1]["tool_calls"][0]["id"], "call_a");
  EXPECT_EQ(body["messages"][1]["tool_calls"][0]["function"]["arguments"]
                .get<std::string>(),
            kWireArguments);
  EXPECT_EQ(body["messages"][2]["role"], "tool");
  EXPECT_EQ(body["messages"][2]["tool_call_id"], "call_a");
  EXPECT_EQ(body["messages"][2]["content"],
            R"({"ok":true,"result":{"applied":true}})");
  EXPECT_EQ(body["messages"][3]["role"], "tool");
  EXPECT_EQ(body["messages"][3]["tool_call_id"], "call_b");
  EXPECT_EQ(body["messages"][4]["role"], "user");
  EXPECT_NE(body["messages"][4]["content"].get<std::string>().find("Minute 70"),
            std::string::npos);
}

TEST(LlmAgentOpenAITest, HistoryIsTrimmedWithoutSplittingDecisions) {
  // Each decision here is three messages (assistant + two tool turns), so
  // eight leaves room for the system prompt, the current context and two of
  // them.
  Fixture fixture(/*max_messages=*/8);

  for (const char* revision : {"rev-1", "rev-2", "rev-3"}) {
    fixture.transport->responses.push_back(TwoCallResponse());
    fixture.agent->UpdateContext(Update("Minute window.", revision));
    const DecisionResult decision = fixture.agent->DecideAsync().get();
    ASSERT_TRUE(decision.completion.IsSuccess())
        << decision.completion.ErrorMessage();
    ASSERT_EQ(fixture.agent->RecordCommandResults(
                  decision.decision_id,
                  {ResultFor("call_a", "set_pressing_intensity"),
                   ResultFor("call_b", "set_defensive_line")}),
              RecordOutcome::Recorded);
  }
  EXPECT_EQ(fixture.agent->ContextSnapshot().decision_count(), 3u);

  fixture.transport->responses.push_back(OneCallResponse());
  const DecisionResult fourth = fixture.Decide("rev-4");
  ASSERT_TRUE(fourth.completion.IsSuccess())
      << fourth.completion.ErrorMessage();

  // The newest two decisions fit; the oldest was dropped as a whole, so no
  // orphan tool message is left behind.
  const nlohmann::json body = fixture.transport->Body(3);
  ASSERT_EQ(body["messages"].size(), 8u);
  EXPECT_EQ(body["messages"][0]["role"], "system");
  EXPECT_EQ(body["messages"][1]["role"], "assistant");
  EXPECT_EQ(body["messages"][2]["role"], "tool");
  EXPECT_EQ(body["messages"][3]["role"], "tool");
  EXPECT_EQ(body["messages"][4]["role"], "assistant");
  EXPECT_EQ(body["messages"][5]["role"], "tool");
  EXPECT_EQ(body["messages"][6]["role"], "tool");
  EXPECT_EQ(body["messages"][7]["role"], "user");
  // Every surviving decision is complete: no assistant turn lost its results.
  for (const char* index : {"1", "4"}) {
    EXPECT_EQ(body["messages"][std::stoi(index)]["tool_calls"].size(), 2u);
  }
  EXPECT_EQ(fixture.agent->ContextSnapshot().Render().decisions_dropped, 1u);
}

TEST(LlmAgentOpenAITest, StaleDecisionsAreDiscardedNotRecorded) {
  Fixture fixture;
  fixture.transport->responses.push_back(TwoCallResponse());
  fixture.transport->responses.push_back(OneCallResponse());

  // A decision taken against rev-62 comes back late.
  const DecisionResult stale = fixture.Decide("rev-62");
  ASSERT_TRUE(stale.completion.IsSuccess());

  // Meanwhile the game moved on; the caller recognises the stale revision and
  // refuses to apply it.
  fixture.agent->UpdateContext(Update("Minute 75.", "rev-75"));
  EXPECT_NE(stale.context_revision, fixture.agent->current_revision());
  EXPECT_TRUE(fixture.agent->DiscardDecision(stale.decision_id));

  const DecisionResult next = fixture.agent->DecideAsync().get();
  ASSERT_TRUE(next.completion.IsSuccess());

  // The discarded decision left no trace in the transcript.
  const nlohmann::json body = fixture.transport->Body(1);
  ASSERT_EQ(body["messages"].size(), 2u);
  EXPECT_EQ(body["messages"][0]["role"], "system");
  EXPECT_EQ(body["messages"][1]["role"], "user");
  EXPECT_EQ(fixture.agent->ContextSnapshot().decision_count(), 0u);
}

TEST(LlmAgentOpenAITest, InvalidBatchesNeverReachTheCallerAsSuccess) {
  Fixture fixture;
  // level is out of the schema's range.
  fixture.transport->responses.push_back(WireResponse(
      R"([{"id":"call_a","type":"function","function":{"name":"set_pressing_intensity","arguments":"{\"level\":9}"}}])"));

  const DecisionResult decision = fixture.Decide("rev-62");

  EXPECT_EQ(decision.status, DecisionStatus::Completed);
  EXPECT_EQ(decision.completion.status, CompletionStatus::InvalidCommandCalls);
  EXPECT_FALSE(decision.completion.IsSuccess());
  EXPECT_NE(decision.completion.ErrorMessage().find("level"),
            std::string::npos);

  // Nothing to confirm, and the agent is free for the next decision.
  EXPECT_EQ(decision.decision_id, 0u);
  EXPECT_EQ(fixture.agent->PendingDecisionCount(), 0u);
  EXPECT_FALSE(fixture.agent->IsBusy());
}

TEST(LlmAgentOpenAITest, ManyRoundsKeepIdAndRevisionBookkeepingStraight) {
  Fixture fixture;
  std::uint64_t expected_id = 1;
  for (int round = 1; round <= 4; ++round) {
    const std::string revision = "rev-" + std::to_string(round);
    fixture.transport->responses.push_back(TwoCallResponse());

    fixture.agent->UpdateContext(Update("Minute window.", revision));
    const DecisionResult decision = fixture.agent->DecideAsync().get();
    ASSERT_TRUE(decision.completion.IsSuccess())
        << decision.completion.ErrorMessage();
    EXPECT_EQ(decision.decision_id, expected_id++);
    EXPECT_EQ(decision.context_revision, revision);
    EXPECT_EQ(fixture.agent->RecordCommandResults(
                  decision.decision_id,
                  {ResultFor("call_a", "set_pressing_intensity"),
                   ResultFor("call_b", "set_defensive_line")}),
              RecordOutcome::Recorded);
  }

  const Context context = fixture.agent->ContextSnapshot();
  EXPECT_EQ(context.decision_count(), 4u);
  for (std::size_t i = 0; i < context.decisions().size(); ++i) {
    const DecisionRecord& record = context.decisions()[i];
    EXPECT_EQ(record.decision_id, i + 1);
    EXPECT_EQ(record.context_revision, "rev-" + std::to_string(i + 1));
    EXPECT_EQ(record.assistant.command_calls.size(), 2u);
    EXPECT_EQ(record.command_results.size(), 2u);
    EXPECT_EQ(record.command_results[0].command_call_id, "call_a");
  }
  EXPECT_EQ(fixture.transport->bodies.size(), 4u);
}

TEST(LlmAgentOpenAITest, RegisteredCommandsRoundTripThroughPrepareAndRecord) {
  Fixture fixture;
  fixture.transport->responses.push_back(TwoCallResponse());

  const DecisionResult first = fixture.Decide("rev-62");
  ASSERT_TRUE(first.completion.IsSuccess()) << first.completion.ErrorMessage();

  // The schema the model saw was generated from the registered parameters.
  const nlohmann::json first_body = fixture.transport->Body(0);
  ASSERT_EQ(first_body["tools"].size(), 2u);
  const nlohmann::json& level =
      first_body["tools"][0]["function"]["parameters"]["properties"]["level"];
  EXPECT_EQ(level["type"], "integer");
  EXPECT_EQ(level["minimum"], 1);
  EXPECT_EQ(level["maximum"], 5);

  llm::PrepareResult prepared =
      fixture.agent->PrepareCommands(first.decision_id);
  ASSERT_TRUE(prepared.ok()) << prepared.error;
  ASSERT_EQ(prepared.commands.size(), 2u);
  EXPECT_EQ(prepared.commands[0].Name(), "set_pressing_intensity");
  EXPECT_EQ(prepared.commands[1].Name(), "set_defensive_line");

  std::vector<CommandResult> results;
  results.reserve(prepared.commands.size());
  for (PreparedCommand& command : prepared.commands) {
    results.push_back(std::move(command).Invoke());
  }
  ASSERT_TRUE(results[0].outcome.success);
  EXPECT_EQ(fixture.agent->RecordCommandResults(first.decision_id,
                                                std::move(results)),
            RecordOutcome::Recorded);

  fixture.transport->responses.push_back(OneCallResponse());
  const DecisionResult second = fixture.Decide("rev-70");
  ASSERT_TRUE(second.completion.IsSuccess());

  // The confirmed decision, in the unified result shape, is on the wire.
  const nlohmann::json body = fixture.transport->Body(1);
  ASSERT_EQ(body["messages"].size(), 5u);
  EXPECT_EQ(body["messages"][2]["role"], "tool");
  EXPECT_EQ(body["messages"][2]["content"], R"({"ok":true})");
  EXPECT_EQ(body["messages"][3]["content"], R"({"ok":true})");
}

}  // namespace
}  // namespace llm
