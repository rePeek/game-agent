// Unit tests for the OpenAI-compatible provider: request building, response
// parsing and status mapping. Everything runs against a scripted transport, so
// no network and no API key are involved.

#include "providers/openai/request_builder.h"
#include "providers/openai/response_parser.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <llm/providers/openai/openai.h>
#include <nlohmann/json.hpp>

namespace llm {
namespace openai {
namespace {

constexpr char kRawFirstArguments[] = R"({ "level" : 3 })";
constexpr char kRawSecondArguments[] = R"({"height":"high"})";

// ---------------------------------------------------------------------------
// Scripted transport
// ---------------------------------------------------------------------------
class FakeTransport : public ITransport {
 public:
  std::vector<HttpResponse> responses;
  std::vector<HttpRequest> requests;

  HttpResponse Send(const HttpRequest& request) override {
    requests.push_back(request);
    if (responses.empty()) {
      return HttpResponse{false, 0, "", "no scripted response",
                          TransportError::Other};
    }
    HttpResponse response = responses.front();
    if (responses.size() > 1) {
      responses.erase(responses.begin());
    }
    return response;
  }

  static HttpResponse Ok(std::string body) {
    return HttpResponse{true, 200, std::move(body), "", TransportError::None};
  }
  static HttpResponse WithStatus(int status, std::string body) {
    return HttpResponse{true, status, std::move(body), "",
                        TransportError::None};
  }
  static HttpResponse Timeout() {
    return HttpResponse{false, 0, "", "read timeout", TransportError::Timeout};
  }
};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------
ToolDefinition PressingTool(bool strict = true) {
  ToolDefinition tool;
  tool.name = "set_pressing_intensity";
  tool.description = "Raise or lower the pressing intensity";
  tool.parameters_schema_json =
      R"({"type":"object","properties":{"level":{"type":"integer"}},"required":["level"],"additionalProperties":false})";
  tool.strict = strict;
  return tool;
}

ToolDefinition DefensiveLineTool(bool strict = true) {
  ToolDefinition tool;
  tool.name = "set_defensive_line";
  tool.description = "Move the defensive line";
  tool.parameters_schema_json =
      R"({"type":"object","properties":{"height":{"type":"string","enum":["high","medium","low"]}},"required":["height"],"additionalProperties":false})";
  tool.strict = strict;
  return tool;
}

std::vector<ToolCall> TwoToolCalls() {
  return {
      ToolCall{"call_001", "set_pressing_intensity", kRawFirstArguments},
      ToolCall{"call_002", "set_defensive_line", kRawSecondArguments},
  };
}

ModelRequest MakeRequest() {
  ModelRequest request;
  request.model = models::kGpt56;
  request.messages = {Message::System("You are a coach."),
                      Message::User("Opponent is pressing high.")};
  request.tools = {PressingTool(), DefensiveLineTool()};
  request.max_tokens = 200;
  return request;
}

std::string MakeResponse(const nlohmann::json& message,
                         const std::string& finish_reason = "tool_calls") {
  const nlohmann::json body{
      {"id", "chatcmpl-test"},
      {"object", "chat.completion"},
      {"model", "gpt-5.6-2026-01-01"},
      {"choices", nlohmann::json::array(
                      {nlohmann::json{{"index", 0},
                                      {"message", message},
                                      {"finish_reason", finish_reason}}})},
      {"usage",
       {{"prompt_tokens", 11},
        {"completion_tokens", 7},
        {"total_tokens", 18}}}};
  return body.dump();
}

nlohmann::json ToolCallsMessage(
    const std::vector<std::pair<std::string, std::string>>& calls) {
  nlohmann::json array = nlohmann::json::array();
  for (const auto& [id, name] : calls) {
    array.push_back({{"id", id},
                     {"type", "function"},
                     {"function", {{"name", name}, {"arguments", ""}}}});
  }
  nlohmann::json message{
      {"role", "assistant"}, {"content", nullptr}, {"tool_calls", array}};
  // Fill the raw argument text separately so the test controls it exactly.
  return message;
}

// Message with the two-call batch, arguments as sent by a conforming server.
nlohmann::json TwoCallsMessage() {
  nlohmann::json message =
      ToolCallsMessage({{"call_001", "set_pressing_intensity"},
                        {"call_002", "set_defensive_line"}});
  message["tool_calls"][0]["function"]["arguments"] = kRawFirstArguments;
  message["tool_calls"][1]["function"]["arguments"] = kRawSecondArguments;
  return message;
}

struct Fixture {
  std::shared_ptr<FakeTransport> transport = std::make_shared<FakeTransport>();

  ProviderOptions Options() const {
    ProviderOptions options;
    options.api_key = "sk-test";
    options.transport = transport;
    options.retry.initial_delay_ms = 0;  // never sleep in tests
    return options;
  }

  CompletionResult Complete(const ModelRequest& request,
                            ProviderOptions options) {
    return OpenAIProvider(std::move(options)).Complete(request);
  }
};

// ---------------------------------------------------------------------------
// Request building
// ---------------------------------------------------------------------------
TEST(LlmOpenAIBuilderTest, BuildsToolsAndRequiredToolChoice) {
  const ModelRequest request = MakeRequest();

  const detail::BuildOutcome built =
      detail::BuildChatCompletionRequest(request);

  ASSERT_TRUE(built.ok) << built.error;
  EXPECT_EQ(built.body["model"], "gpt-5.6");
  EXPECT_EQ(built.body["max_completion_tokens"], 200)
      << "reasoning-first models take max_completion_tokens";
  EXPECT_FALSE(built.body.contains("max_tokens"));
  EXPECT_EQ(built.body["tool_choice"], "required");
  EXPECT_FALSE(built.body.contains("parallel_tool_calls"))
      << "the SDK must not cap the number of tool calls";

  ASSERT_EQ(built.body["messages"].size(), 2u);
  EXPECT_EQ(built.body["messages"][0]["role"], "system");
  EXPECT_EQ(built.body["messages"][0]["content"], "You are a coach.");
  EXPECT_EQ(built.body["messages"][1]["role"], "user");

  ASSERT_EQ(built.body["tools"].size(), 2u);
  const auto& tool = built.body["tools"][0];
  EXPECT_EQ(tool["type"], "function");
  EXPECT_EQ(tool["function"]["name"], "set_pressing_intensity");
  EXPECT_EQ(built.body["tools"][1]["function"]["name"], "set_defensive_line");
  EXPECT_EQ(tool["function"]["description"],
            "Raise or lower the pressing intensity");
  EXPECT_TRUE(tool["function"]["strict"].get<bool>());
  EXPECT_TRUE(tool["function"]["parameters"]["properties"].contains("level"));
}

TEST(LlmOpenAIBuilderTest, ToolChoiceVariants) {
  ModelRequest request = MakeRequest();

  request.tool_choice = ToolChoice::Auto();
  EXPECT_EQ(detail::BuildChatCompletionRequest(request).body["tool_choice"],
            "auto");

  request.tool_choice = ToolChoice::None();
  EXPECT_EQ(detail::BuildChatCompletionRequest(request).body["tool_choice"],
            "none");

  request.tool_choice = ToolChoice::Specific("set_pressing_intensity");
  const nlohmann::json specific =
      detail::BuildChatCompletionRequest(request).body["tool_choice"];
  EXPECT_EQ(specific["type"], "function");
  EXPECT_EQ(specific["function"]["name"], "set_pressing_intensity");
}

TEST(LlmOpenAIBuilderTest, StrictIsOmittedWhenNotRequested) {
  ModelRequest request = MakeRequest();
  request.tools = {PressingTool(/*strict=*/false)};

  const detail::BuildOutcome built =
      detail::BuildChatCompletionRequest(request);

  ASSERT_TRUE(built.ok) << built.error;
  EXPECT_FALSE(built.body["tools"][0]["function"].contains("strict"));
}

// The round trip the SDK exists for: a previous decision (assistant tool calls
// plus their results) must serialize back into an OpenAI-compatible context.
TEST(LlmOpenAIBuilderTest, SerializesPreviousDecisionAsContext) {
  ToolResult applied;
  applied.tool_call_id = "call_001";
  applied.name = "set_pressing_intensity";
  applied.content_json = R"({"applied":true})";

  ToolResult applied_second;
  applied_second.tool_call_id = "call_002";
  applied_second.name = "set_defensive_line";
  applied_second.content_json = R"({"applied":true})";

  ModelRequest request;
  request.model = models::kGpt56;
  request.messages = {
      Message::System("You are a coach."),
      Message::User("Opponent is pressing high."),
      Message::AssistantToolCalls(TwoToolCalls()),
      Message::ToolResult(applied),
      Message::ToolResult(applied_second),
      Message::User("They switched to a low block."),
  };
  request.tools = {PressingTool()};

  const detail::BuildOutcome built =
      detail::BuildChatCompletionRequest(request);

  ASSERT_TRUE(built.ok) << built.error;
  ASSERT_EQ(built.body["messages"].size(), 6u);

  const auto& assistant = built.body["messages"][2];
  EXPECT_EQ(assistant["role"], "assistant");
  EXPECT_TRUE(assistant["content"].is_null())
      << "a tool-only assistant turn carries null content";
  ASSERT_EQ(assistant["tool_calls"].size(), 2u);
  EXPECT_EQ(assistant["tool_calls"][0]["id"], "call_001");
  EXPECT_EQ(assistant["tool_calls"][1]["id"], "call_002");
  EXPECT_EQ(assistant["tool_calls"][0]["type"], "function");
  EXPECT_EQ(assistant["tool_calls"][0]["function"]["name"],
            "set_pressing_intensity");
  // Arguments travel as the raw JSON *text*, byte for byte.
  ASSERT_TRUE(assistant["tool_calls"][0]["function"]["arguments"].is_string());
  EXPECT_EQ(
      assistant["tool_calls"][0]["function"]["arguments"].get<std::string>(),
      kRawFirstArguments);
  EXPECT_EQ(
      assistant["tool_calls"][1]["function"]["arguments"].get<std::string>(),
      kRawSecondArguments);

  const auto& tool_message = built.body["messages"][3];
  EXPECT_EQ(tool_message["role"], "tool");
  EXPECT_EQ(tool_message["tool_call_id"], "call_001");
  EXPECT_EQ(tool_message["content"], R"({"applied":true})");
  EXPECT_FALSE(tool_message.contains("name"))
      << "OpenAI tool messages carry no name";

  EXPECT_EQ(built.body["messages"][5]["role"], "user");
}

TEST(LlmOpenAIBuilderTest, SamplingParametersFollowTheModelFamily) {
  ModelRequest request = MakeRequest();
  request.temperature = 0.7;
  request.top_p = 0.9;

  const detail::BuildOutcome gpt5 = detail::BuildChatCompletionRequest(request);
  ASSERT_TRUE(gpt5.ok) << gpt5.error;
  EXPECT_FALSE(gpt5.body.contains("temperature"))
      << "the gpt-5 family rejects sampling controls on Chat Completions";
  EXPECT_FALSE(gpt5.body.contains("top_p"));

  request.model = models::kGpt41;
  const detail::BuildOutcome gpt41 =
      detail::BuildChatCompletionRequest(request);
  ASSERT_TRUE(gpt41.ok) << gpt41.error;
  EXPECT_DOUBLE_EQ(gpt41.body["temperature"].get<double>(), 0.7);
  EXPECT_DOUBLE_EQ(gpt41.body["top_p"].get<double>(), 0.9);
}

TEST(LlmOpenAIBuilderTest, RejectsInvalidRequests) {
  ModelRequest request = MakeRequest();
  request.model.clear();
  EXPECT_FALSE(detail::BuildChatCompletionRequest(request).ok);

  request = MakeRequest();
  request.messages.clear();
  EXPECT_FALSE(detail::BuildChatCompletionRequest(request).ok);

  request = MakeRequest();
  request.tools = {PressingTool()};
  request.tools[0].parameters_schema_json = "{not json";
  const detail::BuildOutcome broken =
      detail::BuildChatCompletionRequest(request);
  EXPECT_FALSE(broken.ok);
  EXPECT_NE(broken.error.find("not valid JSON"), std::string::npos);

  request = MakeRequest();
  request.tools[0].name.clear();
  EXPECT_FALSE(detail::BuildChatCompletionRequest(request).ok);

  request = MakeRequest();
  request.tool_choice = ToolChoice::Specific("");
  EXPECT_FALSE(detail::BuildChatCompletionRequest(request).ok);
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------
TEST(LlmOpenAIParserTest, KeepsEveryToolCallInOrderWithRawArguments) {
  const detail::ParsedCompletion parsed =
      detail::ParseChatCompletion(MakeResponse(TwoCallsMessage()));

  ASSERT_TRUE(parsed.ok) << parsed.error;
  EXPECT_EQ(parsed.model, "gpt-5.6-2026-01-01");
  EXPECT_EQ(parsed.finish_reason, "tool_calls");
  EXPECT_EQ(parsed.usage.prompt_tokens, 11);
  EXPECT_EQ(parsed.usage.completion_tokens, 7);
  EXPECT_EQ(parsed.usage.total_tokens, 18);
  ASSERT_EQ(parsed.tool_calls.size(), 2u);
  EXPECT_EQ(parsed.tool_calls[0].id, "call_001");
  EXPECT_EQ(parsed.tool_calls[0].name, "set_pressing_intensity");
  EXPECT_EQ(parsed.tool_calls[0].arguments_json, kRawFirstArguments);
  EXPECT_EQ(parsed.tool_calls[1].id, "call_002");
  EXPECT_EQ(parsed.tool_calls[1].arguments_json, kRawSecondArguments);
}

TEST(LlmOpenAIParserTest, AcceptsObjectArgumentsFromCompatibleServers) {
  nlohmann::json message = TwoCallsMessage();
  message["tool_calls"][0]["function"]["arguments"] =
      nlohmann::json{{"level", 3}};

  const detail::ParsedCompletion parsed =
      detail::ParseChatCompletion(MakeResponse(message));

  ASSERT_TRUE(parsed.ok) << parsed.error;
  ASSERT_EQ(parsed.tool_calls.size(), 2u);
  EXPECT_EQ(parsed.tool_calls[0].arguments_json, R"({"level":3})");
}

TEST(LlmOpenAIParserTest, ReadsTextAndRefusal) {
  const detail::ParsedCompletion text = detail::ParseChatCompletion(
      MakeResponse({{"role", "assistant"}, {"content", "2 + 2 is 4"}}, "stop"));
  ASSERT_TRUE(text.ok) << text.error;
  EXPECT_EQ(text.assistant_text, "2 + 2 is 4");
  EXPECT_TRUE(text.tool_calls.empty());
  EXPECT_TRUE(text.refusal.empty());

  const detail::ParsedCompletion refused = detail::ParseChatCompletion(
      MakeResponse({{"role", "assistant"},
                    {"content", nullptr},
                    {"refusal", "I can't help with that"}},
                   "stop"));
  ASSERT_TRUE(refused.ok) << refused.error;
  EXPECT_EQ(refused.refusal, "I can't help with that");
}

TEST(LlmOpenAIParserTest, ReportsUnreadableBodies) {
  EXPECT_FALSE(detail::ParseChatCompletion("not json").ok);
  EXPECT_FALSE(detail::ParseChatCompletion("{}").ok);
  EXPECT_FALSE(detail::ParseChatCompletion(R"({"choices":[]})").ok);
  EXPECT_FALSE(
      detail::ParseChatCompletion(R"({"choices":[{"message":"nope"}]})").ok);
}

TEST(LlmOpenAIParserTest, EmptyMessageIsNotAParseError) {
  // A completion may carry neither text nor tool calls; the provider reports
  // that as NoToolCalls rather than as a broken response.
  const detail::ParsedCompletion parsed =
      detail::ParseChatCompletion(R"({"choices":[{"message":{}}]})");
  ASSERT_TRUE(parsed.ok) << parsed.error;
  EXPECT_TRUE(parsed.assistant_text.empty());
  EXPECT_TRUE(parsed.tool_calls.empty());
}

TEST(LlmOpenAIParserTest, ExtractsProviderErrorMessages) {
  EXPECT_EQ(
      detail::ExtractErrorMessage(
          R"({"error":{"message":"bad model","type":"invalid_request_error"}})"),
      "bad model");
  EXPECT_EQ(detail::ExtractErrorMessage("plain text"), "plain text");
}

TEST(LlmOpenAIParserTest, MapsFinishReasons) {
  EXPECT_EQ(detail::MapFinishReason("stop"), FinishReason::Stop);
  EXPECT_EQ(detail::MapFinishReason("tool_calls"), FinishReason::ToolCalls);
  EXPECT_EQ(detail::MapFinishReason("length"), FinishReason::Length);
  EXPECT_EQ(detail::MapFinishReason("content_filter"),
            FinishReason::ContentFilter);
  EXPECT_EQ(detail::MapFinishReason("something-new"), FinishReason::Unknown);
}

// ---------------------------------------------------------------------------
// Provider behaviour
// ---------------------------------------------------------------------------
TEST(LlmOpenAIProviderTest, SuccessForAValidatedBatch) {
  Fixture fixture;
  fixture.transport->responses.push_back(
      FakeTransport::Ok(MakeResponse(TwoCallsMessage())));

  const CompletionResult result =
      fixture.Complete(MakeRequest(), fixture.Options());

  ASSERT_TRUE(result.IsSuccess()) << result.ErrorMessage();
  EXPECT_EQ(result.status, CompletionStatus::Success);
  EXPECT_FALSE(result.error.has_value());
  EXPECT_EQ(result.tool_calls.size(), 2u);
  EXPECT_EQ(result.model, "gpt-5.6-2026-01-01");
  EXPECT_EQ(result.finish_reason, FinishReason::ToolCalls);
  EXPECT_EQ(result.usage.total_tokens, 18);
  EXPECT_FALSE(result.provider_metadata.empty());

  ASSERT_EQ(fixture.transport->requests.size(), 1u)
      << "one completion is exactly one request";
  const HttpRequest& http = fixture.transport->requests[0];
  EXPECT_EQ(http.url, "https://api.openai.com/v1/chat/completions");
  EXPECT_EQ(http.method, "POST");
  bool has_authorization = false;
  for (const auto& header : http.headers) {
    if (header.name == "Authorization" && header.value == "Bearer sk-test") {
      has_authorization = true;
    }
  }
  EXPECT_TRUE(has_authorization);
}

TEST(LlmOpenAIProviderTest, NormalizesBaseUrl) {
  Fixture fixture;
  fixture.transport->responses.push_back(
      FakeTransport::Ok(MakeResponse(TwoCallsMessage())));

  ProviderOptions options = fixture.Options();
  options.base_url = "https://openrouter.ai/api/";
  fixture.Complete(MakeRequest(), std::move(options));

  ASSERT_EQ(fixture.transport->requests.size(), 1u);
  EXPECT_EQ(fixture.transport->requests[0].url,
            "https://openrouter.ai/api/v1/chat/completions");
}

TEST(LlmOpenAIProviderTest, TextOnlyAnswerIsNotSuccess) {
  Fixture fixture;
  fixture.transport->responses.push_back(FakeTransport::Ok(MakeResponse(
      {{"role", "assistant"}, {"content", "I'd rather talk."}}, "stop")));

  const CompletionResult result =
      fixture.Complete(MakeRequest(), fixture.Options());

  EXPECT_EQ(result.status, CompletionStatus::NoToolCalls);
  EXPECT_FALSE(result.IsSuccess());
  EXPECT_EQ(result.assistant_text, "I'd rather talk.");
  EXPECT_TRUE(result.tool_calls.empty());
}

TEST(LlmOpenAIProviderTest, TokenLimitIsTruncation) {
  Fixture fixture;
  fixture.transport->responses.push_back(
      FakeTransport::Ok(MakeResponse(TwoCallsMessage(), "length")));

  const CompletionResult result =
      fixture.Complete(MakeRequest(), fixture.Options());

  EXPECT_EQ(result.status, CompletionStatus::Truncated);
  EXPECT_FALSE(result.IsSuccess());
}

TEST(LlmOpenAIProviderTest, RefusalAndContentFilterAreRefusals) {
  Fixture refusal_fixture;
  refusal_fixture.transport->responses.push_back(
      FakeTransport::Ok(MakeResponse({{"role", "assistant"},
                                      {"content", nullptr},
                                      {"refusal", "I can't do that"}},
                                     "stop")));
  EXPECT_EQ(
      refusal_fixture.Complete(MakeRequest(), refusal_fixture.Options()).status,
      CompletionStatus::Refused);

  Fixture filtered_fixture;
  filtered_fixture.transport->responses.push_back(
      FakeTransport::Ok(MakeResponse(
          {{"role", "assistant"}, {"content", nullptr}}, "content_filter")));
  EXPECT_EQ(filtered_fixture.Complete(MakeRequest(), filtered_fixture.Options())
                .status,
            CompletionStatus::Refused);
}

TEST(LlmOpenAIProviderTest, AuthFailureIsNotRetried) {
  Fixture fixture;
  fixture.transport->responses.push_back(FakeTransport::WithStatus(
      401, R"({"error":{"message":"Incorrect API key"}})"));

  const CompletionResult result =
      fixture.Complete(MakeRequest(), fixture.Options());

  EXPECT_EQ(result.status, CompletionStatus::ProviderError);
  ASSERT_TRUE(result.error.has_value());
  EXPECT_EQ(result.error->kind, ErrorKind::Auth);
  EXPECT_EQ(result.error->http_status, 401);
  EXPECT_FALSE(result.error->retryable);
  EXPECT_NE(result.ErrorMessage().find("Incorrect API key"), std::string::npos);
  EXPECT_EQ(fixture.transport->requests.size(), 1u);
}

TEST(LlmOpenAIProviderTest, RateLimitIsRetriedOnce) {
  Fixture fixture;
  fixture.transport->responses.push_back(
      FakeTransport::WithStatus(429, R"({"error":{"message":"slow down"}})"));
  fixture.transport->responses.push_back(
      FakeTransport::Ok(MakeResponse(TwoCallsMessage())));

  const CompletionResult result =
      fixture.Complete(MakeRequest(), fixture.Options());

  EXPECT_TRUE(result.IsSuccess()) << result.ErrorMessage();
  EXPECT_EQ(fixture.transport->requests.size(), 2u);
}

TEST(LlmOpenAIProviderTest, TransportFailuresAreRetriedThenReported) {
  Fixture fixture;
  fixture.transport->responses.push_back(FakeTransport::Timeout());

  ProviderOptions options = fixture.Options();
  options.retry.max_attempts = 2;
  const CompletionResult result =
      fixture.Complete(MakeRequest(), std::move(options));

  EXPECT_EQ(result.status, CompletionStatus::ProviderError);
  ASSERT_TRUE(result.error.has_value());
  EXPECT_EQ(result.error->kind, ErrorKind::Timeout);
  EXPECT_TRUE(result.error->retryable);
  EXPECT_EQ(fixture.transport->requests.size(), 2u);
}

TEST(LlmOpenAIProviderTest, UnreadableBodyIsAParseError) {
  Fixture fixture;
  fixture.transport->responses.push_back(
      FakeTransport::Ok("<html>nope</html>"));

  const CompletionResult result =
      fixture.Complete(MakeRequest(), fixture.Options());

  EXPECT_EQ(result.status, CompletionStatus::ProviderError);
  ASSERT_TRUE(result.error.has_value());
  EXPECT_EQ(result.error->kind, ErrorKind::Parse);
}

TEST(LlmOpenAIProviderTest, MissingApiKeyNeverReachesTheTransport) {
  Fixture fixture;
  ProviderOptions options = fixture.Options();
  options.api_key.clear();

  const CompletionResult result =
      fixture.Complete(MakeRequest(), std::move(options));

  EXPECT_EQ(result.status, CompletionStatus::ProviderError);
  ASSERT_TRUE(result.error.has_value());
  EXPECT_EQ(result.error->kind, ErrorKind::Auth);
  EXPECT_TRUE(fixture.transport->requests.empty());
}

TEST(LlmOpenAIProviderTest, InvalidRequestNeverReachesTheTransport) {
  Fixture fixture;
  ModelRequest request = MakeRequest();
  request.tools[0].parameters_schema_json = "{oops";

  const CompletionResult result = fixture.Complete(request, fixture.Options());

  EXPECT_EQ(result.status, CompletionStatus::ProviderError);
  ASSERT_TRUE(result.error.has_value());
  EXPECT_EQ(result.error->kind, ErrorKind::InvalidRequest);
  EXPECT_TRUE(fixture.transport->requests.empty());
}

TEST(LlmOpenAIProviderTest, MalformedBatchIsNotSuccess) {
  // Missing name.
  {
    nlohmann::json message = TwoCallsMessage();
    message["tool_calls"][0]["function"]["name"] = "";
    Fixture fixture;
    fixture.transport->responses.push_back(
        FakeTransport::Ok(MakeResponse(message)));
    EXPECT_EQ(fixture.Complete(MakeRequest(), fixture.Options()).status,
              CompletionStatus::InvalidToolCalls);
  }
  // Missing id: the batch could not be answered with tool results.
  {
    nlohmann::json message = TwoCallsMessage();
    message["tool_calls"][0]["id"] = "";
    Fixture fixture;
    fixture.transport->responses.push_back(
        FakeTransport::Ok(MakeResponse(message)));
    EXPECT_EQ(fixture.Complete(MakeRequest(), fixture.Options()).status,
              CompletionStatus::InvalidToolCalls);
  }
  // Arguments that are not JSON.
  {
    nlohmann::json message = TwoCallsMessage();
    message["tool_calls"][0]["function"]["arguments"] = "{not json";
    Fixture fixture;
    fixture.transport->responses.push_back(
        FakeTransport::Ok(MakeResponse(message)));
    EXPECT_EQ(fixture.Complete(MakeRequest(), fixture.Options()).status,
              CompletionStatus::InvalidToolCalls);
  }
  // More calls than the request allows.
  {
    Fixture fixture;
    fixture.transport->responses.push_back(
        FakeTransport::Ok(MakeResponse(TwoCallsMessage())));
    ModelRequest request = MakeRequest();
    request.max_tool_calls = 1;
    const CompletionResult result =
        fixture.Complete(request, fixture.Options());
    EXPECT_EQ(result.status, CompletionStatus::InvalidToolCalls);
    EXPECT_TRUE(result.HasToolCalls())
        << "the batch is still reported for diagnostics";
  }
}

TEST(LlmOpenAIProviderTest, UnsupportedRequirementsAreRefusedNotIgnored) {
  // A server that ignores `strict` and forced tool_choice must not be credited
  // with enforcing them.
  auto options_without_guarantees = [](const Fixture& fixture) {
    ProviderOptions options = fixture.Options();
    options.capabilities.strict_tool_schema = false;
    options.capabilities.forced_tool_choice = false;
    return options;
  };

  {
    Fixture fixture;
    const CompletionResult result =
        fixture.Complete(MakeRequest(), options_without_guarantees(fixture));
    EXPECT_EQ(result.status, CompletionStatus::Unsupported);
    ASSERT_TRUE(result.error.has_value());
    EXPECT_EQ(result.error->kind, ErrorKind::Unsupported);
    EXPECT_TRUE(fixture.transport->requests.empty())
        << "nothing must be sent when a requirement cannot be met";
  }

  // With an explicit downgrade the call proceeds, under SDK-side validation,
  // and the caller is told about it.
  {
    Fixture fixture;
    fixture.transport->responses.push_back(
        FakeTransport::Ok(MakeResponse(TwoCallsMessage())));
    ProviderOptions options = options_without_guarantees(fixture);
    options.allow_validation_downgrade = true;

    const CompletionResult result =
        fixture.Complete(MakeRequest(), std::move(options));

    EXPECT_TRUE(result.IsSuccess()) << result.ErrorMessage();
    ASSERT_FALSE(result.warnings.empty());
    EXPECT_NE(result.warnings[0].find("downgraded"), std::string::npos);
    EXPECT_EQ(fixture.transport->requests.size(), 1u);
  }
}

TEST(LlmOpenAIProviderTest, ExposesNameAndCapabilities) {
  Fixture fixture;
  OpenAIProvider provider(fixture.Options());

  EXPECT_EQ(provider.Name(), "openai");
  const ProviderCapabilities capabilities = provider.Capabilities();
  EXPECT_TRUE(capabilities.multiple_tool_calls);
  EXPECT_TRUE(capabilities.forced_tool_choice);
  EXPECT_TRUE(capabilities.strict_tool_schema);
  EXPECT_TRUE(capabilities.tool_result_history);
}

// Exercises the real (cpp-httplib) transport, which the other tests replace
// with a fake: a refused connection must surface as a retryable provider
// error rather than as a crash or a silent success.
TEST(LlmOpenAITransportTest, DefaultTransportReportsConnectionFailures) {
  ProviderOptions options;
  options.api_key = "sk-test";
  options.base_url = "http://127.0.0.1:1";  // nothing listens on port 1
  options.timeout_seconds = 2;
  options.retry.max_attempts = 1;

  OpenAIProvider provider(std::move(options));
  const CompletionResult result = provider.Complete(MakeRequest());

  EXPECT_EQ(result.status, CompletionStatus::ProviderError);
  ASSERT_TRUE(result.error.has_value());
  EXPECT_TRUE(result.error->kind == ErrorKind::Network ||
              result.error->kind == ErrorKind::Timeout)
      << "unexpected kind: " << ToString(result.error->kind);
  EXPECT_TRUE(result.error->retryable);
}

// ---------------------------------------------------------------------------
// Validation integration (PROTO-4)
// ---------------------------------------------------------------------------
TEST(LlmOpenAIProviderTest, UnknownToolIsNotSuccess) {
  nlohmann::json message = TwoCallsMessage();
  message["tool_calls"][1]["function"]["name"] = "delete_everything";

  Fixture fixture;
  fixture.transport->responses.push_back(
      FakeTransport::Ok(MakeResponse(message)));

  const CompletionResult result =
      fixture.Complete(MakeRequest(), fixture.Options());

  EXPECT_EQ(result.status, CompletionStatus::InvalidToolCalls);
  EXPECT_FALSE(result.IsSuccess());
  EXPECT_NE(result.ErrorMessage().find("delete_everything"), std::string::npos);
}

TEST(LlmOpenAIProviderTest, SchemaViolationIsNotSuccess) {
  nlohmann::json message = TwoCallsMessage();
  // level must be an integer, and height must be one of the listed values.
  message["tool_calls"][0]["function"]["arguments"] = R"({"level":"high"})";

  Fixture fixture;
  fixture.transport->responses.push_back(
      FakeTransport::Ok(MakeResponse(message)));

  const CompletionResult result =
      fixture.Complete(MakeRequest(), fixture.Options());

  EXPECT_EQ(result.status, CompletionStatus::InvalidToolCalls);
  EXPECT_TRUE(result.HasToolCalls())
      << "an unvalidated batch is still reported for diagnostics";
  EXPECT_NE(result.ErrorMessage().find("level"), std::string::npos);
}

TEST(LlmOpenAIProviderTest, InjectedValidatorDecides) {
  // An application policy: one change per decision, whatever the schema says.
  class SingleCallPolicy : public ToolCallValidator {
   public:
    ValidationReport Validate(
        const ModelRequest&,
        const std::vector<ToolCall>& tool_calls) const override {
      ValidationReport report;
      if (tool_calls.size() != 1) {
        ValidationIssue issue;
        issue.message = "exactly one tool call per decision";
        report.issues.push_back(std::move(issue));
      }
      report.ok = report.issues.empty();
      return report;
    }
  };

  Fixture fixture;
  fixture.transport->responses.push_back(
      FakeTransport::Ok(MakeResponse(TwoCallsMessage())));

  ProviderOptions options = fixture.Options();
  options.validator = std::make_shared<const SingleCallPolicy>();
  const CompletionResult result =
      fixture.Complete(MakeRequest(), std::move(options));

  EXPECT_EQ(result.status, CompletionStatus::InvalidToolCalls);
  EXPECT_NE(result.ErrorMessage().find("exactly one tool call"),
            std::string::npos);
}

TEST(LlmOpenAIProviderTest, DowngradeStillValidates) {
  // Accepting SDK-side validation instead of provider-side enforcement does not
  // mean skipping validation.
  nlohmann::json message = TwoCallsMessage();
  message["tool_calls"][0]["function"]["name"] = "unknown_tool";

  Fixture fixture;
  fixture.transport->responses.push_back(
      FakeTransport::Ok(MakeResponse(message)));

  ProviderOptions options = fixture.Options();
  options.capabilities.strict_tool_schema = false;  // tools request strict
  options.allow_validation_downgrade = true;

  const CompletionResult result =
      fixture.Complete(MakeRequest(), std::move(options));

  EXPECT_EQ(result.status, CompletionStatus::InvalidToolCalls);
  EXPECT_FALSE(result.warnings.empty());
}

// ---------------------------------------------------------------------------
// Whole protocol round trip (PROTO-5)
// ---------------------------------------------------------------------------
TEST(LlmOpenAIProtocolTest, PreviousDecisionRoundTripsThroughTheProvider) {
  Fixture fixture;
  fixture.transport->responses.push_back(
      FakeTransport::Ok(MakeResponse(TwoCallsMessage())));
  fixture.transport->responses.push_back(
      FakeTransport::Ok(MakeResponse(TwoCallsMessage())));

  // 1. The first decision comes back from the provider.
  const CompletionResult decision =
      fixture.Complete(MakeRequest(), fixture.Options());
  ASSERT_TRUE(decision.IsSuccess()) << decision.ErrorMessage();
  ASSERT_EQ(decision.tool_calls.size(), 2u);

  // 2. The caller applies it and assembles the next context: the assistant turn
  //    with those tool calls, one tool turn per applied call, then new state.
  ModelRequest second = MakeRequest();
  second.messages.push_back(decision.ToAssistantMessage());
  for (const ToolCall& call : decision.tool_calls) {
    ToolResult applied;
    applied.tool_call_id = call.id;
    applied.name = call.name;
    applied.content_json = R"({"applied":true})";
    second.messages.push_back(Message::ToolResult(applied));
  }
  second.messages.push_back(Message::User("Opponent switched to a low block."));

  // 3. The second request must serialize that history as OpenAI messages.
  const CompletionResult second_decision =
      fixture.Complete(second, fixture.Options());
  ASSERT_TRUE(second_decision.IsSuccess()) << second_decision.ErrorMessage();
  ASSERT_EQ(fixture.transport->requests.size(), 2u);

  const nlohmann::json body = nlohmann::json::parse(
      fixture.transport->requests[1].body, nullptr, false);
  ASSERT_FALSE(body.is_discarded());

  // system, user, assistant(tool calls), tool, tool, user
  const auto& messages = body["messages"];
  ASSERT_EQ(messages.size(), 6u);

  const auto& assistant = messages[2];
  EXPECT_EQ(assistant["role"], "assistant");
  EXPECT_TRUE(assistant["content"].is_null());
  ASSERT_EQ(assistant["tool_calls"].size(), 2u);
  EXPECT_EQ(assistant["tool_calls"][0]["id"], decision.tool_calls[0].id);
  EXPECT_EQ(assistant["tool_calls"][0]["function"]["name"],
            decision.tool_calls[0].name);
  // The argument text survives the round trip byte for byte.
  EXPECT_EQ(
      assistant["tool_calls"][0]["function"]["arguments"].get<std::string>(),
      decision.tool_calls[0].arguments_json);
  EXPECT_EQ(assistant["tool_calls"][1]["id"], decision.tool_calls[1].id);

  EXPECT_EQ(messages[3]["role"], "tool");
  EXPECT_EQ(messages[3]["tool_call_id"], decision.tool_calls[0].id);
  EXPECT_EQ(messages[3]["content"], R"({"applied":true})");
  EXPECT_EQ(messages[4]["tool_call_id"], decision.tool_calls[1].id);
  EXPECT_EQ(messages[5]["role"], "user");
  EXPECT_EQ(messages[5]["content"], "Opponent switched to a low block.");
}

// ---------------------------------------------------------------------------
// CTX-0: parameter dialect, strict parsing, attempt accounting
// ---------------------------------------------------------------------------
TEST(LlmOpenAIBuilderTest, ParameterDialectFollowsTheModelFamily) {
  // Reasoning-first families: max_completion_tokens, no sampling controls.
  ModelRequest reasoning = MakeRequest();
  reasoning.model = models::kGpt56;
  reasoning.max_tokens = 100;
  reasoning.temperature = 0.5;
  reasoning.top_p = 0.5;
  reasoning.seed = 7;
  const detail::BuildOutcome gpt5 =
      detail::BuildChatCompletionRequest(reasoning);
  ASSERT_TRUE(gpt5.ok) << gpt5.error;
  EXPECT_EQ(gpt5.body["max_completion_tokens"], 100);
  EXPECT_FALSE(gpt5.body.contains("max_tokens"));
  EXPECT_FALSE(gpt5.body.contains("temperature"));
  EXPECT_FALSE(gpt5.body.contains("top_p"));
  EXPECT_FALSE(gpt5.body.contains("seed"));

  // Legacy families keep the documented fields.
  ModelRequest legacy = reasoning;
  legacy.model = models::kGpt41;
  const detail::BuildOutcome gpt41 = detail::BuildChatCompletionRequest(legacy);
  ASSERT_TRUE(gpt41.ok) << gpt41.error;
  EXPECT_EQ(gpt41.body["max_tokens"], 100);
  EXPECT_FALSE(gpt41.body.contains("max_completion_tokens"));
  EXPECT_DOUBLE_EQ(gpt41.body["temperature"].get<double>(), 0.5);
  EXPECT_EQ(gpt41.body["seed"], 7);

  // An endpoint that differs from OpenAI can be told so explicitly.
  const detail::BuildOutcome overridden = detail::BuildChatCompletionRequest(
      reasoning, ParameterDialect::LegacyTokens);
  ASSERT_TRUE(overridden.ok) << overridden.error;
  EXPECT_EQ(overridden.body["max_tokens"], 100);
  EXPECT_FALSE(overridden.body.contains("max_completion_tokens"));

  EXPECT_EQ(DialectForModel(models::kGpt56), ParameterDialect::ReasoningTokens);
  EXPECT_EQ(DialectForModel("o3-mini"), ParameterDialect::ReasoningTokens);
  EXPECT_EQ(DialectForModel(models::kGpt41), ParameterDialect::LegacyTokens);
  EXPECT_EQ(DialectForModel("some-local-model"),
            ParameterDialect::LegacyTokens);
}

TEST(LlmOpenAIParserTest, MissingArgumentsAreNotInvented) {
  nlohmann::json message = TwoCallsMessage();
  message["tool_calls"][0]["function"].erase("arguments");

  const detail::ParsedCompletion parsed =
      detail::ParseChatCompletion(MakeResponse(message));

  EXPECT_FALSE(parsed.ok);
  EXPECT_NE(parsed.error.find("no arguments"), std::string::npos);

  // Same for an explicit null: the SDK never fills in arguments.
  nlohmann::json null_message = TwoCallsMessage();
  null_message["tool_calls"][0]["function"]["arguments"] = nullptr;
  EXPECT_FALSE(detail::ParseChatCompletion(MakeResponse(null_message)).ok);
}

TEST(LlmOpenAIProviderTest, UnenforceableSchemaIsRefusedBeforeSending) {
  Fixture fixture;
  ModelRequest request = MakeRequest();
  request.tools[0].parameters_schema_json =
      R"({"type":"object","properties":{"level":{"type":"integer","pattern":"[0-5]"}}})";

  const CompletionResult result = fixture.Complete(request, fixture.Options());

  EXPECT_EQ(result.status, CompletionStatus::Unsupported);
  ASSERT_TRUE(result.error.has_value());
  EXPECT_EQ(result.error->kind, ErrorKind::Unsupported);
  EXPECT_NE(result.ErrorMessage().find("pattern"), std::string::npos);
  EXPECT_TRUE(fixture.transport->requests.empty());
  EXPECT_EQ(result.http_attempts, 0);
}

TEST(LlmOpenAIProviderTest, AttemptsAreAccountedFor) {
  // Refused before sending: nothing was attempted.
  {
    Fixture fixture;
    ProviderOptions options = fixture.Options();
    options.api_key.clear();
    EXPECT_EQ(fixture.Complete(MakeRequest(), std::move(options)).http_attempts,
              0);
  }
  // One request.
  {
    Fixture fixture;
    fixture.transport->responses.push_back(
        FakeTransport::Ok(MakeResponse(TwoCallsMessage())));
    const CompletionResult result =
        fixture.Complete(MakeRequest(), fixture.Options());
    ASSERT_TRUE(result.IsSuccess()) << result.ErrorMessage();
    EXPECT_EQ(result.http_attempts, 1);
  }
  // Retried once.
  {
    Fixture fixture;
    fixture.transport->responses.push_back(
        FakeTransport::WithStatus(429, R"({"error":{"message":"slow down"}})"));
    fixture.transport->responses.push_back(
        FakeTransport::Ok(MakeResponse(TwoCallsMessage())));
    const CompletionResult result =
        fixture.Complete(MakeRequest(), fixture.Options());
    ASSERT_TRUE(result.IsSuccess()) << result.ErrorMessage();
    EXPECT_EQ(result.http_attempts, 2);
  }
  // Exhausted the budget.
  {
    Fixture fixture;
    fixture.transport->responses.push_back(FakeTransport::Timeout());
    ProviderOptions options = fixture.Options();
    options.retry.max_attempts = 3;
    const CompletionResult result =
        fixture.Complete(MakeRequest(), std::move(options));
    EXPECT_EQ(result.status, CompletionStatus::ProviderError);
    EXPECT_EQ(result.http_attempts, 3);
  }
}

TEST(LlmOpenAIProviderTest, TimeoutIsForwardedToTheTransport) {
  Fixture fixture;
  fixture.transport->responses.push_back(FakeTransport::Timeout());

  ProviderOptions options = fixture.Options();
  options.timeout_seconds = 7;
  options.retry.max_attempts = 1;
  const CompletionResult result =
      fixture.Complete(MakeRequest(), std::move(options));

  ASSERT_EQ(fixture.transport->requests.size(), 1u);
  EXPECT_EQ(fixture.transport->requests[0].timeout_seconds, 7);
  EXPECT_EQ(result.error->kind, ErrorKind::Timeout);
}

}  // namespace
}  // namespace openai
}  // namespace llm
