// Tests for the provider's log events, driven through a scripted transport.

#include "capture_logger.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <llm/core/logger.h>
#include <llm/core/request.h>
#include <llm/core/result.h>
#include <llm/providers/openai/openai.h>
#include <llm/providers/openai/transport.h>
#include <nlohmann/json.hpp>

namespace llm {
namespace openai {
namespace {

constexpr char kPromptText[] = "Opponent presses high down our left side.";

ToolDefinition PressingTool() {
  ToolDefinition tool;
  tool.name = "set_pressing_intensity";
  tool.description = "pressing";
  tool.parameters_schema_json =
      R"({"type":"object","properties":{"level":{"type":"integer","minimum":1,"maximum":5}},"required":["level"],"additionalProperties":false})";
  tool.strict = true;
  return tool;
}

std::string ToolCallResponse(const std::string& arguments = R"({"level":4})") {
  const nlohmann::json body{
      {"model", "gpt-5.6"},
      {"choices",
       nlohmann::json::array({nlohmann::json{
           {"finish_reason", "tool_calls"},
           {"message",
            {{"role", "assistant"},
             {"content", nullptr},
             {"tool_calls", nlohmann::json::array({nlohmann::json{
                                {"id", "call_a"},
                                {"type", "function"},
                                {"function",
                                 {{"name", "set_pressing_intensity"},
                                  {"arguments", arguments}}}}})}}}}})},
      {"usage",
       {{"prompt_tokens", 10},
        {"completion_tokens", 5},
        {"total_tokens", 15}}}};
  return body.dump();
}

class ScriptedTransport : public ITransport {
 public:
  std::vector<HttpResponse> responses;
  int calls = 0;

  HttpResponse Send(const HttpRequest&) override {
    ++calls;
    if (responses.empty()) {
      HttpResponse response;
      response.transport_ok = false;
      response.error = "no scripted response";
      return response;
    }
    const HttpResponse response = responses.front();
    if (responses.size() > 1) {
      responses.erase(responses.begin());
    }
    return response;
  }

  static HttpResponse Ok(std::string body) {
    return HttpResponse{true, 200, std::move(body), "", TransportError::None};
  }
  static HttpResponse Status(int status, std::string body) {
    return HttpResponse{true, status, std::move(body), "",
                        TransportError::None};
  }
  static HttpResponse Timeout() {
    return HttpResponse{false, 0, "", "read timeout", TransportError::Timeout};
  }
};

ModelRequest MakeRequest() {
  ModelRequest request;
  request.model = models::kGpt56;
  request.messages = {Message::System("You are a head coach."),
                      Message::User(kPromptText)};
  request.tools = {PressingTool()};
  request.log.session_id = "home";
  request.log.request_id = 7;
  return request;
}

struct Fixture {
  std::shared_ptr<ScriptedTransport> transport =
      std::make_shared<ScriptedTransport>();
  std::shared_ptr<CaptureLogger> logger = std::make_shared<CaptureLogger>();

  ProviderOptions Options() const {
    ProviderOptions options;
    options.api_key = "sk-test";
    options.transport = transport;
    options.logger = logger;
    options.retry.initial_delay_ms = 0;
    return options;
  }

  CompletionResult Complete(const ModelRequest& request,
                            ProviderOptions options) {
    return OpenAIProvider(std::move(options)).Complete(request);
  }
};

TEST(LlmOpenAILoggingTest, StartedAndCompletedShareTheRequestContext) {
  Fixture fixture;
  fixture.transport->responses.push_back(
      ScriptedTransport::Ok(ToolCallResponse()));

  const CompletionResult result =
      fixture.Complete(MakeRequest(), fixture.Options());
  ASSERT_TRUE(result.IsSuccess()) << result.ErrorMessage();

  const std::vector<LogRecord> records = fixture.logger->Records();
  ASSERT_EQ(records.size(), 2u);
  EXPECT_EQ(records[0].event, log_event::kRequestStarted);
  EXPECT_EQ(records[0].level, LogLevel::Debug);
  EXPECT_NE(records[0].message.find("model=gpt-5.6"), std::string::npos);
  EXPECT_NE(records[0].message.find("tool_choice=required"), std::string::npos);

  EXPECT_EQ(records[1].event, log_event::kRequestCompleted);
  EXPECT_EQ(records[1].level, LogLevel::Info);
  EXPECT_NE(records[1].message.find("status=success"), std::string::npos);
  EXPECT_NE(records[1].message.find("http=200"), std::string::npos);
  EXPECT_NE(records[1].message.find("attempts=1"), std::string::npos);
  EXPECT_NE(records[1].message.find("tokens=15"), std::string::npos);
  EXPECT_NE(records[1].message.find("tools=1"), std::string::npos);

  for (const LogRecord& record : records) {
    EXPECT_EQ(record.session_id, "home")
        << "a shared provider must be attributable to a session";
    EXPECT_EQ(record.request_id, 7u);
  }
}

TEST(LlmOpenAILoggingTest, RetriesAreVisible) {
  Fixture fixture;
  fixture.transport->responses.push_back(
      ScriptedTransport::Status(429, R"({"error":{"message":"slow down"}})"));
  fixture.transport->responses.push_back(
      ScriptedTransport::Ok(ToolCallResponse()));

  const CompletionResult result =
      fixture.Complete(MakeRequest(), fixture.Options());

  ASSERT_TRUE(result.IsSuccess()) << result.ErrorMessage();
  EXPECT_EQ(result.http_attempts, 2);
  bool found = false;
  const LogRecord retry =
      fixture.logger->First(log_event::kRequestRetry, found);
  ASSERT_TRUE(found) << "a retry must be logged";
  EXPECT_EQ(retry.level, LogLevel::Warning);
  EXPECT_NE(retry.message.find("attempt=1"), std::string::npos);
  EXPECT_NE(retry.message.find("http=429"), std::string::npos);
  EXPECT_NE(retry.message.find("kind=rate limit"), std::string::npos);
  EXPECT_NE(retry.message.find("next_delay_ms"), std::string::npos);

  const std::vector<std::string> completed =
      fixture.logger->Messages(log_event::kRequestCompleted);
  ASSERT_EQ(completed.size(), 1u);
  EXPECT_NE(completed[0].find("attempts=2"), std::string::npos)
      << "retries belong to one logical completion";
}

TEST(LlmOpenAILoggingTest, FailuresAreClassified) {
  {
    Fixture fixture;
    fixture.transport->responses.push_back(ScriptedTransport::Timeout());
    ProviderOptions options = fixture.Options();
    options.retry.max_attempts = 2;

    const CompletionResult result =
        fixture.Complete(MakeRequest(), std::move(options));

    EXPECT_EQ(result.status, CompletionStatus::ProviderError);
    bool found = false;
    const LogRecord failed =
        fixture.logger->First(log_event::kRequestFailed, found);
    ASSERT_TRUE(found);
    EXPECT_EQ(failed.level, LogLevel::Error);
    EXPECT_NE(failed.message.find("kind=timeout"), std::string::npos);
    EXPECT_NE(failed.message.find("attempts=2"), std::string::npos);
    EXPECT_FALSE(fixture.logger->Has(log_event::kRequestCompleted));
  }
  {
    Fixture fixture;
    fixture.transport->responses.push_back(ScriptedTransport::Status(
        401, R"({"error":{"message":"Incorrect API key"}})"));

    fixture.Complete(MakeRequest(), fixture.Options());

    const std::vector<std::string> failures =
        fixture.logger->Messages(log_event::kRequestFailed);
    ASSERT_EQ(failures.size(), 1u);
    EXPECT_NE(failures[0].find("kind=auth"), std::string::npos);
    EXPECT_NE(failures[0].find("http=401"), std::string::npos);
    EXPECT_EQ(failures[0].find("Incorrect API key"), std::string::npos)
        << "the provider's text belongs in CompletionResult, not in the log";
  }
}

TEST(LlmOpenAILoggingTest, ValidationFailuresAreLogged) {
  Fixture fixture;
  // level is outside the schema range.
  fixture.transport->responses.push_back(
      ScriptedTransport::Ok(ToolCallResponse(R"({"level":9})")));

  const CompletionResult result =
      fixture.Complete(MakeRequest(), fixture.Options());

  EXPECT_EQ(result.status, CompletionStatus::InvalidToolCalls);
  const std::vector<std::string> failures =
      fixture.logger->Messages(log_event::kValidationFailed);
  ASSERT_EQ(failures.size(), 1u);
  EXPECT_NE(failures[0].find("level"), std::string::npos);

  // The completion is still reported once, with the status the caller sees.
  const std::vector<std::string> completed =
      fixture.logger->Messages(log_event::kRequestCompleted);
  ASSERT_EQ(completed.size(), 1u);
  EXPECT_NE(completed[0].find("status=invalid tool calls"), std::string::npos);
}

TEST(LlmOpenAILoggingTest, UnenforceableSchemaIsLoggedBeforeSending) {
  Fixture fixture;
  ModelRequest request = MakeRequest();
  request.tools[0].parameters_schema_json =
      R"({"type":"object","properties":{"level":{"type":"integer","pattern":"[0-5]"}}})";

  const CompletionResult result = fixture.Complete(request, fixture.Options());

  EXPECT_EQ(result.status, CompletionStatus::Unsupported);
  EXPECT_EQ(fixture.transport->calls, 0) << "nothing may be sent";
  EXPECT_TRUE(fixture.logger->Has(log_event::kValidationFailed));
  EXPECT_FALSE(fixture.logger->Has(log_event::kRequestStarted));
  EXPECT_FALSE(fixture.logger->Has(log_event::kRequestCompleted));
}

TEST(LlmOpenAILoggingTest, DowngradeIsReportedAsAWarning) {
  Fixture fixture;
  fixture.transport->responses.push_back(
      ScriptedTransport::Ok(ToolCallResponse()));

  ProviderOptions options = fixture.Options();
  options.capabilities.strict_tool_schema = false;
  options.allow_validation_downgrade = true;

  const CompletionResult result =
      fixture.Complete(MakeRequest(), std::move(options));

  ASSERT_TRUE(result.IsSuccess()) << result.ErrorMessage();
  EXPECT_FALSE(result.warnings.empty());
  bool found = false;
  const LogRecord downgraded =
      fixture.logger->First(log_event::kRequirementDowngraded, found);
  ASSERT_TRUE(found);
  EXPECT_EQ(downgraded.level, LogLevel::Warning);
}

TEST(LlmOpenAILoggingTest, ProviderTextNeverEntersTheLog) {
  Fixture fixture;
  // A text-only answer: the failure detail embeds the model's own words.
  const nlohmann::json body{
      {"model", "gpt-5.6"},
      {"choices",
       nlohmann::json::array({nlohmann::json{
           {"finish_reason", "stop"},
           {"message",
            {{"role", "assistant"}, {"content", "I would rather talk."}}}}})},
      {"usage",
       {{"prompt_tokens", 3}, {"completion_tokens", 5}, {"total_tokens", 8}}}};
  fixture.transport->responses.push_back(ScriptedTransport::Ok(body.dump()));

  const CompletionResult result =
      fixture.Complete(MakeRequest(), fixture.Options());

  ASSERT_EQ(result.status, CompletionStatus::NoToolCalls);
  EXPECT_NE(result.ErrorMessage().find("I would rather talk."),
            std::string::npos)
      << "the caller still gets the detail in the result";
  for (const LogRecord& record : fixture.logger->Records()) {
    EXPECT_EQ(record.message.find("I would rather talk."), std::string::npos)
        << "model text must not reach the log: " << record.message;
  }
  EXPECT_NE(fixture.logger->Messages(log_event::kRequestCompleted)[0].find(
                "status=no tool calls"),
            std::string::npos)
      << "the status is the classification the log carries";
}

TEST(LlmOpenAILoggingTest, OnlyMetadataIsLogged) {
  Fixture fixture;
  fixture.transport->responses.push_back(
      ScriptedTransport::Ok(ToolCallResponse()));

  const CompletionResult result =
      fixture.Complete(MakeRequest(), fixture.Options());
  ASSERT_TRUE(result.IsSuccess());

  for (const LogRecord& record : fixture.logger->Records()) {
    EXPECT_EQ(record.message.find(kPromptText), std::string::npos)
        << "prompt text must not be logged: " << record.message;
    EXPECT_EQ(record.message.find("sk-test"), std::string::npos)
        << "credentials must not be logged: " << record.message;
    EXPECT_EQ(record.message.find(R"({"level":4})"), std::string::npos)
        << "tool arguments must not be logged: " << record.message;
    EXPECT_EQ(record.message.find("arguments"), std::string::npos)
        << "argument text must not be logged: " << record.message;
  }
}

TEST(LlmOpenAILoggingTest, NoLoggerMeansNoEvents) {
  Fixture fixture;
  fixture.transport->responses.push_back(
      ScriptedTransport::Ok(ToolCallResponse()));

  ProviderOptions options = fixture.Options();
  options.logger = nullptr;
  const CompletionResult result =
      fixture.Complete(MakeRequest(), std::move(options));

  EXPECT_TRUE(result.IsSuccess());
  EXPECT_TRUE(fixture.logger->Records().empty());
}

}  // namespace
}  // namespace openai
}  // namespace llm
