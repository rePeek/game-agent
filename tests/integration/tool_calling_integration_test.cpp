#include "../utils/test_fixtures.h"
#include "ai/logger.h"
#include "ai/openai.h"
#include "ai/tools.h"
#include "ai/types/generate_options.h"
#include "ai/types/tool.h"

#include <future>
#include <memory>
#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace ai {
namespace test {

// Tool definitions used by the integration tests. The SDK reports tool calls
// and never executes them, so these fixtures describe tools only.
class ToolTestFixtures {
 public:
  static Tool createWeatherTool() {
    return create_tool_schema("Get current weather information for a location",
                              create_object_schema({{"location", "string"}}));
  }

  static Tool createCalculatorTool() {
    auto schema =
        JsonValue{{"type", "object"},
                  {"properties",
                   {{"operation",
                     {{"type", "string"},
                      {"enum", {"add", "subtract", "multiply", "divide"}}}},
                    {"a", {{"type", "number"}}},
                    {"b", {{"type", "number"}}}}},
                  {"required", {"operation", "a", "b"}}};

    return Tool("Perform basic arithmetic operations", schema);
  }

  static Tool createTimeTool() {
    return create_tool_schema("Get current time in specified timezone",
                              create_object_schema({{"timezone", "string"}}));
  }

  static ToolSet createTestToolSet() {
    return {{"weather", createWeatherTool()},
            {"calculator", createCalculatorTool()},
            {"time", createTimeTool()}};
  }
};

// Parameterized test class for the OpenAI-compatible provider
class ToolCallingIntegrationTest
    : public ::testing::TestWithParam<std::string> {
 protected:
  void SetUp() override {
    std::string provider = GetParam();

    if (provider == "openai") {
      const char* api_key = std::getenv("OPENAI_API_KEY");
      if (api_key && *api_key != '\0') {
        use_real_api_ = true;
        client_ = ai::openai::create_client(api_key);
        model_ = ai::openai::models::kGpt56Luna;
      } else {
        use_real_api_ = false;
      }
    }

    tools_ = ToolTestFixtures::createTestToolSet();
  }

  bool use_real_api_ = false;
  std::optional<ai::Client> client_;
  std::string model_;
  ToolSet tools_;
};

// Test single tool calling
TEST_P(ToolCallingIntegrationTest, SingleToolCall) {
  if (!use_real_api_) {
    GTEST_SKIP() << "No API key set for " << GetParam();
  }

  GenerateOptions options(model_, "What's the weather like in San Francisco?");
  options.tools = tools_;
  options.max_tokens = 200;

  auto result = client_->generate_text(options);

  // Result should be successful, though text might be empty if only tool calls
  // are returned
  EXPECT_TRUE(result.is_success())
      << "Expected successful result but got error: " << result.error_message();
  EXPECT_FALSE(result.error.has_value())
      << "Expected no error in successful result";

  // Should have called the weather tool
  EXPECT_TRUE(result.has_tool_calls());
  EXPECT_GT(result.tool_calls.size(), 0);

  auto weather_call = std::find_if(
      result.tool_calls.begin(), result.tool_calls.end(),
      [](const ToolCall& call) { return call.tool_name == "weather"; });
  EXPECT_NE(weather_call, result.tool_calls.end());

  if (weather_call != result.tool_calls.end()) {
    EXPECT_TRUE(weather_call->arguments.contains("location"));
  }
}

// Test multiple tool calls in one request
TEST_P(ToolCallingIntegrationTest, MultipleDifferentToolCalls) {
  if (!use_real_api_) {
    GTEST_SKIP() << "No API key set for " << GetParam();
  }

  GenerateOptions options(
      model_, "What's the weather in New York and what's 25 plus 17?");
  options.tools = tools_;
  options.max_tokens = 300;

  auto result = client_->generate_text(options);

  // Result should be successful, text might be empty if only tool calls are
  // returned
  EXPECT_TRUE(result.is_success())
      << "Expected successful result but got error: " << result.error_message();
  EXPECT_FALSE(result.error.has_value())
      << "Expected no error in successful result";

  // Should have made tool calls
  EXPECT_TRUE(result.has_tool_calls());
  EXPECT_GE(result.tool_calls.size(), 1);

  // Check for weather tool call
  bool found_weather = false;
  bool found_calculator = false;

  for (const auto& call : result.tool_calls) {
    if (call.tool_name == "weather")
      found_weather = true;
    if (call.tool_name == "calculator")
      found_calculator = true;
  }

  // At least one tool should be called, preferably both
  EXPECT_TRUE(found_weather || found_calculator);
}

// Test tool choice control - force specific tool
TEST_P(ToolCallingIntegrationTest, ForcedToolChoice) {
  if (!use_real_api_) {
    GTEST_SKIP() << "No API key set for " << GetParam();
  }

  GenerateOptions options(model_, "Calculate something for me");
  options.tools = tools_;
  options.tool_choice = ToolChoice::specific("calculator");
  options.max_tokens = 200;

  auto result = client_->generate_text(options);

  // When forcing tool use, the result should be successful but may have empty
  // text
  EXPECT_TRUE(result.is_success())
      << "Expected successful result but got error: " << result.error_message();
  EXPECT_FALSE(result.error.has_value())
      << "Expected no error in successful result";

  // Should have made exactly one tool call to calculator
  EXPECT_TRUE(result.has_tool_calls());
  EXPECT_EQ(result.tool_calls.size(), 1);
  EXPECT_EQ(result.tool_calls[0].tool_name, "calculator");
}

// Test tool choice - auto mode
TEST_P(ToolCallingIntegrationTest, AutoToolChoice) {
  if (!use_real_api_) {
    GTEST_SKIP() << "No API key set for " << GetParam();
  }

  GenerateOptions options(model_, "What's 10 + 5?");
  options.tools = tools_;
  options.tool_choice = ToolChoice::auto_choice();
  options.max_tokens = 200;

  auto result = client_->generate_text(options);

  // With auto tool choice, result should be successful
  EXPECT_TRUE(result.is_success())
      << "Expected successful result but got error: " << result.error_message();
  EXPECT_FALSE(result.error.has_value())
      << "Expected no error in successful result";

  // Model may or may not include text when using tools
  // Text content is optional when tool calls are made

  // Model should decide whether to use tools (likely will for math)
  // If tools are used, should be calculator
  if (result.has_tool_calls()) {
    bool found_calculator = false;
    for (const auto& call : result.tool_calls) {
      if (call.tool_name == "calculator")
        found_calculator = true;
    }
    EXPECT_TRUE(found_calculator);
  }
}

// Test tool choice - none (disable tools)
TEST_P(ToolCallingIntegrationTest, NoToolChoice) {
  if (!use_real_api_) {
    GTEST_SKIP() << "No API key set for " << GetParam();
  }

  GenerateOptions options(model_, "What's the weather like?");
  options.tools = tools_;
  options.tool_choice = ToolChoice::none();
  options.max_tokens = 200;

  auto result = client_->generate_text(options);

  TestAssertions::assertSuccess(result);
  EXPECT_FALSE(result.text.empty());

  // Should not have made any tool calls
  EXPECT_FALSE(result.has_tool_calls());
}

// Test tool choice - required (any tool)
TEST_P(ToolCallingIntegrationTest, RequiredToolChoice) {
  if (!use_real_api_) {
    GTEST_SKIP() << "No API key set for " << GetParam();
  }

  GenerateOptions options(model_, "Help me with something");
  options.tools = tools_;
  options.tool_choice = ToolChoice::required();
  options.max_tokens = 200;

  auto result = client_->generate_text(options);

  // When requiring tool use, the result should be successful but may have empty
  // text
  EXPECT_TRUE(result.is_success())
      << "Expected successful result but got error: " << result.error_message();
  EXPECT_FALSE(result.error.has_value())
      << "Expected no error in successful result";

  // Should have made at least one tool call
  EXPECT_TRUE(result.has_tool_calls());
  EXPECT_GT(result.tool_calls.size(), 0);
}

// Test complex tool parameters
TEST_P(ToolCallingIntegrationTest, ComplexToolParameters) {
  if (!use_real_api_) {
    GTEST_SKIP() << "No API key set for " << GetParam();
  }

  GenerateOptions options(model_, "Calculate 25.5 multiplied by 3.7");
  options.tools = tools_;
  options.max_tokens = 200;

  auto result = client_->generate_text(options);

  // Result should be successful, text is optional when using tools
  EXPECT_TRUE(result.is_success())
      << "Expected successful result but got error: " << result.error_message();
  EXPECT_FALSE(result.error.has_value())
      << "Expected no error in successful result";

  // Should use calculator with floating point numbers
  EXPECT_TRUE(result.has_tool_calls());

  auto calc_call = std::find_if(
      result.tool_calls.begin(), result.tool_calls.end(),
      [](const ToolCall& call) { return call.tool_name == "calculator"; });

  if (calc_call != result.tool_calls.end()) {
    EXPECT_TRUE(calc_call->arguments.contains("operation"));
    EXPECT_TRUE(calc_call->arguments.contains("a"));
    EXPECT_TRUE(calc_call->arguments.contains("b"));
    EXPECT_EQ(calc_call->arguments["operation"].get<std::string>(), "multiply");
  }
}

// Test invalid tool schema validation
TEST_P(ToolCallingIntegrationTest, InvalidToolArguments) {
  if (!use_real_api_) {
    GTEST_SKIP() << "No API key set for " << GetParam();
  }

  // This test relies on the model providing arguments that match the schema.
  // Whether those arguments would satisfy the caller is not the SDK's call.
  GenerateOptions options(model_,
                          "Calculate the result without providing numbers");
  options.tools = tools_;
  options.tool_choice = ToolChoice::specific("calculator");
  options.max_tokens = 200;

  auto result = client_->generate_text(options);

  // Result should be successful (API call worked), text might be empty when
  // forcing tool
  EXPECT_TRUE(result.is_success())
      << "Expected successful result but got error: " << result.error_message();
  EXPECT_FALSE(result.error.has_value())
      << "Expected no error in successful result";

  if (result.has_tool_calls()) {
    EXPECT_EQ(result.tool_calls[0].tool_name, "calculator");
  }
}

// Test concurrent requests
TEST_P(ToolCallingIntegrationTest, ConcurrentToolCalls) {
  if (!use_real_api_) {
    GTEST_SKIP() << "No API key set for " << GetParam();
  }

  const int num_requests = 2;
  std::vector<std::future<GenerateResult>> futures;
  futures.reserve(num_requests);

  for (int i = 0; i < num_requests; ++i) {
    futures.push_back(std::async(std::launch::async, [this, i]() {
      GenerateOptions options(model_, "What's " + std::to_string(10 + i * 5) +
                                          " + " + std::to_string(20 + i * 3) +
                                          "?");
      options.tools = tools_;
      options.max_tokens = 200;
      return client_->generate_text(options);
    }));
  }

  // Wait for all requests to complete
  for (auto& future : futures) {
    auto result = future.get();
    // Result should be successful, text might be empty if only tool calls are
    // returned
    EXPECT_TRUE(result.is_success())
        << "Expected successful result but got error: "
        << result.error_message();
    EXPECT_FALSE(result.error.has_value())
        << "Expected no error in successful result";
  }
}

// Test tool schema validation
TEST_P(ToolCallingIntegrationTest, ToolSchemaValidation) {
  if (!use_real_api_) {
    GTEST_SKIP() << "No API key set for " << GetParam();
  }

  // Test with a tool that has strict schema requirements
  auto strict_tool = Tool(
      "Tool with strict schema validation",
      JsonValue{
          {"type", "object"},
          {"properties",
           {{"required_string", {{"type", "string"}, {"minLength", 1}}},
            {"required_number", {{"type", "number"}, {"minimum", 0}}},
            {"enum_value",
             {{"type", "string"},
              {"enum", {"option1", "option2", "option3"}}}}}},
          {"required", {"required_string", "required_number", "enum_value"}},
          {"additionalProperties", false}});

  ToolSet strict_tools = {{"strict_tool", strict_tool}};

  GenerateOptions options(model_,
                          "Use the strict tool with required_string='test', "
                          "required_number=42, and enum_value='option1'");
  options.tools = strict_tools;
  options.tool_choice = ToolChoice::specific("strict_tool");
  options.max_tokens = 200;

  auto result = client_->generate_text(options);

  // When forcing specific tool, text might be empty
  EXPECT_TRUE(result.is_success())
      << "Expected successful result but got error: " << result.error_message();
  EXPECT_FALSE(result.error.has_value())
      << "Expected no error in successful result";
  EXPECT_TRUE(result.has_tool_calls());

  if (result.has_tool_calls()) {
    const auto& call = result.tool_calls[0];
    EXPECT_EQ(call.tool_name, "strict_tool");

    // Verify the model provided arguments that match the schema
    if (call.arguments.contains("required_string") &&
        call.arguments.contains("required_number") &&
        call.arguments.contains("enum_value")) {
      EXPECT_TRUE(call.arguments["required_string"].is_string());
      EXPECT_TRUE(call.arguments["required_number"].is_number());
      EXPECT_TRUE(call.arguments["enum_value"].is_string());

      std::string enum_val = call.arguments["enum_value"].get<std::string>();
      EXPECT_THAT(enum_val, testing::AnyOf("option1", "option2", "option3"));
    }
  }
}

// Instantiate tests for the OpenAI-compatible provider
INSTANTIATE_TEST_SUITE_P(
    ProviderTests,
    ToolCallingIntegrationTest,
    ::testing::Values("openai"),
    [](const ::testing::TestParamInfo<ToolCallingIntegrationTest::ParamType>&
           info) { return info.param; });

// Provider-specific tests

class OpenAISpecificToolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* api_key = std::getenv("OPENAI_API_KEY");
    if (api_key && *api_key != '\0') {
      use_real_api_ = true;
      client_ = ai::openai::create_client(api_key);
      model_ = ai::openai::models::kGpt56Luna;
    } else {
      use_real_api_ = false;
    }

    tools_ = ToolTestFixtures::createTestToolSet();
  }

  bool use_real_api_ = false;
  std::optional<ai::Client> client_;
  std::string model_;
  ToolSet tools_;
};

// Test OpenAI-specific tool choice "none"
TEST_F(OpenAISpecificToolTest, ToolChoiceNoneSupport) {
  if (!use_real_api_) {
    GTEST_SKIP() << "No OPENAI_API_KEY environment variable set";
  }

  GenerateOptions options(model_, "Tell me about the weather");
  options.tools = tools_;
  options.tool_choice = ToolChoice::none();
  options.max_tokens = 150;

  auto result = client_->generate_text(options);

  TestAssertions::assertSuccess(result);
  EXPECT_FALSE(result.text.empty());
  EXPECT_FALSE(result.has_tool_calls());
}

}  // namespace test
}  // namespace ai
