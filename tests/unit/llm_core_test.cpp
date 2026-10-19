#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <llm/core.h>

namespace llm {
namespace {

// A realistic decision batch: two calls, in the order the model produced them.
std::vector<ToolCall> MakeBatch() {
  return {
      ToolCall{"call_001", "set_pressing_intensity", R"({"level":3})"},
      ToolCall{"call_002", "set_defensive_line", R"({"height":"high"})"},
  };
}

TEST(LlmCoreMessageTest, FactoriesSetRoleAndContent) {
  const Message system = Message::System("You are a coach.");
  EXPECT_EQ(system.role, Role::System);
  EXPECT_EQ(system.content, "You are a coach.");
  EXPECT_FALSE(system.HasToolCalls());

  const Message user = Message::User("Opponent is pressing high.");
  EXPECT_EQ(user.role, Role::User);

  const Message assistant = Message::Assistant("Raising pressing.");
  EXPECT_EQ(assistant.role, Role::Assistant);

  EXPECT_EQ(ToString(system.role), "system");
  EXPECT_EQ(ToString(user.role), "user");
  EXPECT_EQ(ToString(assistant.role), "assistant");
  EXPECT_EQ(ToString(Message::ToolResult(ToolResult{}).role), "tool");
}

TEST(LlmCoreMessageTest, AssistantToolCallsKeepsBatchAndOrder) {
  Message message = Message::AssistantToolCalls(MakeBatch());

  EXPECT_EQ(message.role, Role::Assistant);
  EXPECT_TRUE(message.content.empty());
  EXPECT_TRUE(message.HasToolCalls());
  ASSERT_EQ(message.tool_calls.size(), 2u);
  EXPECT_EQ(message.tool_calls[0].id, "call_001");
  EXPECT_EQ(message.tool_calls[0].name, "set_pressing_intensity");
  EXPECT_EQ(message.tool_calls[1].id, "call_002");
  EXPECT_EQ(message.tool_calls[1].name, "set_defensive_line");
}

TEST(LlmCoreMessageTest, ToolResultCarriesCallIdAndPayload) {
  ToolResult result;
  result.tool_call_id = "call_001";
  result.name = "set_pressing_intensity";
  result.content_json = R"({"applied":true,"intensity":3})";

  const Message message = Message::ToolResult(result);

  EXPECT_EQ(message.role, Role::Tool);
  EXPECT_EQ(message.tool_call_id, "call_001");
  EXPECT_EQ(message.name, "set_pressing_intensity");
  EXPECT_EQ(message.content, R"({"applied":true,"intensity":3})");
}

// The context protocol must be able to express a previous decision: the
// assistant turn with its tool calls followed by the matching tool turns.
TEST(LlmCoreMessageTest, ExpressesPreviousDecisionAsContext) {
  const auto batch = MakeBatch();

  std::vector<Message> context = {
      Message::System("You are a coach."),
      Message::User("Opponent is pressing high."),
      Message::AssistantToolCalls(batch),
  };
  for (const auto& call : batch) {
    ToolResult result;
    result.tool_call_id = call.id;
    result.name = call.name;
    result.content_json = R"({"applied":true})";
    context.push_back(Message::ToolResult(result));
  }
  context.push_back(Message::User("They switched to a low block."));

  ASSERT_EQ(context.size(), 6u);
  EXPECT_EQ(context[2].role, Role::Assistant);
  EXPECT_EQ(context[2].tool_calls.size(), batch.size());
  EXPECT_EQ(context[3].role, Role::Tool);
  EXPECT_EQ(context[3].tool_call_id, batch[0].id);
  EXPECT_EQ(context[4].role, Role::Tool);
  EXPECT_EQ(context[4].tool_call_id, batch[1].id);
  EXPECT_EQ(context[5].role, Role::User);
}

TEST(LlmCoreToolCallTest, KeepsRawArgumentTextVerbatim) {
  // Formatting, key order and numbers must survive the core untouched: the
  // provider layer is the only place allowed to interpret them.
  const std::string raw = R"({ "level" : 3, "reason": "trailing" })";
  const ToolCall call{"call_001", "set_pressing_intensity", raw};

  EXPECT_EQ(call.arguments_json, raw);
  EXPECT_TRUE(call.IsValid());
  EXPECT_EQ(
      call.ToString(),
      "ToolCall{id='call_001', name='set_pressing_intensity', arguments=" +
          raw + "}");
}

TEST(LlmCoreToolCallTest, EmptyNameIsInvalid) {
  EXPECT_FALSE(ToolCall{}.IsValid());
  EXPECT_FALSE((ToolCall{"call_001", "", "{}"}.IsValid()));
  EXPECT_TRUE((ToolCall{"call_001", "tool", "{}"}.IsValid()));
}

TEST(LlmCoreRequestTest, DefaultsToRequiredToolChoice) {
  ModelRequest request;
  request.model = "test-model";

  EXPECT_EQ(request.tool_choice.mode, ToolChoiceMode::Required);
  EXPECT_EQ(request.tool_choice.ToString(), "required");
  EXPECT_FALSE(request.HasTools());
  EXPECT_EQ(request.max_tool_calls, std::nullopt);
}

TEST(LlmCoreRequestTest, ToolChoiceVariants) {
  EXPECT_EQ(ToolChoice::Auto().ToString(), "auto");
  EXPECT_EQ(ToolChoice::None().ToString(), "none");
  EXPECT_EQ(ToolChoice::Specific("set_pressing_intensity").ToString(),
            "set_pressing_intensity");
  EXPECT_EQ(ToolChoice::Specific("x").mode, ToolChoiceMode::Specific);
}

TEST(LlmCoreRequestTest, FindsToolByName) {
  ModelRequest request;
  request.tools.push_back(ToolDefinition{"set_pressing_intensity",
                                         "Pressing intensity",
                                         R"({"type":"object"})", true});
  request.tools.push_back(ToolDefinition{"keep_current_tactics", "No change",
                                         R"({"type":"object"})", false});

  ASSERT_TRUE(request.HasTools());
  const ToolDefinition* found = request.FindTool("keep_current_tactics");
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->description, "No change");
  EXPECT_FALSE(found->strict);
  EXPECT_EQ(request.FindTool("missing"), nullptr);
}

TEST(LlmCoreResultTest, SuccessOnlyWhenStatusIsSuccess) {
  CompletionResult result;
  EXPECT_FALSE(result.IsSuccess());
  EXPECT_FALSE(result.HasToolCalls());
  EXPECT_EQ(result.status, CompletionStatus::ProviderError);

  result.status = CompletionStatus::Success;
  result.tool_calls = MakeBatch();
  EXPECT_TRUE(result.IsSuccess());
  EXPECT_TRUE(static_cast<bool>(result));
  EXPECT_TRUE(result.HasToolCalls());

  result.status = CompletionStatus::InvalidToolCalls;
  EXPECT_FALSE(result.IsSuccess());
  EXPECT_TRUE(result.HasToolCalls()) << "invalid calls are still reported";
}

TEST(LlmCoreResultTest, ErrorMessageFallsBackToStatusText) {
  CompletionResult result;
  result.status = CompletionStatus::NoToolCalls;
  EXPECT_EQ(result.ErrorMessage(), "no tool calls");

  ModelError error;
  error.kind = ErrorKind::RateLimit;
  error.message = "429 from provider";
  error.http_status = 429;
  error.retryable = true;
  result.error = error;

  EXPECT_EQ(result.ErrorMessage(), "429 from provider");
  EXPECT_EQ(ToString(error.kind), "rate limit");
  EXPECT_EQ(result.status, CompletionStatus::NoToolCalls)
      << "the status stays the caller's branch condition";
}

TEST(LlmCoreResultTest, StatusAndFinishReasonSpellings) {
  EXPECT_EQ(ToString(CompletionStatus::Success), "success");
  EXPECT_EQ(ToString(CompletionStatus::Truncated), "truncated");
  EXPECT_EQ(ToString(CompletionStatus::Unsupported), "unsupported");
  EXPECT_EQ(ToString(FinishReason::ToolCalls), "tool_calls");
  EXPECT_EQ(ToString(FinishReason::Length), "length");
  EXPECT_EQ(ToString(FinishReason::Unknown), "unknown");
}

TEST(LlmCoreResultTest, ToAssistantMessageKeepsBatch) {
  CompletionResult result;
  result.status = CompletionStatus::Success;
  result.assistant_text = "Raising pressing.";
  result.tool_calls = MakeBatch();

  const Message message = result.ToAssistantMessage();

  EXPECT_EQ(message.role, Role::Assistant);
  EXPECT_EQ(message.content, "Raising pressing.");
  ASSERT_EQ(message.tool_calls.size(), 2u);
  EXPECT_EQ(message.tool_calls[1].arguments_json, R"({"height":"high"})");
}

// A provider is only a request/response pair: nothing in the core forces an
// implementation to execute tools or advance a conversation.
class ScriptedProvider : public IModelProvider {
 public:
  std::string Name() const override { return "scripted"; }

  ProviderCapabilities Capabilities() const override {
    ProviderCapabilities capabilities;
    capabilities.forced_tool_choice = true;
    return capabilities;
  }

  CompletionResult Complete(const ModelRequest& request) override {
    last_request = request;
    CompletionResult result;
    result.status = CompletionStatus::Success;
    result.tool_calls = MakeBatch();
    return result;
  }

  ModelRequest last_request;
};

TEST(LlmCoreProviderTest, CompleteIsTheWholeSurface) {
  ScriptedProvider provider;

  ModelRequest request;
  request.model = "test-model";
  request.tools.push_back(ToolDefinition{"set_pressing_intensity",
                                         "Pressing intensity",
                                         R"({"type":"object"})", true});

  const CompletionResult result = provider.Complete(request);

  EXPECT_EQ(provider.Name(), "scripted");
  EXPECT_TRUE(provider.Capabilities().forced_tool_choice);
  EXPECT_FALSE(provider.Capabilities().strict_tool_schema);
  EXPECT_EQ(provider.last_request.model, "test-model");
  EXPECT_TRUE(result.IsSuccess());
  ASSERT_EQ(result.tool_calls.size(), 2u);
}

}  // namespace
}  // namespace llm
