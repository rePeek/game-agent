#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <llm/core/message.h>
#include <llm/core/provider.h>
#include <llm/core/request.h>
#include <llm/core/result.h>
#include <llm/core/types.h>

namespace llm {
namespace {

// A realistic decision batch: two calls, in the order the model produced them.
std::vector<CommandCall> MakeBatch() {
  return {
      CommandCall{"call_001", "set_pressing_intensity", R"({"level":3})"},
      CommandCall{"call_002", "set_defensive_line", R"({"height":"high"})"},
  };
}

TEST(LlmCoreMessageTest, FactoriesSetRoleAndContent) {
  const Message system = Message::System("You are a coach.");
  EXPECT_EQ(system.role, Role::System);
  EXPECT_EQ(system.content, "You are a coach.");
  EXPECT_FALSE(system.HasCommandCalls());

  const Message user = Message::User("Opponent is pressing high.");
  EXPECT_EQ(user.role, Role::User);

  const Message assistant = Message::Assistant("Raising pressing.");
  EXPECT_EQ(assistant.role, Role::Assistant);

  EXPECT_EQ(ToString(system.role), "system");
  EXPECT_EQ(ToString(user.role), "user");
  EXPECT_EQ(ToString(assistant.role), "assistant");
  EXPECT_EQ(ToString(Message::CommandResultMessage(CommandResult{}).role), "tool");
}

TEST(LlmCoreMessageTest, AssistantCommandCallsKeepsBatchAndOrder) {
  Message message = Message::AssistantCommandCalls(MakeBatch());

  EXPECT_EQ(message.role, Role::Assistant);
  EXPECT_TRUE(message.content.empty());
  EXPECT_TRUE(message.HasCommandCalls());
  ASSERT_EQ(message.command_calls.size(), 2u);
  EXPECT_EQ(message.command_calls[0].id, "call_001");
  EXPECT_EQ(message.command_calls[0].name, "set_pressing_intensity");
  EXPECT_EQ(message.command_calls[1].id, "call_002");
  EXPECT_EQ(message.command_calls[1].name, "set_defensive_line");
}

TEST(LlmCoreMessageTest, CommandResultCarriesCallIdAndPayload) {
  CommandResult result;
  result.command_call_id = "call_001";
  result.name = "set_pressing_intensity";
  result.content_json = R"({"applied":true,"intensity":3})";

  const Message message = Message::CommandResultMessage(result);

  EXPECT_EQ(message.role, Role::Tool);
  EXPECT_EQ(message.command_call_id, "call_001");
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
      Message::AssistantCommandCalls(batch),
  };
  for (const auto& call : batch) {
    CommandResult result;
    result.command_call_id = call.id;
    result.name = call.name;
    result.content_json = R"({"applied":true})";
    context.push_back(Message::CommandResultMessage(result));
  }
  context.push_back(Message::User("They switched to a low block."));

  ASSERT_EQ(context.size(), 6u);
  EXPECT_EQ(context[2].role, Role::Assistant);
  EXPECT_EQ(context[2].command_calls.size(), batch.size());
  EXPECT_EQ(context[3].role, Role::Tool);
  EXPECT_EQ(context[3].command_call_id, batch[0].id);
  EXPECT_EQ(context[4].role, Role::Tool);
  EXPECT_EQ(context[4].command_call_id, batch[1].id);
  EXPECT_EQ(context[5].role, Role::User);
}

TEST(LlmCoreCommandCallTest, KeepsRawArgumentTextVerbatim) {
  // Formatting, key order and numbers must survive the core untouched: the
  // provider layer is the only place allowed to interpret them.
  const std::string raw = R"({ "level" : 3, "reason": "trailing" })";
  const CommandCall call{"call_001", "set_pressing_intensity", raw};

  EXPECT_EQ(call.arguments_json, raw);
  EXPECT_TRUE(call.IsValid());
  EXPECT_EQ(
      call.ToString(),
      "CommandCall{id='call_001', name='set_pressing_intensity', arguments=" +
          raw + "}");
}

TEST(LlmCoreCommandCallTest, EmptyNameIsInvalid) {
  EXPECT_FALSE(CommandCall{}.IsValid());
  EXPECT_FALSE((CommandCall{"call_001", "", "{}"}.IsValid()));
  EXPECT_TRUE((CommandCall{"call_001", "tool", "{}"}.IsValid()));
}

TEST(LlmCoreRequestTest, DefaultsToRequiredCommandChoice) {
  ModelRequest request;
  request.model = "test-model";

  EXPECT_EQ(request.command_choice.mode, CommandChoiceMode::Required);
  EXPECT_EQ(request.command_choice.ToString(), "required");
  EXPECT_FALSE(request.HasCommands());
  EXPECT_EQ(request.max_command_calls, std::nullopt);
}

TEST(LlmCoreRequestTest, CommandChoiceVariants) {
  EXPECT_EQ(CommandChoice::Auto().ToString(), "auto");
  EXPECT_EQ(CommandChoice::None().ToString(), "none");
  EXPECT_EQ(CommandChoice::Specific("set_pressing_intensity").ToString(),
            "set_pressing_intensity");
  EXPECT_EQ(CommandChoice::Specific("x").mode, CommandChoiceMode::Specific);
}

TEST(LlmCoreRequestTest, FindsToolByName) {
  ModelRequest request;
  request.commands.push_back(CommandDefinition{"set_pressing_intensity",
                                         "Pressing intensity",
                                         R"({"type":"object"})", true});
  request.commands.push_back(CommandDefinition{"keep_current_tactics", "No change",
                                         R"({"type":"object"})", false});

  ASSERT_TRUE(request.HasCommands());
  const CommandDefinition* found = request.FindCommand("keep_current_tactics");
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->description, "No change");
  EXPECT_FALSE(found->strict);
  EXPECT_EQ(request.FindCommand("missing"), nullptr);
}

TEST(LlmCoreResultTest, SuccessOnlyWhenStatusIsSuccess) {
  CompletionResult result;
  EXPECT_FALSE(result.IsSuccess());
  EXPECT_FALSE(result.HasCommandCalls());
  EXPECT_EQ(result.status, CompletionStatus::ProviderError);

  result.status = CompletionStatus::Success;
  result.command_calls = MakeBatch();
  EXPECT_TRUE(result.IsSuccess());
  EXPECT_TRUE(static_cast<bool>(result));
  EXPECT_TRUE(result.HasCommandCalls());

  result.status = CompletionStatus::InvalidCommandCalls;
  EXPECT_FALSE(result.IsSuccess());
  EXPECT_TRUE(result.HasCommandCalls()) << "invalid calls are still reported";
}

TEST(LlmCoreResultTest, ErrorMessageFallsBackToStatusText) {
  CompletionResult result;
  result.status = CompletionStatus::NoCommandCalls;
  EXPECT_EQ(result.ErrorMessage(), "no command calls");

  ModelError error;
  error.kind = ErrorKind::RateLimit;
  error.message = "429 from provider";
  error.http_status = 429;
  error.retryable = true;
  result.error = error;

  EXPECT_EQ(result.ErrorMessage(), "429 from provider");
  EXPECT_EQ(ToString(error.kind), "rate limit");
  EXPECT_EQ(result.status, CompletionStatus::NoCommandCalls)
      << "the status stays the caller's branch condition";
}

TEST(LlmCoreResultTest, StatusAndFinishReasonSpellings) {
  EXPECT_EQ(ToString(CompletionStatus::Success), "success");
  EXPECT_EQ(ToString(CompletionStatus::Truncated), "truncated");
  EXPECT_EQ(ToString(CompletionStatus::Unsupported), "unsupported");
  EXPECT_EQ(ToString(FinishReason::CommandCalls), "tool_calls");
  EXPECT_EQ(ToString(FinishReason::Length), "length");
  EXPECT_EQ(ToString(FinishReason::Unknown), "unknown");
}

TEST(LlmCoreResultTest, ToAssistantMessageKeepsBatch) {
  CompletionResult result;
  result.status = CompletionStatus::Success;
  result.assistant_text = "Raising pressing.";
  result.command_calls = MakeBatch();

  const Message message = result.ToAssistantMessage();

  EXPECT_EQ(message.role, Role::Assistant);
  EXPECT_EQ(message.content, "Raising pressing.");
  ASSERT_EQ(message.command_calls.size(), 2u);
  EXPECT_EQ(message.command_calls[1].arguments_json, R"({"height":"high"})");
}

// A provider is only a request/response pair: nothing in the core forces an
// implementation to execute tools or advance a conversation.
class ScriptedProvider : public IModelProvider {
 public:
  std::string Name() const override { return "scripted"; }

  ProviderCapabilities Capabilities() const override {
    ProviderCapabilities capabilities;
    capabilities.forced_command_choice = true;
    return capabilities;
  }

  CompletionResult Complete(const ModelRequest& request) override {
    last_request = request;
    CompletionResult result;
    result.status = CompletionStatus::Success;
    result.command_calls = MakeBatch();
    return result;
  }

  ModelRequest last_request;
};

TEST(LlmCoreProviderTest, CompleteIsTheWholeSurface) {
  ScriptedProvider provider;

  ModelRequest request;
  request.model = "test-model";
  request.commands.push_back(CommandDefinition{"set_pressing_intensity",
                                         "Pressing intensity",
                                         R"({"type":"object"})", true});

  const CompletionResult result = provider.Complete(request);

  EXPECT_EQ(provider.Name(), "scripted");
  EXPECT_TRUE(provider.Capabilities().forced_command_choice);
  EXPECT_FALSE(provider.Capabilities().strict_command_schema);
  EXPECT_EQ(provider.last_request.model, "test-model");
  EXPECT_TRUE(result.IsSuccess());
  ASSERT_EQ(result.command_calls.size(), 2u);
}

}  // namespace
}  // namespace llm
