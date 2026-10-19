// Unit tests for the context/message-history layer.

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <llm/context/context.h>
#include <llm/core/message.h>
#include <llm/core/provider.h>
#include <llm/core/request.h>
#include <llm/core/result.h>
#include <llm/core/types.h>

namespace llm {
namespace {

ContextUpdate Update(std::string context,
                     std::string history = {},
                     std::string revision = "rev-1") {
  ContextUpdate update;
  update.current_context = std::move(context);
  update.recent_history = std::move(history);
  update.context_revision = std::move(revision);
  return update;
}

ContextUpdate EmptyUpdate() {
  ContextUpdate update;
  update.current_context = "Minute 62.";
  update.context_revision = "rev-1";
  return update;
}

struct DecisionSpec {
  std::uint64_t id = 1;
  std::string command_name = "set_pressing_intensity";
  std::string arguments = R"({"level":3})";
  std::string assistant_text;
};

DecisionRecord MakeDecision(const DecisionSpec& spec) {
  CommandCall call{"call_" + std::to_string(spec.id), spec.command_name,
                spec.arguments};
  DecisionRecord record;
  record.decision_id = spec.id;
  record.context_revision = "rev-" + std::to_string(spec.id);
  record.assistant = Message::AssistantCommandCalls({call});
  if (!spec.assistant_text.empty()) {
    record.assistant.content = spec.assistant_text;
  }
  CommandResult applied;
  applied.command_call_id = call.id;
  applied.name = call.name;
  applied.outcome.success = true; applied.outcome.content_json = R"({"applied":true})";
  record.command_results = {Message::CommandResultMessage(applied)};
  return record;
}

std::vector<std::string> ContentsOf(const std::vector<Message>& messages) {
  std::vector<std::string> contents;
  for (const Message& message : messages) {
    contents.push_back(message.content);
  }
  return contents;
}

TEST(LlmContextTest, CurrentContextReplacesTheSnapshot) {
  Context context;

  context.SetContextUpdate(
      Update("Minute 10, 0-0.", "no pressure yet", "rev-1"));
  ASSERT_EQ(context.context_revision(), "rev-1");

  context.SetContextUpdate(
      Update("Minute 62, 1-1.", "they press high", "rev-2"));

  EXPECT_EQ(context.current_context(), "Minute 62, 1-1.");
  EXPECT_EQ(context.recent_history(), "they press high");
  EXPECT_EQ(context.context_revision(), "rev-2");

  const RenderedContext rendered = context.Render();
  // Recent history then the current snapshot: the context owns neither the
  // system prompt nor the tools.
  ASSERT_EQ(rendered.messages.size(), 2u);
  EXPECT_EQ(rendered.messages[0].role, Role::User);
  // The old snapshot is gone, not accumulated.
  for (const std::string& content : ContentsOf(rendered.messages)) {
    EXPECT_EQ(content.find("Minute 10"), std::string::npos) << content;
  }
  EXPECT_NE(rendered.messages[0].content.find("they press high"),
            std::string::npos);
  EXPECT_NE(rendered.messages[1].content.find("Minute 62"), std::string::npos);
}

TEST(LlmContextTest, EmptyBlocksAreSkipped) {
  Context context;
  context.SetContextUpdate(Update("Minute 1."));

  const RenderedContext rendered = context.Render();

  ASSERT_EQ(rendered.messages.size(), 1u);
  EXPECT_EQ(rendered.messages[0].role, Role::User);
  EXPECT_NE(rendered.messages[0].content.find("Minute 1."), std::string::npos);

  context.SetContextUpdate(Update("Minute 2.", "some history"));
  const RenderedContext with_history = context.Render();
  ASSERT_EQ(with_history.messages.size(), 2u);
  EXPECT_EQ(with_history.messages[0].role, Role::User);
  EXPECT_NE(with_history.messages[0].content.find("Recent history"),
            std::string::npos);
  EXPECT_NE(with_history.messages[1].content.find("Minute 2."),
            std::string::npos);
}

TEST(LlmContextTest, DecisionsKeepCommandCallsAndResultsTogether) {
  Context context;
  context.SetContextUpdate(EmptyUpdate());
  context.AppendDecision(MakeDecision(DecisionSpec{1}));
  context.AppendDecision(MakeDecision(
      DecisionSpec{2, "set_defensive_line", R"({"height":"low"})"}));

  const RenderedContext rendered = context.Render();

  // system prompt is empty here, so: decision 1 (assistant + tool),
  // decision 2 (assistant + tool), current context.
  ASSERT_EQ(rendered.messages.size(), 5u);
  EXPECT_EQ(rendered.decisions_included, 2u);
  EXPECT_EQ(rendered.decisions_dropped, 0u);

  EXPECT_EQ(rendered.messages[0].role, Role::Assistant);
  ASSERT_EQ(rendered.messages[0].command_calls.size(), 1u);
  EXPECT_EQ(rendered.messages[0].command_calls[0].name, "set_pressing_intensity");
  EXPECT_EQ(rendered.messages[1].role, Role::Tool);
  EXPECT_EQ(rendered.messages[1].command_call_id, "call_1");

  EXPECT_EQ(rendered.messages[2].role, Role::Assistant);
  EXPECT_EQ(rendered.messages[2].command_calls[0].name, "set_defensive_line");
  EXPECT_EQ(rendered.messages[3].role, Role::Tool);
  EXPECT_EQ(rendered.messages[3].command_call_id, "call_2");

  EXPECT_EQ(rendered.messages[4].role, Role::User);
  EXPECT_NE(rendered.messages[4].content.find("Current context"),
            std::string::npos);
}

TEST(LlmContextTest, MultiToolDecisionsStayContiguous) {
  CommandCall first{"call_a", "set_pressing_intensity", R"({"level":4})"};
  CommandCall second{"call_b", "set_defensive_line", R"({"height":"high"})"};

  DecisionRecord record;
  record.decision_id = 7;
  record.context_revision = "rev-7";
  record.assistant = Message::AssistantCommandCalls({first, second});
  for (const CommandCall& call : {first, second}) {
    CommandResult result;
    result.command_call_id = call.id;
    result.name = call.name;
    result.outcome.success = true; result.outcome.content_json = R"({"applied":true})";
    record.command_results.push_back(Message::CommandResultMessage(result));
  }
  ASSERT_EQ(record.MessageCount(), 3u);

  Context context;
  context.SetContextUpdate(EmptyUpdate());
  context.AppendDecision(std::move(record));

  const RenderedContext rendered = context.Render();
  // assistant (2 tool calls) + 2 tool turns + the current context
  ASSERT_EQ(rendered.messages.size(), 4u);
  EXPECT_EQ(rendered.messages[0].role, Role::Assistant);
  EXPECT_EQ(rendered.messages[0].command_calls.size(), 2u);
  EXPECT_EQ(rendered.messages[1].role, Role::Tool);
  EXPECT_EQ(rendered.messages[1].command_call_id, "call_a");
  EXPECT_EQ(rendered.messages[2].role, Role::Tool);
  EXPECT_EQ(rendered.messages[2].command_call_id, "call_b");
  EXPECT_EQ(rendered.messages[3].role, Role::User);
}

TEST(LlmContextTest, BudgetDropsWholeDecisionsOldestFirst) {
  ContextConfig config;
  // The context uses 1 of the 6, leaving room for two 2-message records.
  config.max_messages = 6;
  Context context(config);
  context.SetContextUpdate(EmptyUpdate());
  for (std::uint64_t id = 1; id <= 5; ++id) {
    context.AppendDecision(MakeDecision(DecisionSpec{id}));
  }

  const RenderedContext rendered = context.Render();

  EXPECT_EQ(rendered.decisions_included, 2u);
  EXPECT_EQ(rendered.decisions_dropped, 3u);
  ASSERT_EQ(rendered.messages.size(), 5u);  // 2*2 + 1
  // The two newest survive, the older three are gone as complete units.
  EXPECT_EQ(rendered.messages[0].command_calls[0].id, "call_4");
  EXPECT_EQ(rendered.messages[2].command_calls[0].id, "call_5");
  EXPECT_EQ(rendered.messages[4].role, Role::User);

  // The stored history is untouched: trimming is a rendering decision.
  EXPECT_EQ(context.decision_count(), 5u);
}

TEST(LlmContextTest, MaxDecisionsCapsWhatIsKept) {
  ContextConfig config;
  config.max_messages = 1000;
  config.max_decisions = 2;
  Context context(config);
  context.SetContextUpdate(EmptyUpdate());
  for (std::uint64_t id = 1; id <= 4; ++id) {
    context.AppendDecision(MakeDecision(DecisionSpec{id}));
  }

  const RenderedContext rendered = context.Render();

  EXPECT_EQ(rendered.decisions_included, 2u);
  EXPECT_EQ(rendered.decisions_dropped, 2u);
  ASSERT_EQ(rendered.messages.size(), 5u);
  EXPECT_EQ(rendered.messages[0].command_calls[0].id, "call_3");
  EXPECT_EQ(rendered.messages[2].command_calls[0].id, "call_4");
}

TEST(LlmContextTest, EssentialBlocksAreSentEvenWithoutBudget) {
  ContextConfig config;
  config.max_messages = 2;  // history + context already use both
  Context context(config);
  context.SetContextUpdate(Update("Minute 90.", "tense"));
  context.AppendDecision(MakeDecision(DecisionSpec{1}));

  const RenderedContext rendered = context.Render();

  EXPECT_EQ(rendered.decisions_included, 0u);
  EXPECT_EQ(rendered.decisions_dropped, 1u);
  ASSERT_EQ(rendered.messages.size(), 2u);
  EXPECT_NE(rendered.messages[0].content.find("Recent history"),
            std::string::npos);
  EXPECT_NE(rendered.messages[1].content.find("Current context"),
            std::string::npos);
}

TEST(LlmContextTest, RenderIsTheWholeContextApi) {
  Context context;
  context.SetContextUpdate(Update("Minute 20.", "quiet"));
  context.AppendDecision(MakeDecision(DecisionSpec{1}));

  // Rendering twice gives the same transcript, and the context adds neither a
  // system prompt nor tools: those belong to the agent that owns it.
  const std::vector<Message> first = context.BuildMessages();
  const std::vector<Message> second = context.BuildMessages();
  ASSERT_EQ(first.size(), second.size());
  ASSERT_EQ(first.size(), 4u);
  EXPECT_EQ(first[0].role, Role::Assistant);
  EXPECT_EQ(first[3].role, Role::User);
  for (const Message& message : first) {
    EXPECT_NE(message.role, Role::System);
  }
}

TEST(LlmContextTest, ClearDecisionsResetsTheHistory) {
  Context context;
  context.SetContextUpdate(EmptyUpdate());
  context.AppendDecision(MakeDecision(DecisionSpec{1}));
  ASSERT_EQ(context.decision_count(), 1u);

  context.ClearDecisions();

  EXPECT_EQ(context.decision_count(), 0u);
  EXPECT_EQ(context.Render().decisions_included, 0u);
}

TEST(LlmContextTest, RevisionsTrackTheNewestUpdateOnly) {
  Context context;
  context.SetContextUpdate(Update("Minute 10.", {}, "rev-10"));
  context.AppendDecision(MakeDecision(DecisionSpec{1}));

  // A decision recorded under an older revision does not change what the
  // session considers current.
  context.SetContextUpdate(Update("Minute 70.", {}, "rev-70"));

  EXPECT_EQ(context.context_revision(), "rev-70");
  EXPECT_EQ(context.decisions()[0].context_revision, "rev-1");
}

}  // namespace
}  // namespace llm
