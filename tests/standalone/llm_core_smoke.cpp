// Standalone smoke test for the llm:: core.
//
// This translation unit includes only the llm:: core headers and the standard
// library, and its target links only llm::core — no JSON, HTTP, TLS or test
// framework. If the core ever grows a third-party dependency, this target stops
// building.

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <llm/core/message.h>
#include <llm/core/provider.h>
#include <llm/core/request.h>
#include <llm/core/result.h>
#include <llm/core/types.h>

namespace {

int failures = 0;

void Check(bool condition, const std::string& what) {
  if (!condition) {
    std::cerr << "FAIL: " << what << "\n";
    ++failures;
  }
}

}  // namespace

int main() {
  const std::string raw_arguments = R"({ "level" : 3 })";
  llm::ToolCall call{"call_001", "set_pressing_intensity", raw_arguments};
  Check(call.IsValid(), "a named tool call is valid");
  Check(call.arguments_json == raw_arguments, "raw arguments are preserved");

  llm::ToolResult tool_result;
  tool_result.tool_call_id = "call_001";
  tool_result.name = "set_pressing_intensity";
  tool_result.content_json = R"({"applied":true})";

  llm::ModelRequest request;
  request.model = "gpt-5.6";
  request.messages = {
      llm::Message::System("You are a coach."),
      llm::Message::User("Opponent is pressing high."),
      llm::Message::AssistantToolCalls({call}),
      llm::Message::ToolResult(tool_result),
  };
  request.tools.push_back(llm::ToolDefinition{
      "set_pressing_intensity", "Pressing intensity",
      R"({"type":"object","properties":{"level":{"type":"integer"}}})", true});
  request.max_tool_calls = 4;

  Check(request.messages.size() == 4u, "context keeps four messages");
  Check(request.messages[2].HasToolCalls(), "assistant turn keeps its calls");
  Check(request.messages[3].tool_call_id == call.id, "tool turn answers it");
  Check(request.tool_choice.mode == llm::ToolChoiceMode::Required,
        "forced tool calling is the default");
  Check(request.FindTool("set_pressing_intensity") != nullptr,
        "tool lookup works");

  llm::CompletionResult result;
  Check(!result.IsSuccess(), "a default result is not a success");
  Check(result.ErrorMessage() == "provider error", "status has a description");

  result.status = llm::CompletionStatus::Success;
  result.tool_calls = {call};
  result.usage = llm::Usage{11, 7, 18};
  result.finish_reason = llm::FinishReason::ToolCalls;
  result.assistant_text = "Raising pressing.";

  Check(result.IsSuccess(), "status carries the branch condition");
  Check(result.tool_calls.size() == 1u, "batch is reported");
  Check(result.usage.total_tokens == 18, "usage is preserved");
  Check(llm::ToString(result.finish_reason) == "tool_calls",
        "finish reason is preserved");

  const llm::Message assistant = result.ToAssistantMessage();
  Check(assistant.role == llm::Role::Assistant, "assistant message role");
  Check(assistant.tool_calls.size() == 1u, "assistant message keeps the batch");
  Check(assistant.tool_calls[0].arguments_json == raw_arguments,
        "batch arguments survive the round trip");

  if (failures == 0) {
    std::cout << "llm::core smoke test passed\n";
  }
  return failures == 0 ? 0 : 1;
}
