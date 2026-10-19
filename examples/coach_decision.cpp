/**
 * Coach Decision Example - AI SDK C++
 *
 * The whole SDK contract in one program:
 *
 *   context + tool definitions -> one completion -> a validated batch of tool
 *   calls -> the caller decides what to execute.
 *
 * Note what is *not* here: no agent loop, no conversation bookkeeping, no tool
 * execution inside the SDK. The game owns the state and the side effects.
 *
 * Usage:
 *   export OPENAI_API_KEY=your_key_here
 *   ./coach_decision
 */

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <llm/sdk.h>

namespace {

llm::ToolDefinition PressingTool() {
  llm::ToolDefinition tool;
  tool.name = "set_pressing_intensity";
  tool.description =
      "Raise or lower how aggressively the team presses the opponent";
  tool.parameters_schema_json = R"({
    "type": "object",
    "properties": {
      "level": {"type": "integer", "minimum": 1, "maximum": 5},
      "reason_code": {"type": "string", "enum": ["chasing", "stable", "protect"]}
    },
    "required": ["level", "reason_code"],
    "additionalProperties": false
  })";
  tool.strict = true;
  return tool;
}

llm::ToolDefinition DefensiveLineTool() {
  llm::ToolDefinition tool;
  tool.name = "set_defensive_line";
  tool.description = "Move the defensive line up or down the pitch";
  tool.parameters_schema_json = R"({
    "type": "object",
    "properties": {
      "height": {"type": "string", "enum": ["high", "medium", "low"]}
    },
    "required": ["height"],
    "additionalProperties": false
  })";
  tool.strict = true;
  return tool;
}

/// A decision round that changes nothing must be expressible, otherwise a
/// forced tool call would push the model into inventing a change.
llm::ToolDefinition KeepTacticsTool() {
  llm::ToolDefinition tool;
  tool.name = "keep_current_tactics";
  tool.description = "Keep the current setup; no change is needed this round";
  tool.parameters_schema_json = R"({
    "type": "object",
    "properties": {
      "reason_code": {"type": "string", "enum": ["stable", "no_signal"]}
    },
    "required": ["reason_code"],
    "additionalProperties": false
  })";
  tool.strict = true;
  return tool;
}

llm::ModelRequest BuildRequest() {
  llm::ModelRequest request;
  request.model = llm::openai::models::kGpt56;

  // The caller assembles the context: match state, tactical board, and — in a
  // real loop — the previous decision with its tool results.
  request.messages = {
      llm::Message::System(
          "You are the head coach. Answer only with tool calls that change or "
          "keep the tactical setup. Never explain in prose."),
      llm::Message::User(
          "Match state: minute 62, score 1-1, opponent is pushing both "
          "full-backs high and attacking down our left side.\n"
          "Tactical board: 4-3-3, medium block, pressing level 2, defensive "
          "line medium.\n"
          "Available decision: adjust pressing, adjust the defensive line, or "
          "keep the current tactics."),
  };

  request.tools = {PressingTool(), DefensiveLineTool(), KeepTacticsTool()};
  request.tool_choice = llm::ToolChoice::Required();
  request.max_tokens = 300;
  request.max_tool_calls = 3;  // application policy
  return request;
}

/// Executing a decision is the game's job. The SDK hands over validated calls
/// and the raw argument text; here we only report what would be applied.
void ApplyDecision(const llm::ToolCall& call) {
  std::cout << "  applying " << call.name << " with arguments "
            << call.arguments_json << " (id " << call.id << ")\n";
}

}  // namespace

int main() {
  std::cout << "AI SDK C++ - Coach Decision Example\n";
  std::cout << "====================================\n\n";

  const llm::ModelRequest request = BuildRequest();

  auto provider = llm::openai::CreateProviderFromEnvironment();
  if (std::getenv("OPENAI_API_KEY") == nullptr) {
    std::cout << "Set OPENAI_API_KEY to run this example against the API.\n"
              << "It sends one completion with " << request.tools.size()
              << " tools and a tool_choice of '"
              << request.tool_choice.ToString() << "'.\n"
              << "See examples/offline_transport.cpp for the same flow without "
                 "a network.\n";
    return 0;
  }

  const llm::CompletionResult result = provider->Complete(request);

  std::cout << "status: " << llm::ToString(result.status) << "\n";
  if (!result.model.empty()) {
    std::cout << "model: " << result.model << "\n";
  }
  std::cout << "tokens: " << result.usage.total_tokens << " (prompt "
            << result.usage.prompt_tokens << ", completion "
            << result.usage.completion_tokens << ")\n";
  for (const auto& warning : result.warnings) {
    std::cout << "warning: " << warning << "\n";
  }

  if (!result.IsSuccess()) {
    std::cout << "no decision to apply: " << result.ErrorMessage() << "\n";
    if (!result.assistant_text.empty()) {
      std::cout << "model said: " << result.assistant_text << "\n";
    }
    return 1;
  }

  std::cout << "\ndecision batch (" << result.tool_calls.size()
            << " call(s)):\n";
  for (const auto& call : result.tool_calls) {
    ApplyDecision(call);
  }

  std::cout << "\nThe batch is validated against the tool schemas, so the game "
               "can apply it directly.\n";
  return 0;
}
