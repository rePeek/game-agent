// Debugging aid: watch a complete decision round on the console.
//
// The SDK ships no logger, so this uses the test-tree ConsoleLogger. It is
// registered as a test (and does not fail for logging reasons), which makes the
// event stream easy to read:
//
//   ctest -R console_logger_demo -V
//   ./build/tests/integration/console_logger_demo
//
// The scripted transport keeps it offline: it shows the agent lifecycle
// (started, completed, recorded, rejected, discarded, closed) and the provider
// events, all correlated by [session_id:request_id].

#include "console_logger.h"

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <llm/sdk.h>

namespace {

class ScriptedTransport : public llm::ITransport {
 public:
  llm::HttpResponse Send(const llm::HttpRequest&) override {
    llm::HttpResponse response;
    response.transport_ok = true;
    response.status = 200;
    response.body = R"({
      "model": "gpt-5.6",
      "choices": [{
        "finish_reason": "tool_calls",
        "message": {
          "role": "assistant",
          "content": null,
          "tool_calls": [{
            "id": "call_a",
            "type": "function",
            "function": {
              "name": "set_pressing_intensity",
              "arguments": "{\"level\":4}"
            }
          }]
        }
      }],
      "usage": {"prompt_tokens": 64, "completion_tokens": 12, "total_tokens": 76}
    })";
    return response;
  }
};

llm::ToolDefinition PressingTool() {
  llm::ToolDefinition tool;
  tool.name = "set_pressing_intensity";
  tool.description = "Raise or lower the pressing intensity";
  tool.parameters_schema_json =
      R"({"type":"object","properties":{"level":{"type":"integer","minimum":1,"maximum":5}},"required":["level"],"additionalProperties":false})";
  tool.strict = true;
  return tool;
}

llm::ContextUpdate Update(const std::string& revision) {
  llm::ContextUpdate update;
  update.current_context = "minute=62; opponent presses high on our left";
  update.recent_history = "possession 42%, opponent shots 3";
  update.context_revision = revision;
  return update;
}

llm::ToolResult ResultFor(const std::string& call_id) {
  llm::ToolResult result;
  result.tool_call_id = call_id;
  result.name = "set_pressing_intensity";
  result.content_json = R"({"applied":true})";
  return result;
}

}  // namespace

int main() {
  std::cout << "game-agent - ConsoleLogger demo\n";
  std::cout << "================================\n\n";

  // One logger, injected into both the provider and the agent, so the whole
  // decision is visible as one stream.
  auto logger = std::make_shared<ConsoleLogger>();

  llm::openai::ProviderOptions provider_options;
  provider_options.api_key = "scripted";
  provider_options.logger = logger;

  llm::AgentConfig config;
  config.id = "home";
  config.model = llm::openai::models::kGpt56;
  config.system_prompt = "You are the head coach. Answer only with tool calls.";
  config.logger = logger;
  config.max_tokens = 300;
  config.max_tool_calls = 3;

  // A scripted transport keeps the demo offline; with a real key, drop it and
  // use CreateProviderFromEnvironment(logger) instead.
  provider_options.transport = std::make_shared<ScriptedTransport>();
  llm::Agent agent(llm::openai::CreateProvider(std::move(provider_options)),
                   std::move(config));
  agent.SetTools({PressingTool()});

  std::cout << "-- decision 1: a normal round\n";
  agent.UpdateContext(Update("rev-62"));
  const llm::DecisionResult first = agent.DecideAsync().get();
  if (!first.completion.IsSuccess()) {
    std::cerr << "unexpected: " << first.completion.ErrorMessage() << "\n";
    return 1;
  }

  std::cout << "\n-- a second decision before recording the first: rejected\n";
  agent.UpdateContext(Update("rev-63"));
  agent.DecideAsync().get();

  std::cout << "\n-- recording the first decision's results\n";
  std::vector<llm::ToolResult> results;
  for (const llm::ToolCall& call : first.completion.tool_calls) {
    results.push_back(ResultFor(call.id));
  }
  agent.RecordToolResults(first.decision_id, results);

  std::cout << "\n-- decision 2, then discarding it (stale revision)\n";
  agent.UpdateContext(Update("rev-70"));
  const llm::DecisionResult second = agent.DecideAsync().get();
  if (second.decision_id != 0) {
    agent.DiscardDecision(second.decision_id);
  }

  std::cout << "\n-- closing the agent\n";
  agent.Close();

  std::cout << "\nagent state: " << agent.ContextSnapshot().decision_count()
            << " confirmed decision(s), " << agent.PendingDecisionCount()
            << " pending\n";
  return 0;
}
