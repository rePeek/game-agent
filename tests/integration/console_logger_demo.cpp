// Debugging aid: watch a complete decision round on the console.
//
// The SDK ships no logger, so this uses the test-tree ConsoleLogger. It is
// registered as a test (and does not fail for logging reasons), which makes the
// event stream easy to read:
//
//   ctest -R console_logger_demo -V
//   ./build/tests/integration/console_logger_demo
//
// The scripted transport keeps it offline: it shows the session lifecycle
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

  // One logger, injected into both the provider and the session, so the whole
  // decision is visible as one stream.
  auto logger = std::make_shared<ConsoleLogger>();

  llm::openai::ProviderOptions provider_options;
  provider_options.api_key = "scripted";
  provider_options.logger = logger;

  llm::SessionConfig config;
  config.model = llm::openai::models::kGpt56;
  config.system_prompt = "You are the head coach. Answer only with tool calls.";
  config.session_id = "home";
  config.logger = logger;
  config.max_tokens = 300;
  config.max_tool_calls = 3;

  // A scripted transport keeps the demo offline; with a real key, drop it and
  // use CreateProviderFromEnvironment(logger) instead.
  provider_options.transport = std::make_shared<ScriptedTransport>();
  auto session = llm::CreateSession(
      llm::openai::CreateProvider(std::move(provider_options)),
      std::move(config));
  session->SetTools({PressingTool()});

  std::cout << "-- decision 1: a normal round\n";
  const llm::DecisionResult first = session->Complete(Update("rev-62"));
  if (!first.completion.IsSuccess()) {
    std::cerr << "unexpected: " << first.completion.ErrorMessage() << "\n";
    return 1;
  }

  std::cout << "\n-- a second decision before recording the first: rejected\n";
  session->Complete(Update("rev-63"));

  std::cout << "\n-- recording the first decision's results\n";
  std::vector<llm::ToolResult> results;
  for (const llm::ToolCall& call : first.completion.tool_calls) {
    results.push_back(ResultFor(call.id));
  }
  session->RecordToolResults(first.decision_id, results);

  std::cout << "\n-- decision 2, then discarding it (stale revision)\n";
  const llm::DecisionResult second = session->Complete(Update("rev-70"));
  if (second.decision_id != 0) {
    session->DiscardDecision(second.decision_id);
  }

  std::cout << "\n-- closing the session\n";
  session->Close();

  std::cout << "\nsession state: "
            << session->ContextSnapshot().decision_count()
            << " confirmed decision(s), " << session->pending_decision_count()
            << " pending\n";
  return 0;
}
