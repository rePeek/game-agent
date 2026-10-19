/**
 * Offline Transport Example - AI SDK C++
 *
 * Providers talk to the world through llm::ITransport, so the whole SDK can be
 * driven without a network. This example scripts three responses and shows:
 *
 *   - what the request looks like on the wire;
 *   - a validated batch (Success);
 *   - a batch that fails schema validation (InvalidToolCalls);
 *   - a batch naming a tool the caller never offered (InvalidToolCalls).
 *
 * The same trick is how the SDK's own tests run in CI.
 *
 * Usage:
 *   ./offline_transport
 */

#include <iostream>
#include <memory>
#include <string>

#include <llm/core/message.h>
#include <llm/core/provider.h>
#include <llm/core/request.h>
#include <llm/core/result.h>
#include <llm/core/types.h>
#include <llm/providers/openai/openai.h>

namespace {

/// Returns the response the test scripted for this call, and prints the request
/// the provider would have sent.
class ScriptedTransport : public llm::ITransport {
 public:
  explicit ScriptedTransport(std::string response)
      : response_(std::move(response)) {}

  llm::HttpResponse Send(const llm::HttpRequest& request) override {
    std::cout << "  -> POST " << request.url << "\n";
    std::cout << "     body: " << request.body << "\n";

    llm::HttpResponse response;
    response.transport_ok = true;
    response.status = 200;
    response.body = response_;
    return response;
  }

 private:
  std::string response_;
};

std::string ResponseWith(const std::string& tool_calls) {
  return R"({"id":"chatcmpl-scripted","model":"gpt-5.6","choices":[{"index":0,)"
         R"("finish_reason":"tool_calls","message":{"role":"assistant",)"
         R"("content":null,"tool_calls":)" +
         tool_calls +
         R"(}}],"usage":{"prompt_tokens":42,"completion_tokens":12,"total_tokens":54}})";
}

llm::ModelRequest BuildRequest() {
  llm::ModelRequest request;
  request.model = llm::openai::models::kGpt56;
  request.messages = {
      llm::Message::System("You are the head coach."),
      llm::Message::User("Opponent is pressing high down our left side."),
  };

  llm::ToolDefinition tool;
  tool.name = "set_pressing_intensity";
  tool.description = "Raise or lower the pressing intensity";
  tool.parameters_schema_json = R"({
    "type": "object",
    "properties": {
      "level": {"type": "integer", "minimum": 1, "maximum": 5}
    },
    "required": ["level"],
    "additionalProperties": false
  })";
  tool.strict = true;
  request.tools = {tool};
  request.tool_choice = llm::ToolChoice::Required();
  return request;
}

std::shared_ptr<llm::IModelProvider> MakeProvider(std::string response) {
  llm::openai::ProviderOptions options;
  options.api_key = "scripted-key";
  options.transport = std::make_shared<ScriptedTransport>(std::move(response));
  return llm::openai::CreateProvider(std::move(options));
}

void RunScenario(const std::string& title, const std::string& response) {
  std::cout << title << "\n";
  const llm::CompletionResult result =
      MakeProvider(response)->Complete(BuildRequest());

  std::cout << "  status: " << llm::ToString(result.status) << "\n";
  for (const auto& warning : result.warnings) {
    std::cout << "  warning: " << warning << "\n";
  }
  if (result.IsSuccess()) {
    for (const auto& call : result.tool_calls) {
      std::cout << "  call: " << call.name << " " << call.arguments_json
                << " (id " << call.id << ")\n";
    }
  } else {
    std::cout << "  rejected: " << result.ErrorMessage() << "\n";
  }
  std::cout << "\n";
}

}  // namespace

int main() {
  std::cout << "AI SDK C++ - Offline Transport Example\n";
  std::cout << "======================================\n\n";

  RunScenario(
      "1. A valid batch",
      ResponseWith(
          R"([{"id":"call_1","type":"function","function":)"
          R"({"name":"set_pressing_intensity","arguments":"{\"level\":4}"}}])"));

  RunScenario(
      "2. Arguments that break the schema",
      ResponseWith(
          R"([{"id":"call_1","type":"function","function":)"
          R"({"name":"set_pressing_intensity","arguments":"{\"level\":9}"}}])"));

  RunScenario("3. A tool the caller never offered",
              ResponseWith(R"([{"id":"call_1","type":"function","function":)"
                           R"({"name":"sign_a_striker","arguments":"{}"}}])"));

  std::cout
      << "Only the first batch is safe to apply: every other status means "
         "the game must not execute anything.\n";
  return 0;
}
