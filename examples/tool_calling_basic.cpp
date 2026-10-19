/**
 * Basic Tool Calling Example - AI SDK C++
 *
 * The SDK turns one completion into a batch of tool calls. It does not run
 * them: this example defines tools, asks for a decision, prints the calls the
 * model requested, and then dispatches them itself.
 *
 * It shows how to:
 * - Describe tools with JSON schemas
 * - Receive one or more tool calls from a single request
 * - Execute them on the caller's side (the SDK never does)
 * - Control tool choice (required, specific, none)
 *
 * Usage:
 *   export OPENAI_API_KEY=your_key_here
 *   ./tool_calling_basic
 */

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>

#include <ai/openai.h>
#include <ai/tools.h>

namespace {

// The caller owns execution: this is plain application code, not SDK code.
ai::JsonValue ExecuteTool(const std::string& name, const ai::JsonValue& args) {
  if (name == "weather") {
    const std::string location = args.value("location", "unknown");
    std::srand(std::time(nullptr));
    const int temperature = 60 + (std::rand() % 40);  // 60-100 F
    std::cout << "  [executed] weather(" << location << ")\n";
    return ai::JsonValue{{"location", location},
                         {"temperature", temperature},
                         {"unit", "Fahrenheit"},
                         {"condition", "Partly cloudy"}};
  }

  if (name == "cityAttractions") {
    const std::string city = args.value("city", "unknown");
    ai::JsonValue attractions;
    if (city == "San Francisco") {
      attractions = {"Golden Gate Bridge", "Alcatraz Island",
                     "Fisherman's Wharf"};
    } else if (city == "New York") {
      attractions = {"Statue of Liberty", "Central Park", "Times Square"};
    } else {
      attractions = {"Downtown area", "Local parks"};
    }
    std::cout << "  [executed] cityAttractions(" << city << ")\n";
    return ai::JsonValue{{"city", city}, {"attractions", attractions}};
  }

  return ai::JsonValue{{"error", "unknown tool: " + name}};
}

ai::ToolSet BuildToolSet() {
  return {
      {"weather", ai::create_tool_schema(
                      "Get current weather information for a location",
                      ai::create_object_schema({{"location", "string"}}))},
      {"cityAttractions",
       ai::create_tool_schema("Get popular tourist attractions for a city",
                              ai::create_object_schema({{"city", "string"}}))},
  };
}

// Print the batch and dispatch each call on the caller's side.
void HandleResult(const ai::GenerateResult& result) {
  if (!result) {
    std::cout << "Error: " << result.error_message() << "\n\n";
    return;
  }

  if (!result.text.empty()) {
    std::cout << "Assistant text: " << result.text << "\n";
  }

  if (!result.has_tool_calls()) {
    std::cout << "No tool calls (finish reason: "
              << result.finishReasonToString() << ")\n\n";
    return;
  }

  std::cout << "Tool calls (" << result.tool_calls.size() << "):\n";
  for (const auto& call : result.tool_calls) {
    std::cout << "  - " << call.tool_name << "(" << call.arguments.dump()
              << ") id=" << call.id << "\n";
    const ai::JsonValue output = ExecuteTool(call.tool_name, call.arguments);
    std::cout << "    -> " << output.dump() << "\n";
  }

  std::cout << "Usage: " << result.usage.total_tokens << " tokens\n\n";
}

}  // namespace

int main() {
  std::cout << "AI SDK C++ - Basic Tool Calling Example\n";
  std::cout << "=========================================\n\n";

  auto client = ai::openai::create_client();
  const ai::ToolSet tools = BuildToolSet();

  // 1. The model must call a tool; the caller executes it.
  std::cout << "1. Required tool call\n";
  ai::GenerateOptions single(ai::openai::models::kGpt56,
                             "What's the weather like in San Francisco?");
  single.tools = tools;
  single.tool_choice = ai::ToolChoice::required();
  single.max_tokens = 200;
  HandleResult(client.generate_text(single));

  // 2. One request may come back with several tool calls.
  std::cout << "2. Several tools in one batch\n";
  ai::GenerateOptions multiple(
      ai::openai::models::kGpt56,
      "I'm planning a trip to San Francisco: check the weather and suggest "
      "attractions.");
  multiple.tools = tools;
  multiple.max_tokens = 300;
  HandleResult(client.generate_text(multiple));

  // 3. Force one specific tool.
  std::cout << "3. Forced tool choice\n";
  ai::GenerateOptions forced(ai::openai::models::kGpt56,
                             "Tell me about the weather in New York");
  forced.tools = tools;
  forced.tool_choice = ai::ToolChoice::specific("weather");
  forced.max_tokens = 150;
  HandleResult(client.generate_text(forced));

  // 4. Offer the tools but forbid their use.
  std::cout << "4. Tools disabled\n";
  ai::GenerateOptions disabled(ai::openai::models::kGpt56,
                               "What's the weather like in Boston?");
  disabled.tools = tools;
  disabled.tool_choice = ai::ToolChoice::none();
  disabled.max_tokens = 100;
  HandleResult(client.generate_text(disabled));

  std::cout << "Tool calling example completed!\n";
  return 0;
}
