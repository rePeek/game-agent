/**
 * Coach Session Example - AI SDK C++
 *
 * The intended application flow: the game talks to one llm::Session per
 * decision stream (one coach, one side).
 *
 *   decision window opens
 *     -> session.CompleteAsync(context update)   // never blocks the tick
 *   ... the game keeps simulating ...
 *     -> future.get() / WaitForIdle()            // decision is ready
 *     -> the game checks the context revision
 *     -> the game executes the batch itself
 *     -> session.RecordToolResults(...)          // history moves forward
 *
 * The SDK owns the transcript (system prompt, tools, current context, recent
 * history, confirmed decisions). The game owns the state, the side effects and
 * the decision to apply anything.
 *
 * Without OPENAI_API_KEY the example drives a scripted transport, so it runs
 * anywhere and still shows the whole flow.
 *
 * Usage:
 *   ./coach_session                  # offline, scripted
 *   export OPENAI_API_KEY=... && ./coach_session
 */

#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <llm/sdk.h>

namespace {

/// A logger is application-side: this one prints what the SDK reports. Football
/// Sim would adapt its own logging system here instead (and the SDK would not
/// change at all).
class EventPrinter final : public llm::ILogger {
 public:
  void Log(const llm::LogRecord& record) override {
    std::cout << "  " << LevelName(record.level) << " [" << record.session_id
              << ":" << record.request_id << "] " << record.event << ": "
              << record.message << "\n";
  }

 private:
  static const char* LevelName(llm::LogLevel level) {
    switch (level) {
      case llm::LogLevel::Debug:
        return "DEBUG";
      case llm::LogLevel::Info:
        return "INFO ";
      case llm::LogLevel::Warning:
        return "WARN ";
      case llm::LogLevel::Error:
        return "ERROR";
    }
    return "?????";
  }
};

/// Answers with a fixed tool-call batch: used when no API key is available.
class ScriptedTransport : public llm::ITransport {
 public:
  llm::HttpResponse Send(const llm::HttpRequest& request) override {
    std::cout << "  [offline] would POST " << request.url << "\n";
    const std::size_t window = calls_++;
    static const char* const kBodies[] = {
        R"([{"id":"call_1","type":"function","function":{"name":"set_pressing_intensity","arguments":"{\"level\":4,\"reason_code\":\"chasing\"}"}}])",
        R"([{"id":"call_2","type":"function","function":{"name":"keep_current_tactics","arguments":"{\"reason_code\":\"stable\"}"}}])",
    };
    const std::string tool_calls = kBodies[window < 2 ? window : 1];

    llm::HttpResponse response;
    response.transport_ok = true;
    response.status = 200;
    response.body =
        R"({"id":"chatcmpl-scripted","model":"gpt-5.6","choices":[{"index":0,)"
        R"("finish_reason":"tool_calls","message":{"role":"assistant",)"
        R"("content":null,"tool_calls":)" +
        tool_calls +
        R"(}}],"usage":{"prompt_tokens":64,"completion_tokens":12,"total_tokens":76}})";
    return response;
  }

 private:
  std::size_t calls_ = 0;
};

llm::ToolDefinition PressingTool() {
  llm::ToolDefinition tool;
  tool.name = "set_pressing_intensity";
  tool.description = "Raise or lower how aggressively the team presses";
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

llm::ToolDefinition LineTool() {
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

/// A round that changes nothing has to be expressible, otherwise a forced tool
/// call pushes the model into inventing a change.
llm::ToolDefinition KeepTool() {
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

llm::ContextUpdate BuildContext(int minute, const std::string& revision) {
  llm::ContextUpdate update;
  update.current_context =
      "minute=" + std::to_string(minute) +
      "; score=1-1; opponent=high full-backs, attacking down our left; " +
      "board=4-3-3, medium block, pressing 2, defensive line medium";
  update.recent_history =
      "last 10 minutes: possession 42%, opponent shots 3, our shots 0";
  update.context_revision = revision;
  return update;
}

/// Executing a decision is the game's job. Here we only report what would be
/// applied, and hand the SDK a result per call (the SDK never runs tools).
std::vector<llm::ToolResult> ExecuteBatch(
    const std::vector<llm::ToolCall>& calls) {
  std::vector<llm::ToolResult> results;
  results.reserve(calls.size());
  for (const llm::ToolCall& call : calls) {
    std::cout << "  applying " << call.name << " " << call.arguments_json
              << "\n";
    llm::ToolResult result;
    result.tool_call_id = call.id;
    result.name = call.name;
    result.content_json = R"({"applied":true})";
    result.is_error = false;
    results.push_back(std::move(result));
  }
  return results;
}

/// One decision window: submit, (in a game: keep simulating), collect, and
/// move the history forward only if the answer is still about the current
/// state.
void RunWindow(const std::shared_ptr<llm::Session>& session,
               int minute,
               const std::string& revision) {
  std::cout << "decision window (minute " << minute << ", revision " << revision
            << ")\n";

  std::future<llm::DecisionResult> future =
      session->CompleteAsync(BuildContext(minute, revision));

  // A game would keep ticking here. The example simply waits for the answer.
  const llm::DecisionResult decision = future.get();

  if (decision.status != llm::DecisionStatus::Completed) {
    std::cout << "  no decision: " << llm::ToString(decision.completion.status)
              << " (" << decision.completion.ErrorMessage() << ")\n\n";
    return;
  }
  if (!decision.completion.IsSuccess()) {
    std::cout << "  rejected by the SDK: "
              << llm::ToString(decision.completion.status) << " ("
              << decision.completion.ErrorMessage() << ")\n\n";
    return;
  }

  // The revision travelled with the decision, so a stale answer can be
  // recognised before anything is applied.
  if (decision.context_revision != session->current_revision()) {
    std::cout << "  stale decision (" << decision.context_revision
              << " != " << session->current_revision() << "), discarded\n\n";
    session->DiscardDecision(decision.decision_id);
    return;
  }

  const std::vector<llm::ToolResult> results =
      ExecuteBatch(decision.completion.tool_calls);

  switch (session->RecordToolResults(decision.decision_id, results)) {
    case llm::RecordOutcome::Recorded:
      std::cout << "  decision " << decision.decision_id
                << " recorded; the next request sees it as history\n\n";
      break;
    case llm::RecordOutcome::IncompleteResults:
      std::cout << "  results were incomplete; nothing was recorded\n\n";
      break;
    case llm::RecordOutcome::UnknownDecision:
      std::cout << "  no such pending decision\n\n";
      break;
    case llm::RecordOutcome::Closed:
      std::cout << "  session is closed\n\n";
      break;
  }
}

}  // namespace

int main() {
  std::cout << "AI SDK C++ - Coach Session Example\n";
  std::cout << "===================================\n\n";

  const bool offline = std::getenv("OPENAI_API_KEY") == nullptr;

  // The SDK ships no logger: this example injects its own into both the
  // provider and the session, so every event carries the same session id.
  auto logger = std::make_shared<EventPrinter>();
  std::cout << "log events (session " << "home" << "):\n";

  std::shared_ptr<llm::IModelProvider> provider;
  if (offline) {
    std::cout << "OPENAI_API_KEY is not set: running against a scripted "
                 "transport.\n\n";
    llm::openai::ProviderOptions options;
    options.api_key = "scripted";
    options.transport = std::make_shared<ScriptedTransport>();
    options.logger = logger;
    provider = llm::openai::CreateProvider(std::move(options));
  } else {
    provider = llm::openai::CreateProviderFromEnvironment(logger);
  }

  llm::SessionConfig config;
  config.model = llm::openai::models::kGpt56;
  config.system_prompt =
      "You are the head coach. Answer only with tool calls that change or keep "
      "the tactical setup. Never explain in prose.";
  config.context.max_messages = 32;  // budget for the transcript
  config.context.max_decisions = 4;
  config.max_tokens = 300;
  config.max_tool_calls = 3;
  config.session_id = "home";
  config.logger = logger;

  const std::shared_ptr<llm::Session> session =
      llm::CreateSession(provider, std::move(config));
  session->SetTools({PressingTool(), LineTool(), KeepTool()});

  RunWindow(session, 62, "rev-62");
  RunWindow(session, 70, "rev-70");

  const llm::Context context = session->ContextSnapshot();
  const llm::RenderedContext rendered = context.Render();
  std::cout << "\nsession state\n";
  std::cout << "  confirmed decisions: " << context.decision_count() << "\n";
  std::cout << "  messages to send:    " << rendered.messages.size()
            << " (included " << rendered.decisions_included << ", dropped "
            << rendered.decisions_dropped << ")\n";
  std::cout << "  pending decisions:   " << session->pending_decision_count()
            << "\n";
  std::cout << "  current revision:    " << session->current_revision() << "\n";

  std::cout << "\nThe SDK kept the transcript; the game kept the state and the "
               "side effects.\n";
  return 0;
}
