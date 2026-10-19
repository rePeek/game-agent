/**
 * Coach Agent Example - AI SDK C++
 *
 * The intended application flow: the game owns one llm::Agent per decision
 * stream (one coach, one side) and drives it.
 *
 *   decision window opens
 *     -> agent.UpdateContext(context)     // the world changed
 *     -> agent.DecideAsync()              // never blocks the tick
 *   ... the game keeps simulating ...
 *     -> future.get()                     // the model answered
 *     -> the game checks the context revision
 *     -> agent.PrepareCommands(id)        // bind arguments, run nothing
 *     -> the game invokes the commands at its own tick boundary
 *     -> agent.RecordCommandResults(...)  // history moves forward
 *
 * The SDK owns the transcript (system prompt, commands, current context, recent
 * history, confirmed decisions) and the argument binding. The game owns the
 * state, the side effects and the decision to apply anything.
 *
 * Without OPENAI_API_KEY the example drives a scripted transport, so it runs
 * anywhere and still shows the whole flow.
 *
 * Usage:
 *   ./coach_agent                    # offline, scripted
 *   export OPENAI_API_KEY=... && ./coach_agent
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

/// Answers with a fixed command-call batch: used when no API key is available.
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

/// The game registers its own C++ functions; the SDK derives the schema the
/// model sees and the argument binding from the same declaration.
void RegisterCoachCommands(llm::Agent& agent) {
  (void)agent.RegisterCommand(
      "set_pressing_intensity",
      "Raise or lower how aggressively the team presses",
      llm::Params(llm::Param<int>("level").Range(1, 5).Required(),
                  llm::Param<std::string>("reason_code")
                      .OneOf("chasing", "stable", "protect")
                      .Required()),
      [](int level, const std::string& reason) -> llm::CommandOutcome {
        std::cout << "  set_pressing_intensity(level=" << level
                  << ", reason=" << reason << ")\n";
        return llm::CommandOutcome{true, R"({"applied":true})", {}};
      });

  (void)agent.RegisterCommand(
      "set_defensive_line", "Move the defensive line up or down the pitch",
      llm::Params(llm::Param<std::string>("height")
                      .OneOf("high", "medium", "low")
                      .Required()),
      [](const std::string& height) -> llm::CommandOutcome {
        std::cout << "  set_defensive_line(height=" << height << ")\n";
        return llm::CommandOutcome{true, R"({"applied":true})", {}};
      });

  // A round that changes nothing has to be expressible, otherwise a forced
  // command call pushes the model into inventing a change.
  (void)agent.RegisterCommand(
      "keep_current_tactics",
      "Keep the current setup; no change is needed this round",
      llm::Params(llm::Param<std::string>("reason_code")
                      .OneOf("stable", "no_signal")
                      .Required()),
      [](const std::string& reason) -> llm::CommandOutcome {
        std::cout << "  keep_current_tactics(reason=" << reason << ")\n";
        return llm::CommandOutcome{true, R"({"applied":true})", {}};
      });
}

llm::ContextUpdate BuildContext(int minute,
                                const std::string& revision,
                                const std::string& decision_context) {
  llm::ContextUpdate update;
  update.current_context =
      "minute=" + std::to_string(minute) +
      "; score=1-1; opponent=high full-backs, attacking down our left; " +
      "board=4-3-3, medium block, pressing 2, defensive line medium";
  update.recent_history =
      "last 10 minutes: possession 42%, opponent shots 3, our shots 0";
  update.decision_context = decision_context;
  update.context_revision = revision;
  return update;
}

/// One decision window: submit, (in a game: keep simulating), collect, bind and
/// execute, and move the history forward only if the answer is still about the
/// current state.
void RunWindow(llm::Agent& agent, int minute, const std::string& revision) {
  std::cout << "decision window (minute " << minute << ", revision " << revision
            << ")\n";

  agent.UpdateContext(BuildContext(
      minute, revision,
      "decision at minute " + std::to_string(minute) + ", score 1-1"));
  std::future<llm::DecisionResult> future = agent.DecideAsync();

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
  if (decision.context_revision != agent.current_revision()) {
    std::cout << "  stale decision (" << decision.context_revision
              << " != " << agent.current_revision() << "), discarded\n\n";
    agent.DiscardDecision(decision.decision_id);
    return;
  }

  // Binding argues no side effects; the game decides when to run.
  llm::PrepareResult prepared = agent.PrepareCommands(decision.decision_id);
  if (!prepared.ok()) {
    std::cout << "  could not prepare commands: " << prepared.error << "\n\n";
    agent.DiscardDecision(decision.decision_id);
    return;
  }

  std::vector<llm::CommandResult> results;
  results.reserve(prepared.commands.size());
  for (llm::PreparedCommand& command : prepared.commands) {
    results.push_back(std::move(command).Invoke());
  }

  switch (
      agent.RecordCommandResults(decision.decision_id, std::move(results))) {
    case llm::RecordOutcome::Recorded:
      std::cout << "  decision " << decision.decision_id
                << " recorded; the next request sees it as history\n\n";
      break;
    case llm::RecordOutcome::IncompleteResults:
      std::cout << "  results were incomplete; nothing was recorded\n\n";
      break;
    case llm::RecordOutcome::NameMismatch:
      std::cout
          << "  a result named a different command; nothing was recorded\n\n";
      break;
    case llm::RecordOutcome::UnknownDecision:
      std::cout << "  no such pending decision\n\n";
      break;
    case llm::RecordOutcome::Closed:
      std::cout << "  agent is closed\n\n";
      break;
  }
}

}  // namespace

int main() {
  std::cout << "AI SDK C++ - Coach Agent Example\n";
  std::cout << "===================================\n\n";

  const bool offline = std::getenv("OPENAI_API_KEY") == nullptr;

  // The SDK ships no logger: this example injects its own into both the
  // provider and the agent, so every event carries the same stream id.
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

  llm::AgentConfig config;
  config.id = "home";
  config.model = llm::openai::models::kGpt56;
  config.system_prompt =
      "You are the head coach. Answer only with command calls that change or "
      "keep "
      "the tactical setup. Never explain in prose.";
  config.context.max_messages = 32;  // budget for the transcript
  config.context.max_decisions = 4;
  config.max_output_tokens = 300;
  config.max_command_calls = 3;
  config.logger = logger;

  llm::Agent agent(provider, std::move(config));
  RegisterCoachCommands(agent);

  RunWindow(agent, 62, "rev-62");
  RunWindow(agent, 70, "rev-70");

  const llm::Context context = agent.ContextSnapshot();
  const llm::RenderedContext rendered = context.Render();
  std::cout << "\nsession state\n";
  std::cout << "  confirmed decisions: " << context.decision_count() << "\n";
  std::cout << "  messages to send:    " << rendered.messages.size()
            << " (included " << rendered.decisions_included << ", dropped "
            << rendered.decisions_dropped << ")\n";
  std::cout << "  pending decisions:   " << agent.PendingDecisionCount()
            << "\n";
  std::cout << "  current revision:    " << agent.current_revision() << "\n";

  std::cout << "\nThe SDK kept the transcript; the game kept the state and the "
               "side effects.\n";
  return 0;
}
