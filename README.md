# game-agent

A C++20 SDK for **structured tool calling**: one request in, one validated batch
of tool calls out.

```
context + tool definitions
           |
           v
   provider.Complete(request)          one model request, no retries into a
           |                           conversation, no tool execution
           v
   CompletionResult
     status         Success | NoToolCalls | InvalidToolCalls | Truncated |
                    Refused | Unsupported | ProviderError
     tool_calls     ids, names and argument JSON, validated
           |
           v
   the application decides what to execute and what to remember
```

The SDK does not chat, does not run an agent loop and does not execute tools.
That is deliberate: a game or an application owns its state, its permissions and
its side effects, and it needs the model's decisions as data, not as prose.

## What it does and does not do

| Does | Does not |
| --- | --- |
| Describe tools with JSON schemas and forward them to the provider | Execute tools or apply their results |
| Force tool calling (`tool_choice`) and request strict schemas | Open the next decision window on its own |
| Return every tool call in order, with ids and raw arguments | Stream tokens or SSE |
| Validate the batch (known tools, count, JSON, schema) before Success | Summarize history with another model call |
| Keep the transcript of a decision stream (`llm::Session`) | Let a rejected or stale batch into that transcript |
| Report failures as statuses instead of silent fallbacks | Provide a general chat/text-generation API |

## Completion statuses

`CompletionStatus` is the only thing a caller has to branch on:

| Status | Meaning | Safe to execute the batch? |
| --- | --- | --- |
| `Success` | The batch passed every check | yes |
| `NoToolCalls` | The model answered with text or nothing | no |
| `InvalidToolCalls` | Unknown tool, bad count, unparseable or schema-breaking arguments | no |
| `Truncated` | The token limit cut the completion off | no |
| `Refused` | The model refused, or a content filter blocked it | no |
| `Unsupported` | The request needs a guarantee this provider cannot make | no |
| `ProviderError` | Transport, HTTP, auth, rate limit or parse failure (`error->kind`) | no |

`result.error` carries a `ModelError` (kind, message, http status, retryable,
raw body) and `result.provider_metadata` keeps the raw provider response.

## Quick start

```cpp
#include <iostream>
#include <llm/core.h>
#include <llm/openai.h>

int main() {
  llm::ModelRequest request;
  request.model = llm::openai::models::kGpt56;
  request.messages = {
      llm::Message::System(
          "You are the head coach. Answer only with tool calls."),
      llm::Message::User(
          "Minute 62, 1-1, opponent presses high down our left side."),
  };

  llm::ToolDefinition pressing;
  pressing.name = "set_pressing_intensity";
  pressing.description = "Raise or lower how aggressively the team presses";
  pressing.parameters_schema_json = R"({
    "type": "object",
    "properties": {"level": {"type": "integer", "minimum": 1, "maximum": 5}},
    "required": ["level"],
    "additionalProperties": false
  })";
  pressing.strict = true;

  request.tools = {
      pressing,
      // A decision round that changes nothing must be expressible, otherwise a
      // forced tool call pushes the model into inventing a change.
      llm::ToolDefinition{"keep_current_tactics",
                          "Keep the current setup",
                          R"({"type":"object","properties":{"reason_code":{"type":"string"}},"required":["reason_code"],"additionalProperties":false})",
                          true},
  };
  request.tool_choice = llm::ToolChoice::Required();
  request.max_tool_calls = 3;  // application policy

  auto provider = llm::openai::CreateProviderFromEnvironment();  // OPENAI_API_KEY
  const llm::CompletionResult result = provider->Complete(request);

  if (!result.IsSuccess()) {
    std::cerr << llm::ToString(result.status) << ": " << result.ErrorMessage()
              << "\n";
    return 1;
  }

  for (const llm::ToolCall& call : result.tool_calls) {
    // The application dispatches; the SDK never does.
    std::cout << call.name << " " << call.arguments_json << "\n";
  }
}
```

Runnable versions: [`examples/coach_decision.cpp`](examples/coach_decision.cpp)
(real API) and [`examples/offline_transport.cpp`](examples/offline_transport.cpp)
(scripted transport, no key, no network — it also prints the request body the
SDK would send).

## Sessions

A game usually does not want to assemble messages per decision. `llm::Session`
owns the transcript of one decision stream (one coach, one side) and hands out
decisions:

```cpp
llm::SessionConfig config;
config.model = llm::openai::models::kGpt56;
config.system_prompt = "You are the head coach. Answer only with tool calls.";
config.context.max_messages = 32;   // transcript budget
config.max_tokens = 300;
config.max_tool_calls = 3;          // enforced by the validator

auto session = llm::CreateSession(
    llm::openai::CreateProviderFromEnvironment(), std::move(config));
session->SetTools(coach_tools);     // std::vector<llm::ToolDefinition>

// Decision window: submit and keep simulating.
std::future<llm::DecisionResult> future = session->CompleteAsync({
    .current_context = board.Describe(),      // replaces the previous snapshot
    .recent_history = coach.RecentTrends(),   // replaces the previous window
    .context_revision = board.Revision(),
});

// Later in the game loop:
if (session->WaitForIdle(std::chrono::milliseconds(0))) {
  const llm::DecisionResult decision = future.get();
  if (decision.status == llm::DecisionStatus::Completed &&
      decision.completion.IsSuccess() &&
      decision.context_revision == board.Revision()) {   // staleness check
    auto results = coach_tools.ExecuteBatch(decision.completion.tool_calls);
    session->RecordToolResults(decision.decision_id, results);
  } else {
    session->DiscardDecision(decision.decision_id);
  }
}
```

What the Session guarantees:

- **state is not history**: the current context and the recent-history window
  replace their predecessors, so old snapshots never pile up;
- **one decision at a time**: while a decision is in flight or awaiting
  results, a new one is refused as `Busy` (and its context update is not
  applied), so a decision is never taken with an incomplete transcript;
- **atomic confirmation**: a decision enters the transcript only when the
  application reports exactly one result per tool call — otherwise nothing is
  recorded and the decision stays pending. `DiscardDecision` drops it instead
  (a stale revision, a rolled-back transaction);
- **staleness is visible**: every decision carries the `context_revision` it
  was taken from; deciding whether it is still relevant stays with the game;
- **no blocking**: `CompleteAsync` snapshots the request on the calling thread
  and runs the provider call on the session's own worker, so a 100 Hz loop
  never waits on the network. `Close()` refuses further work and discards a
  decision that returns afterwards; the destructor joins the worker.

The transcript is trimmed deterministically by `ContextConfig::max_messages`
and `max_decisions`: whole decisions are dropped oldest-first, never half of
one, and `RenderedContext` reports what was kept. No summarization model call
is involved.

## Logging

The SDK defines a logging interface and produces events; it ships no logger,
no formatting and no logging dependency. An application injects one:

```cpp
class MyLogger : public llm::ILogger {
 public:
  void Log(const llm::LogRecord& record) override;  // level, event, message,
                                                    // session_id, request_id
};

auto logger = std::make_shared<MyLogger>();

llm::SessionConfig config;
config.model  = llm::openai::models::kGpt56;
config.logger = logger;          // session events
config.session_id = "home";      // which decision stream this is

llm::openai::ProviderOptions provider_options;
provider_options.api_key = "...";
provider_options.logger  = logger;   // provider events, same sink
```

Events are stable names (`llm::log_event::*`): `session.created`,
`session.decision_started`, `session.decision_completed`,
`session.decision_recorded`, `session.decision_discarded`,
`session.decision_rejected` (busy or closed — the answer to "why was this
Busy?"), `session.decision_record_rejected`, `session.closed`,
`provider.request_started`, `provider.retry`, `provider.request_completed`,
`provider.request_failed`, `provider.requirement_downgraded` and
`tools.validation_failed`.

Every record carries `session_id` and the per-attempt `request_id`, so a
provider shared by several agents can still be attributed:

```
INFO  [home:1] session.decision_started: rev=rev-62, tools=3, messages=2
INFO  [home:1] provider.request_started: model=gpt-5.6, tools=3, …
WARN  [home:1] provider.retry: attempt=1, http=429, kind=rate limit, …
INFO  [home:1] provider.request_completed: status=success, http=200, attempts=2, …
INFO  [home:1] session.decision_completed: status=success, tools=3, tokens=128, …
INFO  [home:1] session.decision_recorded: decision_id=1, results=3, rev=rev-62
```

What the SDK guarantees, and what it deliberately leaves out:

- **no logger means no logging**, and a logger that throws cannot change a
  decision's outcome (dispatch swallows the exception);
- the logger is never called while a session lock is held, so an
  implementation may call back into the session;
- only metadata is logged — never prompts, API keys or raw responses;
- events are per decision and per request, not per simulation tick;
- logging is diagnostics only: `CompletionResult` and the session history stay
  authoritative.

Test-side implementations live in `tests/support/`: `ConsoleLogger` for reading
a run by hand and `CaptureLogger` for asserting on events.

## Feeding a previous decision back

`llm::Session` does this for you; the section is here because the pieces it
uses are public. `llm::Message` keeps the tool vocabulary, so a decision round —
the assistant's tool calls plus their results — can be sent back as context.
The SDK builds neither the messages nor the loop:

```cpp
llm::ModelRequest next;
next.messages = previous_messages;
next.messages.push_back(decision.ToAssistantMessage());  // assistant + tool_calls
for (const auto& call : decision.tool_calls) {
  llm::ToolResult applied;
  applied.tool_call_id = call.id;
  applied.name = call.name;
  applied.content_json = R"({"applied":true})";  // your tool output as JSON text
  next.messages.push_back(llm::Message::ToolResult(applied));
}
next.messages.push_back(llm::Message::User("They switched to a low block."));
```

On the wire this becomes the OpenAI messages an OpenAI-compatible server
expects: an assistant turn with `tool_calls` (arguments as the raw JSON text)
followed by one `role: "tool"` message per call.

## Validation

`Success` is only ever returned for a batch that passed:

- the batch size window (at least one call; at most `ModelRequest::max_tool_calls`);
- ids that can be answered with tool results;
- tool names that the request actually offered;
- arguments that parse as a JSON object;
- arguments that satisfy the tool's JSON Schema (type, enum, const, properties,
  required, additionalProperties, items with min/maxItems, minimum/maximum,
  minLength/maxLength, anyOf/oneOf/allOf).

An application can add its own policy by injecting a validator:

```cpp
class OneChangePerRound : public llm::ToolCallValidator {
 public:
  llm::ValidationReport Validate(const llm::ModelRequest&,
                                 const std::vector<llm::ToolCall>& calls) const override;
};

llm::openai::ProviderOptions options;
options.api_key = "...";
options.validator = std::make_shared<const OneChangePerRound>();
```

## Providers and guarantees

A provider is what a Session wraps; it is also usable directly (see the coach
decision example).

A provider publishes what it can actually guarantee
(`ProviderCapabilities`: parallel tool calls, forced tool choice, strict
schemas, tool results in the context). If a request needs a guarantee the
provider does not have, the SDK answers `Unsupported` **before** sending
anything, rather than pretending the model was constrained. Callers that accept
SDK-side validation instead set `allow_validation_downgrade = true`; the
validation still runs, and the result carries a warning.

```cpp
llm::openai::ProviderOptions options;
options.api_key = "...",
options.base_url = "https://openrouter.ai/api";  // any OpenAI-compatible endpoint
options.capabilities.strict_tool_schema = false; // what this endpoint honours
options.retry.max_attempts = 5;
options.timeout_seconds = 30;
options.extra_headers.push_back({"X-Title", "my game"});
```

Retrying covers 408/409/429/5xx responses and transport failures, with
configurable attempts, backoff and maximum delay.

## Offline and in tests

`llm::ITransport` is the only thing a provider needs from the outside world, so
the whole SDK runs without a network:

```cpp
class ScriptedTransport : public llm::ITransport {
 public:
  llm::HttpResponse Send(const llm::HttpRequest&) override;
};

llm::openai::ProviderOptions options;
options.api_key = "scripted";
options.transport = std::make_shared<ScriptedTransport>();
auto provider = llm::openai::CreateProvider(std::move(options));
```

The SDK's own tests use exactly this seam, which is why they need no API key.

## Building

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON
cmake --build build --parallel
cmake --install build --prefix /usr/local
```

`BUILD_EXAMPLES` (default `ON`) and `BUILD_TESTS` (default `OFF`) control the
optional targets. `scripts/build.py` wraps the same flow for development:

```bash
uv run scripts/build.py --mode release --tests --export-compile-commands
```

## Using it from another project

```cmake
find_package(game-agent CONFIG REQUIRED COMPONENTS core openai)
target_link_libraries(my_target PRIVATE llm::openai)
```

Four modules, one target each:

| Component | Target | Depends on |
| --- | --- | --- |
| `core` | `llm::core` | the standard library |
| `tools` | `llm::tools` | `llm::core` (JSON is private) |
| `context` | `llm::context` | `llm::core` (no provider, no TLS) |
| `openai` | `llm::openai` | `llm::core`, `llm::tools` (+ OpenSSL for its transport) |
| `sdk` | `llm::sdk` | all of the above (an INTERFACE target, no extra archive) |

`target_link_libraries(... llm::sdk)` is the shortcut for everything, and only
`llm::openai` brings OpenSSL into the link line. `#include <llm/sdk.h>` is the
matching convenience header.

## Dependencies

The public headers are standard-library-only; these are implementation details:

- **cpp-httplib** — vendored header-only HTTP client, used by the OpenAI
  provider (with OpenSSL for TLS);
- **nlohmann/json** — request/response JSON and schema validation; an existing
  `nlohmann_json::nlohmann_json` target is reused when the enclosing project
  provides one, otherwise `find_package()` is tried, otherwise CMake fetches the
  official release with a pinned SHA256;
- **OpenSSL** — required by the HTTP transport;
- **GoogleTest** — tests only, fetched by CMake.

A consumer of the installed package needs OpenSSL and Threads only.

## Requirements

- C++20 compiler (GCC 10+, Clang 12+, MSVC 2019+)
- CMake 3.16+

## Examples

- [Coach session](examples/coach_session.cpp) — the intended flow for a game:
  a Session, asynchronous decisions, revision checks and history that moves
  forward only when results are reported (runs offline without a key).
- [Coach decision](examples/coach_decision.cpp) — the provider on its own:
  context, tools, one completion, a validated batch, caller-side dispatch.
- [Offline transport](examples/offline_transport.cpp) — the same flow with a
  scripted transport, including what an invalid batch looks like.

See [DEVELOPMENT.md](DEVELOPMENT.md) for the build scripts, testing and
linting.

## Acknowledgments

Inspired by the excellent [Vercel AI SDK](https://github.com/vercel/ai) for
TypeScript/JavaScript developers.
