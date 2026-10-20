# game-agent

A C++23 SDK for **structured command calling**: a game registers its own C++
functions, the model picks them, and the SDK hands back validated, bound
callables that the game chooses when to run.

```
context + command definitions
           |
           v
   Agent::DecideAsync()              snapshot the request, start the model call,
           |                           return immediately (owner thread)
           v  (model thread)
   provider.Complete(request)          one model request, no retries into a
           |                           conversation, no command execution
           v
   CompletionResult
     status         Success | NoCommandCalls | InvalidCommandCalls | Truncated |
                    Refused | Unsupported | ProviderError
    command_calls  ids, names and argument JSON, validated
           |
           v  Agent::PollDecision() on the owner thread stages the decision
   Agent::PrepareCommands()            decode arguments, bind the C++ callback,
           |                           run nothing
           v
   the game invokes the bound commands at its own tick boundary, then
   Agent::RecordCommandResults() moves the transcript forward
```

The SDK does not chat, does not run an agent loop and never executes a command.
That is deliberate: a game owns its state, its permissions and its side effects,
and it needs the model's decisions as data — bound to the game's own functions,
but not applied on the SDK's schedule.

## What it does and does not do

| Does | Does not |
| --- | --- |
| Register C++ functions with typed parameters and derive the JSON Schema | Execute commands or apply their results |
| Force command calling and request strict schemas | Open the next decision window on its own |
| Return every command call in order, with ids and raw arguments | Stream tokens or SSE |
| Validate the batch (known commands, count, JSON, schema) before `Success` | Summarize history with another model call |
| Bind validated arguments to the registered callback (`PrepareCommands`) | Run the callback for you |
| Keep the transcript of a decision stream (`llm::Agent`) | Let a rejected or stale batch into that transcript |
| Report failures as statuses instead of silent fallbacks | Provide a general chat/text-generation API |

## Completion statuses

`CompletionStatus` is the only thing a caller has to branch on:

| Status | Meaning | Safe to prepare and execute? |
| --- | --- | --- |
| `Success` | The batch passed every check | yes |
| `NoCommandCalls` | The model answered with text or nothing | no |
| `InvalidCommandCalls` | Unknown command, bad count, unparseable or schema-breaking arguments | no |
| `Truncated` | The token limit cut the completion off | no |
| `Refused` | The model refused, or a content filter blocked it | no |
| `Unsupported` | The request needs a guarantee this provider cannot make | no |
| `ProviderError` | Transport, HTTP, auth, rate limit or parse failure (`error->kind`) | no |

`result.error` carries a `ModelError` (kind, message, http status, retryable,
raw body) and `result.provider_metadata` keeps the raw provider response.

## Quick start

### An Agent with typed commands

```cpp
#include <llm/sdk.h>

llm::AgentConfig config;
config.id = "home";                       // how this stream appears in logs
config.model = llm::openai::models::kGpt56;
config.system_prompt = "You are the head coach. Answer only with command calls.";
config.context.max_messages = 32;         // transcript budget
config.context.max_input_tokens = 8000;   // approximate request budget
config.max_output_tokens = 300;
config.max_command_calls = 3;             // enforced by the validator

llm::Agent agent(std::move(config));
agent.SetProvider(llm::openai::CreateProviderFromEnvironment());

// Register the game's own functions. The parameter list types the callback, so
// a mismatch is a compile-time error, and the JSON Schema is derived from it.
agent.RegisterCommand(
    "set_pressing_intensity", "Raise or lower how aggressively the team presses",
    llm::Params(llm::Param<int>("level").Range(1, 5).Required(),
                llm::Param<std::string>("reason_code")
                    .OneOf("chasing", "stable", "protect")
                    .Required()),
    [&coach](int level, const std::string& reason) -> llm::CommandOutcome {
      return coach.SetPressing(level, reason);
    });

// A decision window opens: the world changed...
agent.UpdateContext({.current_context = board.Describe(),
                     .recent_history = coach.RecentTrends(),
                     .decision_context = "minute 62, score 1-1",
                     .context_revision = board.Revision()});
// ...and the model is asked, without blocking the tick.
const llm::StartOutcome started = agent.DecideAsync();
if (!started.ok()) {
  return;  // NotConfigured / Busy / ContextTooLarge / Closed / Failed
}

// Later in the game loop: PollDecision() never blocks.
std::optional<llm::DecisionResult> decision = agent.PollDecision();
if (!decision.has_value()) {
  return;  // the model has not answered yet
}
const llm::DecisionResult& result = *decision;
if (result.status != llm::DecisionStatus::Completed ||
    !result.completion.IsSuccess()) {
  return;  // handle a refusal or a model error
}
if (result.context_revision != board.Revision()) {
  agent.DiscardDecision(result.decision_id);  // stale answer
  return;
}

// Bind the model's arguments to the registered callbacks. No side effects yet.
llm::PrepareResult prepared = agent.PrepareCommands(result.decision_id);
if (!prepared.ok()) {
  agent.DiscardDecision(result.decision_id);  // e.g. bad arguments
  return;
}

// The game chooses the execution point.
std::vector<llm::CommandResult> results;
for (llm::PreparedCommand& command : prepared.commands) {
  results.push_back(std::move(command).Invoke());
}
agent.RecordCommandResults(result.decision_id, std::move(results));
```

What the agent guarantees:

- **state is not history**: `UpdateContext` replaces the snapshot, the
  recent-history window and the decision framing. It may be called at any time,
  including while a decision is in flight — the update then applies to the
  *next* decision, never to the request already sent;
- **one decision at a time**, per agent: while a decision is in flight or
  awaiting results, a new one is refused as `Busy`. Two agents may share one
  provider and be in flight simultaneously; they share nothing else;
- **typed registration**: `RegisterCommand` records one declaration that drives
  both the schema the model sees and the decoder used to bind arguments, so the
  two can never disagree. A callback must return `llm::CommandOutcome` (or
  `void`, for a command with no result);
- **every provider's output is revalidated**: whatever `IModelProvider`
  answered, a batch is staged only after the SDK's own validator checks it
  against the request snapshot (batch size, known commands, unique ids, an
  argument object and the command's schema). A provider that reports `Success`
  without validating cannot reach a callback. A bad batch becomes
  `InvalidCommandCalls` and stages nothing;
- **a decision needs commands**: with an empty registry `DecideAsync` returns
  `NoCommands` rather than sending a request that forces a call the model
  cannot make;
- **result JSON is checked**: `RecordCommandResults` refuses a
  `content_json` that is not well-formed (`InvalidResultJson`) instead of
  letting a broken callback poison the next request;
- **prepare never runs anything**: `PrepareCommands` decodes and binds the whole
  batch. If any call cannot be decoded, the batch fails as a whole and no
  command is returned — never half a decision;
- **one-shot commands**: `PreparedCommand` is move-only and `Invoke()` succeeds
  exactly once. A command is invalidated when its decision is recorded,
  discarded, or the agent is closed;
- **atomic confirmation**: a decision enters the transcript only when the game
  reports exactly one result per command call, naming the same commands.
  Otherwise nothing is recorded and the decision stays pending;
- **visibility of failure**: a command that throws, or is no longer valid, is
  reported as a failed `CommandOutcome` instead of escaping into the game loop;
- **staleness is visible**: every decision carries the `context_revision` it was
  taken from; whether that is still current is the game's call;
- **no blocking**: `DecideAsync` snapshots the request and returns immediately;
  the provider runs on its own thread and only hands back a `CompletionResult`
  through a future. `PollDecision()` is how the owner thread collects it.
  `Close()` refuses further work and invalidates prepared commands; a request
  that is already running is answered as `Closed` when polled, or waited out
  when the agent is destroyed.

`WaitForIdle()` waits for the model call to finish (it does not collect the
result).

### Thread ownership

An Agent has an **owner thread** — the thread that created it — and every
public method must be called from it. The model thread only calls the provider
and returns a `CompletionResult`; it never reads or writes Agent state. That
is why the Agent holds no mutex or condition variable.

Two consequences:

- collect results with `PollDecision()` on the owner thread, typically at a
  safe point in your tick;
- a logger may be invoked from the model thread (provider events). A logger
  must therefore not call back into an Agent; capture into a thread-safe sink
  and consume it on the owner thread.

### Lifetime and timeouts

`Close()` does not cancel a request that is already on the wire, and the Agent
has no cancellation API. Destroying an Agent waits for the in-flight call
(the future's destructor joins the model thread), so do not destroy a working
Agent in a real-time tick: stop it on a non-real-time path, or keep it alive.
The provider's own `timeout_seconds` and retry policy decide the worst-case
wait; a local `llama-server` should use a short timeout and no retries.

### Parameters and the generated schema

`llm::Param<T>` supports `bool`, integers, floating point, `std::string` and
string enumerations. Fluent constraints: `Description`, `Required`/`Optional`,
`Default`, `Range`/`Min`/`Max`, `MinLength`/`MaxLength`, `OneOf`.

The generated schema only uses constraints the SDK validator implements, so a
schema the SDK produces is always one it can enforce. Three rules keep the
schema, the validator and the decoder in agreement:

- a parameter that is neither `Required` nor has a `Default` is refused at
  registration: an omitted value must have one defined meaning;
- an optional parameter is expressed as a nullable type (`["integer",
  "null"]`) and stays out of `required`, so it can be omitted or sent as
  `null`, which the SDK then fills from the default;
- a command with `strict = true` is sent to a strict-capable provider with
  every property moved into `required` (OpenAI's strict dialect) and `default`
  dropped (it is not in that subset). A provider that cannot enforce strict
  gets the logical schema and no `strict` flag, and the SDK validator remains
  the guarantee.

```cpp
llm::Params(
    llm::Param<int>("level").Range(1, 5).Required(),
    llm::Param<std::string>("reason").OneOf("a", "b").Default("a"))
// -> level is required; reason is optional and nullable, and means "a" when
//    the model leaves it out.
```

### Context, budgets and memory

`llm::Context` is the component the Agent owns: the current snapshot, the
recent-history window, the decision framing and the confirmed decisions. The
transcript is trimmed deterministically, oldest whole decision first:

- `max_messages` / `max_decisions` bound the rendered request;
- `max_memory_decisions` bounds what is kept in memory;
- `max_input_tokens` (approximate, see `EstimateTokens`) covers the system
  prompt, the command schemas, the current facts and the messages;
  `reserved_output_tokens` is subtracted first. History is expendable: an
  oversized decision is dropped (oldest first) rather than failing the
  request. Only when the system prompt, command schemas and current facts
  alone exceed the budget does the agent answer `ContextTooLarge` **without
  sending anything**.

A decision is stored and trimmed as one unit — its framing user turn, the
assistant command calls and one result per call. No summarization model call is
involved.

## Using a provider directly

The provider is also usable without an Agent (see the coach decision example).
The vocabulary is the same; only the binding is left to you.

```cpp
llm::ModelRequest request;
request.model = llm::openai::models::kGpt56;
request.messages = {
    llm::Message::System("You are the head coach. Answer only with command calls."),
    llm::Message::User("Minute 62, 1-1, opponent presses high down our left side."),
};
request.commands = {
    llm::CommandDefinition{
        "set_pressing_intensity",
        "Raise or lower how aggressively the team presses",
        R"({"type":"object","properties":{"level":{"type":"integer","minimum":1,"maximum":5}},"required":["level"],"additionalProperties":false})",
        true},
    // A round that changes nothing must be expressible, otherwise a forced
    // command call pushes the model into inventing a change.
    llm::CommandDefinition{
        "keep_current_tactics", "Keep the current setup",
        R"({"type":"object","properties":{"reason_code":{"type":"string"}},"required":["reason_code"],"additionalProperties":false})",
        true},
};
request.command_choice = llm::CommandChoice::Required();
request.max_command_calls = 3;  // application policy

auto provider = llm::openai::CreateProviderFromEnvironment();  // OPENAI_API_KEY
const llm::CompletionResult result = provider->Complete(request);

if (!result.IsSuccess()) {
  std::cerr << llm::ToString(result.status) << ": " << result.ErrorMessage() << "\n";
  return 1;
}
for (const llm::CommandCall& call : result.command_calls) {
  // The application dispatches; the SDK never does.
  std::cout << call.name << " " << call.arguments_json << "\n";
}
```

### Feeding a previous decision back

`llm::Agent` does this for you; the pieces are public because a direct provider
user may want them. A `CommandResult` is built from a `CommandOutcome`:

```cpp
llm::ModelRequest next;
next.messages = previous_messages;
next.messages.push_back(previous.ToAssistantMessage());  // assistant + command_calls
for (const llm::CommandCall& call : previous.command_calls) {
  llm::CommandResult applied;
  applied.command_call_id = call.id;
  applied.name = call.name;
  applied.outcome = coach.Execute(call);  // your result, as a CommandOutcome
  next.messages.push_back(llm::Message::CommandResultMessage(applied));
}
next.messages.push_back(llm::Message::User("They switched to a low block."));
```

On the wire this becomes the OpenAI messages an OpenAI-compatible server
expects: an assistant turn with `tool_calls` (arguments as the raw JSON text)
followed by one `role: "tool"` message per call. The tool message body has one
shape for success and failure alike:

```json
{"ok": true, "result": {...}}
{"ok": false, "error": "substitution_not_allowed"}
```

so the success flag is transmitted rather than implied by the payload.

## Validation

`Success` is only ever returned for a batch that passed:

- the batch size window (at least one call; at most `ModelRequest::max_command_calls`);
- ids that can be answered with command results;
- command names that the request actually offered;
- arguments that parse as a JSON object;
- arguments that satisfy the command's JSON Schema (type, enum, const,
  properties, required, additionalProperties, items with min/maxItems,
  minimum/maximum, minLength/maxLength, anyOf/oneOf/allOf).

An application can add its own policy by injecting a validator:

```cpp
class OneChangePerRound : public llm::CommandValidator {
 public:
  llm::ValidationReport Validate(
      const llm::ModelRequest&,
      const std::vector<llm::CommandCall>& calls) const override;
};

llm::openai::ProviderOptions options;
options.api_key = "...";
options.validator = std::make_shared<const OneChangePerRound>();
```

## Providers and guarantees

A provider publishes what it can actually guarantee (`ProviderCapabilities`:
parallel command calls, forced command choice, strict schemas, command results
in the context). If a request needs a guarantee the provider does not have, the
SDK answers `Unsupported` **before** sending anything, rather than pretending
the model was constrained. Callers that accept SDK-side validation instead set
`allow_validation_downgrade = true`; the validation still runs, and the result
carries a warning.

```cpp
llm::openai::ProviderOptions options;
options.api_key = "...";
options.base_url = "https://openrouter.ai/api";  // any OpenAI-compatible endpoint
options.capabilities.strict_command_schema = false;  // what this endpoint honours
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

One optional test talks to a real OpenAI-compatible server (for example a local
`llama-server`). It is skipped unless two environment variables are set:

```bash
export GAME_AGENT_LLAMA_BASE_URL=http://127.0.0.1:8080
export GAME_AGENT_LLAMA_MODEL=<model-id>
ctest --preset debug -R LlamaSmoke --output-on-failure
```

For a real API key, copy `.env.example` to `.env` (direnv loads it into the
development shell) and set `OPENAI_API_KEY`, `OPENAI_BASE_URL` and
`OPENAI_MODEL`. CMake never reads the file.

It exercises a single decision, two agents sharing one provider, and repeated
decisions with a bounded history, asserting that a real server's output is
either a fully validated, bound batch or a reported failure.

## Building

The build is driven by CMake presets; Debug and Release use separate build
directories under `build/`. The first configure fetches the third-party
sources with CPM into `.cache/CPM` (git-ignored), then reuses them offline.

```bash
cmake --preset debug        # or: release
cmake --build --preset debug --parallel
ctest --preset debug
```

Formatting is a CMake target (no Python wrapper):

```bash
cmake --build --preset debug --target format        # rewrite in place
cmake --build --preset debug --target format-check  # CI
```

`BUILD_EXAMPLES` (default `ON`) and CTest's `BUILD_TESTING` (default `ON`
through the presets) control the optional targets. To install:

```bash
cmake --install build/release --prefix /usr/local
```

## Using it from another project

```cmake
find_package(game-agent CONFIG REQUIRED COMPONENTS core openai)
target_link_libraries(my_target PRIVATE llm::openai)
```

Five modules, one target each:

| Component | Target | Depends on |
| --- | --- | --- |
| `core` | `llm::core` | the standard library |
| `commands` | `llm::commands` | `llm::core` (JSON is private) |
| `context` | `llm::context` | `llm::core`, `llm::commands` (the Agent and its Context; no provider, no TLS) |
| `openai` | `llm::openai` | `llm::core`, `llm::commands` (+ OpenSSL for its transport) |
| `sdk` | `llm::sdk` | all of the above (an INTERFACE target, no extra archive) |

`target_link_libraries(... llm::sdk)` is the shortcut for everything, and only
`llm::openai` brings OpenSSL into the link line. `#include <llm/sdk.h>` is the
matching convenience header.

## Dependencies

The public headers are standard-library-only. Third-party C++ sources are
fetched by CPM with pinned versions (`cmake/CPM.cmake` fixes the CPM release,
`cmake/Dependencies.cmake` declares the dependencies); system libraries stay
with `find_package`:

- **nlohmann/json** `v3.11.3` — request/response JSON, schema generation and
  validation (kept in step with Football Sim). Private to `llm::commands` and
  `llm::openai`, so it never reaches a consumer's include path;
- **cpp-httplib** `v0.22.0` — header-only HTTP client for the OpenAI provider
  (with OpenSSL for TLS). Header-only on purpose: the compiled mode needs
  Python, which this project does not use;
- **OpenSSL** (`find_package(OpenSSL 3.0 REQUIRED)`) and **Threads**
  (`find_package(Threads REQUIRED)`) — system libraries, not CPM;
- **GoogleTest** — a pinned revision, tests only (`BUILD_TESTING`).

Fetched sources live in `CPM_SOURCE_CACHE` (`.cache/CPM`, git-ignored). A
parent build that embeds the SDK can point the cache at its own directory and
reuse packages it already registered.

A consumer of the installed package needs OpenSSL and Threads only, and only
when it links `llm::openai`.

## Requirements

The project is Linux + GCC only (that is the toolchain its host games use):

- GCC 14 or newer, in C++23 mode (`-std=c++23`);
- CMake 3.24 or newer;
- Ninja;
- OpenSSL (development headers) for `llm::openai`.

`clang-format`, `clang-tidy` and `clangd` are used as tools, never as the
C++ compiler.

## Examples

- [Coach agent](examples/coach_agent.cpp) — the intended flow for a game: an
  Agent, typed command registration, asynchronous decisions, revision checks,
  `PrepareCommands`/`Invoke` and history that moves forward only when results
  are recorded (runs offline without a key).
- [Coach decision](examples/coach_decision.cpp) — the provider on its own:
  context, commands, one completion, a validated batch, caller-side dispatch.
- [Offline transport](examples/offline_transport.cpp) — the same flow with a
  scripted transport, including what an invalid batch looks like.

See [DEVELOPMENT.md](DEVELOPMENT.md) for the build scripts, testing and linting.

## Acknowledgments

Inspired by the excellent [Vercel AI SDK](https://github.com/vercel/ai) for
TypeScript/JavaScript developers.
