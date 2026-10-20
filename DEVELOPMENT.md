# Development Guide

How to build, test and contribute to game-agent.

## Prerequisites

The project is **Linux + GCC** only (the toolchain its host games use). The Nix
flake provides everything below in one shell.

- **GCC 14+** — the only C++ compiler (`-std=c++23`);
- **CMake 3.24+**;
- **Ninja**;
- **OpenSSL** development headers (for `llm::openai`);
- **clang-format** — formatting (a tool, not the compiler);
- **clang-tidy** — optional static analysis;
- **gdb** — debugging.

`clang-format`, `clang-tidy` and `clangd` never compile C++ here; GCC does.

## Development setup

```bash
git clone git@github.com:rePeek/game-agent.git
cd game-agent

# Enter the pinned GCC dev shell (or `direnv allow` once).
nix develop
```

The shell prints the compiler and the commands below. GoogleTest and
nlohmann/json are resolved or fetched by CMake during configure, so the first
configure needs network access.

### Live API configuration

The offline suite needs no key. For runs against a real endpoint, copy the
example environment file and fill it in:

```bash
cp .env.example .env
# OPENAI_API_KEY=...
# OPENAI_BASE_URL=https://api.openai.com
# OPENAI_MODEL=...
```

`.env` is git-ignored. `.envrc` (`use flake` + `dotenv_if_exists .env`) loads it
into the development shell with direnv; CMake never reads it, so keys cannot
reach `CMakeCache.txt`, a build log or the Nix store. Tests read the variables
at run time.

## Building

CMake presets drive everything. Debug and Release use separate build
directories (`build/debug`, `build/release`).

```bash
cmake --preset debug
cmake --build --preset debug --parallel
ctest --preset debug
```

Release is the same with `release`:

```bash
cmake --preset release
cmake --build --preset release --parallel
ctest --preset release
```

Native CMake flags cover what a wrapper script used to:

```bash
cmake --build --preset debug --clean-first     # rebuild from scratch
cmake --build --preset release --verbose
cmake --build --preset release --parallel 12
```

Options: `BUILD_EXAMPLES` (default `ON`) and CTest's `BUILD_TESTING` (the presets
set it `ON`). Compile commands are always exported to
`build/<preset>/compile_commands.json` for clangd and clang-tidy.

Installing:

```bash
cmake --install build/release --prefix /usr/local
```

## Formatting

clang-format runs through CMake targets; there is no Python wrapper.

```bash
cmake --build --preset debug --target format        # rewrite in place
cmake --build --preset debug --target format-check  # check only (CI)
```

The target scans only `include/`, `src/`, `tests/` and `examples/`, so
`third_party/`, `build/` and FetchContent sources are never reformatted. Pin the
same clang-format version in the Nix shell and CI (the flake provides it) so the
result is stable.

## Static analysis (optional)

`clang-tidy` reads the compile database produced during configure:

```bash
clang-tidy -p build/debug src/agent.cpp
```

There is no bespoke lint script; use `clang-tidy` directly (or a
`run-clang-tidy -p build/debug` invocation).

## Testing

The suite is hermetic: providers are driven through a scripted transport, so it
needs no API key and no network.

```bash
ctest --preset debug

ctest --preset debug --verbose
ctest --preset debug -R "Llm"
ctest --preset debug --parallel
```

Each module has its own test binary, so the linker checks the module boundary:

- `tests/core/` — `llm_core_tests` (the protocol types) and `llm_core_smoke`, a
  target that links only `llm::core` (no JSON, no HTTP, no TLS, not even a test
  framework). If the core ever grows a third-party dependency, it stops
  building;
- `tests/commands/` — `llm_commands_tests` (parameter descriptors, schema
  generation, argument decoding, registry and batch validation);
- `tests/context/` — `llm_context_tests` (snapshot replacement, decision
  contiguity, budget trimming);
- `tests/agent/` — `llm_agent_tests` (request shape, staging, atomic
  confirmation, the one-decision-at-a-time rule, isolation between agents,
  owner-thread polling and closing) driven by a scripted `IModelProvider`;
- `tests/providers/openai/` — `llm_openai_tests` (request building, response
  parsing, provider status mapping, log events) against a fake transport,
  including one test that exercises the real cpp-httplib transport through a
  refused connection;
- `tests/integration/` — `llm_integration_tests`: Agent + Context + the OpenAI
  provider end to end over a scripted transport, asserting the wire bodies of
  consecutive rounds; `llama_smoke_tests`, an optional real-server test that is
  skipped unless `GAME_AGENT_LLAMA_BASE_URL` and `GAME_AGENT_LLAMA_MODEL` are
  set; and `console_logger_demo`, a program that prints a full decision round
  through the test-tree ConsoleLogger.

`tests/support/` holds what the SDK deliberately does not ship: the two loggers
(`ConsoleLogger`, `CaptureLogger`) and `agent_driver.h`, the helper that drives
the owner-thread Agent (`DecideAsync`/`PollDecision`).

### Live smoke test

```bash
GAME_AGENT_LLAMA_BASE_URL=http://127.0.0.1:8080 \
GAME_AGENT_LLAMA_MODEL=<model-id> \
  ctest --preset debug -R LlamaSmoke --output-on-failure
```

The manual `Live API smoke` GitHub workflow runs the same test against a real
endpoint using repository secrets; normal CI never needs a key.

### Writing tests

Add the file under the module's test directory (`tests/<module>/`) and list it in
that directory's `CMakeLists.txt`. The provider takes its transport from
`ProviderOptions`, so a test scripts responses instead of hitting the API:

```cpp
#include <gtest/gtest.h>
#include <llm/providers/openai/openai.h>

class ScriptedTransport : public llm::ITransport {
 public:
  llm::HttpResponse Send(const llm::HttpRequest&) override {
    llm::HttpResponse response;
    response.transport_ok = true;
    response.status = 200;
    response.body = R"({"choices":[{"message":{"tool_calls":[]}}]})";
    return response;
  }
};

TEST(MyProviderTest, RejectsAnEmptyBatch) {
  llm::openai::ProviderOptions options;
  options.api_key = "scripted";
  options.transport = std::make_shared<ScriptedTransport>();
  // ...
}
```

## Diagnostics

Every call reports what happened twice, for two different audiences.

For the application, on the returned `llm::CompletionResult` — the authoritative
data:

```cpp
const llm::CompletionResult result = provider->Complete(request);

std::cout << llm::ToString(result.status) << "\n";
if (!result.IsSuccess()) {
  std::cout << result.ErrorMessage() << "\n";        // human readable
  if (result.error) {
    std::cout << llm::ToString(result.error->kind)      // network/auth/...
              << " retryable: " << result.error->retryable
              << " http: " << result.error->http_status << "\n";
  }
}
for (const auto& warning : result.warnings) {         // e.g. a downgrade
  std::cout << "warning: " << warning << "\n";
}
```

`result.provider_metadata` keeps the raw provider response, and
`result.finish_reason`, `result.usage`, `result.model` and `result.http_attempts`
carry the protocol data unchanged.

For the developer, as an event stream: the SDK emits records through an
`llm::ILogger` injected via `AgentConfig::logger` and `ProviderOptions::logger`.
The SDK ships no implementation — `tests/support/` has the two the tests use,
and `console_logger_demo` prints a whole decision round:

```bash
ctest --preset debug -R console_logger_demo -V
# or run it directly:
./build/debug/tests/integration/console_logger_demo
```

Log records carry classifications and counts (kind, status, http code, attempts,
tokens, command counts, schema summaries). Provider and model text is never
logged: it stays in `CompletionResult` for the caller.

## Dependencies

The public headers depend on nothing but the standard library. Third-party C++
sources come from CPM, pinned in `cmake/Dependencies.cmake` (the CPM release
itself is pinned in `cmake/CPM.cmake`); system libraries use `find_package`:

- **nlohmann-json** `v3.11.3`: request/response JSON and schema generation and
  validation. Private to `llm::commands` and `llm::openai`;
- **cpp-httplib** `v0.22.0`: header-only HTTP client for the transport (private
  to `llm::openai`). Header-only on purpose, so no Python is needed;
- **openssl** (`find_package(OpenSSL 3.0 REQUIRED)`) and **threads**
  (`find_package(Threads REQUIRED)`): system libraries. Only `llm::openai`
  needs them, and the installed target exports them for static linking;
- **googletest**: a pinned revision, fetched only when `BUILD_TESTING` is on.

`CPM_SOURCE_CACHE` (`.cache/CPM`) keeps the fetched sources out of the source
tree and lets a reconfigure work offline. A parent project that embeds
game-agent can reuse its own cache and the packages it already registered.

Neither nlohmann-json nor cpp-httplib appears in a public header, so a consumer
of the installed package only needs OpenSSL and Threads.

## Pre-commit workflow

```bash
# 1. Format
cmake --build --preset debug --target format

# 2. Build and test
cmake --build --preset debug --parallel
ctest --preset debug

# 3. Optional static analysis
clang-tidy -p build/debug src/agent.cpp
```

CI runs the same steps for both presets, plus `format-check`.

## Contributing

### Code style

clang-format (Chromium base, 2-space indent, 80 columns) is the source of truth:

- **Naming**: CamelCase for types, snake_case for functions and variables;
- **Modern C++**: C++23, `auto` for obvious types, `nullptr`, smart pointers;
- **RAII** and move semantics over manual resource management;
- mark functions `const`, `constexpr`, `noexcept` when appropriate;
- brace initialization `{}`; capture lambda variables explicitly, avoid `[=]`
  and `[&]`.

### Pull request process

1. Fork the repository;
2. Create a feature branch: `git checkout -b feature/amazing-feature`;
3. Make your changes following the code style;
4. Run the pre-commit workflow (format, build, test);
5. Commit with descriptive messages;
6. Push to your fork and open a Pull Request.
