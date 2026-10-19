# Development Guide

This document provides comprehensive information about developing, building, and contributing to AI SDK C++.

## Prerequisites

### Required Tools

- **C++ Compiler**: C++20 compatible compiler (GCC 10+, Clang 12+, MSVC 2019+)
- **CMake**: Version 3.16 or higher
- **Python**: 3.8+ with uv package manager for development scripts
- **Git**: For version control

### Optional Tools (Recommended)

- **clang-format**: For code formatting
- **clang-tidy**: For static analysis and linting
- **gdb/lldb**: For debugging

## Development Setup

### 1. Clone the Repository

```bash
git clone git@github.com:rePeek/game-agent.git
cd game-agent
```

### 2. Install Python Dependencies

We use `uv` for managing Python development tools:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Development scripts will automatically install their dependencies
# when run with 'uv run'
```

## Building the Project

We provide a modern Python-based build script that handles all CMake configuration and building.

### Basic Build

```bash
# Debug build (default)
uv run scripts/build.py

# Or explicitly specify debug mode
uv run scripts/build.py --mode debug
```

### Release Build

```bash
uv run scripts/build.py --mode release
```

### Build with Tests

```bash
uv run scripts/build.py --mode debug --tests
```

### Clean Build

```bash
uv run scripts/build.py --mode release --clean
```

### Export Compile Commands (for IDE integration)

```bash
uv run scripts/build.py --mode debug --export-compile-commands
```

This generates `compile_commands.json` in both the build directory and project root, enabling excellent IDE support in VS Code, CLion, and other editors.

### Advanced Build Options

```bash
# Custom parallel jobs
uv run scripts/build.py --mode release --jobs 8

# Verbose output
uv run scripts/build.py --mode debug --verbose

# All options combined
uv run scripts/build.py --mode release --tests --clean --export-compile-commands --jobs 12
```

### Manual CMake Build

If you prefer using CMake directly:

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTS=ON \
  -DBUILD_EXAMPLES=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build
cmake --build build --parallel $(nproc)
```

Note: the vendored headers are used directly and everything else is resolved (or fetched) during the configuration step; see "Dependencies" below.

## Development Scripts

We provide several Python scripts to streamline development. All scripts use modern CLI tools (Click) and rich terminal output.

### 🔨 build.py - Build System

**Location**: `scripts/build.py`  
**Purpose**: Build the project with CMake

```bash
# Show all options
uv run scripts/build.py --help

# Common usage patterns
uv run scripts/build.py --mode debug                    # Debug build
uv run scripts/build.py --mode release --tests         # Release with tests
uv run scripts/build.py --clean --export-compile-commands  # Clean + IDE support
```

**Features**:
- ✅ Debug/Release builds
- ✅ Test building toggle
- ✅ Clean builds
- ✅ Export compile commands for IDEs
- ✅ Parallel building
- ✅ Automatic dependency management via CMake FetchContent
- ✅ Rich terminal output with progress indicators

### 🎨 format.py - Code Formatting

**Location**: `scripts/format.py`  
**Purpose**: Format C++ code using clang-format

```bash
# Format all files
uv run scripts/format.py

# Check formatting without modifying files
uv run scripts/format.py --check
```

**Features**:
- ✅ Formats all C++ files (`.h`, `.hpp`, `.cc`, `.cpp`, `.cxx`)
- ✅ Excludes build directories and dependency files
- ✅ Check mode for CI/CD
- ✅ Progress indicators

### 🔍 lint.py - Static Analysis

**Location**: `scripts/lint.py`  
**Purpose**: Run clang-tidy for static analysis and linting

```bash
# Lint all files
uv run scripts/lint.py

# Auto-fix issues where possible
uv run scripts/lint.py --fix

# Custom parallel jobs
uv run scripts/lint.py --jobs 8
```

**Features**:
- ✅ Parallel linting for faster execution
- ✅ Auto-fix capability
- ✅ Requires `compile_commands.json` (build with `--export-compile-commands`)
- ✅ macOS system include path handling
- ✅ Rich progress reporting

## Testing

### Running Tests

The test suite is hermetic: providers are driven through a scripted
transport, so it needs no API key and no network.

```bash
cd build

# Run all tests
ctest

# Verbose output
ctest --verbose

# Run the protocol tests only
ctest -R "Llm"

# Parallel test execution
ctest --parallel $(nproc)
```

The suite is made of:

Each module has its own test binary, so the linker checks the module boundary:

- `tests/core/` — `llm_core_tests` (the protocol types) and `llm_core_smoke`, a
  target that links only `llm::core` (no JSON, no HTTP, no TLS, not even a test
  framework). If the core ever grows a third-party dependency, it stops
  building;
- `tests/tools/` — `llm_tools_tests` (schema and batch validation);
- `tests/context/` — `llm_context_tests` (snapshot replacement, decision
  contiguity, budget trimming, staging, atomic confirmation, the
  one-decision-at-a-time rule and asynchronous submission) driven by a scripted
  `IModelProvider`, so it needs no provider and no TLS;
- `tests/providers/openai/` — `llm_openai_tests` (request building, response
  parsing, provider status mapping, log events) against a fake transport,
  including one test that exercises the real cpp-httplib transport through a
  refused connection;
- `tests/integration/` — `llm_integration_tests`: Session + Context + the
  OpenAI provider end to end over a scripted transport, asserting the wire
  bodies of consecutive rounds; plus `console_logger_demo`, a program that
  prints a full decision round through the test-tree ConsoleLogger (run it with
  `ctest -R console_logger_demo -V`).

`tests/support/` holds the two loggers the SDK deliberately does not ship:
`ConsoleLogger` (prints events, for reading a run by hand) and `CaptureLogger`
(collects them, so tests assert on events instead of console output).

GoogleTest is fetched by CMake FetchContent when tests are enabled:

```bash
uv run scripts/build.py --tests
```

### Writing Tests

Add the file under the module's test directory (`tests/<module>/`) and list it
in that directory's `CMakeLists.txt`. The
provider takes its transport from `ProviderOptions`, so a test can script
responses instead of hitting the API:

```cpp
#include <gtest/gtest.h>
#include <llm/openai.h>

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
## Code Quality

### Formatting Standards

We use clang-format with Google C++ style. Key guidelines:

- **Indentation**: 2 spaces
- **Line Length**: 80 characters
- **Naming**: CamelCase for types, snake_case for variables/functions
- **Headers**: Include guards and proper ordering

### Linting Rules

We use clang-tidy with these key checks:

- **Modernize**: Use modern C++ features (auto, nullptr, etc.)
- **Performance**: Avoid unnecessary copies, use move semantics
- **Readability**: Clear naming and structure
- **Safety**: Avoid common pitfalls and unsafe patterns

### Pre-commit Workflow

Before committing code:

```bash
# 1. Format code
uv run scripts/format.py

# 2. Build and test
uv run scripts/build.py --mode debug --tests

# 3. Run linting
uv run scripts/build.py --export-compile-commands  # Ensure compile_commands.json exists
uv run scripts/lint.py

# 4. Run tests
cd build && ctest
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
`llm::ILogger` injected via `SessionConfig::logger` and
`ProviderOptions::logger` (see the README). The SDK ships no implementation —
`tests/support/` has the two the tests use, and `console_logger_demo` prints a
whole decision round:

```bash
ctest -R console_logger_demo -V          # the event stream, with output
uv run scripts/build.py --tests          # then: ./build/tests/integration/console_logger_demo
```

Log records carry classifications and counts (kind, status, http code, attempts,
tokens, tool counts, schema summaries). Provider and model text is never logged:
it stays in `CompletionResult` for the caller.

## Dependencies

### Core Dependencies

The public headers depend on nothing but the standard library; these are
implementation details of the shipped components:

- **nlohmann-json**: request/response JSON and schema validation (an existing
  `nlohmann_json::nlohmann_json` target is reused; otherwise `find_package()`,
  otherwise a pinned upstream fetch by CMake). Private to `llm::tools` and
  `llm::openai`;
- **cpp-httplib**: HTTP client for the transport (vendored header-only, private
  to `llm::openai`);
- **openssl**: TLS for HTTPS endpoints (system-provided; only `llm::openai`
  needs it, and the installed target exports it for static linking)

Neither nlohmann-json nor cpp-httplib appears in a public header, so a
consumer of the installed package only needs OpenSSL and Threads.

### Test Dependencies

Automatically fetched when tests are enabled:

- **gtest**: Google's C++ testing framework (fetched from GitHub by CMake)

### Development Dependencies

Python tools (automatically managed by uv):

- **click**: Modern CLI framework
- **rich**: Rich terminal output
- **asyncio**: Async support for parallel operations

## Contributing

### Code Style

We follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) with these key principles:

1. **One class per file**: Keep classes focused and maintainable
2. **Immutable by default**: Prefer const and immutable designs
3. **Modern C++**: Use C++20 features, auto, smart pointers
4. **RAII**: Resource management through object lifetime
5. **Move semantics**: Prefer move over copy for performance

### Modern C++ Guidelines

- Use `auto` for obvious types
- Use `nullptr` instead of `NULL` or `0`
- Prefer smart pointers (`std::unique_ptr`, `std::shared_ptr`) over raw pointers
- Use move semantics with `std::move`
- Mark functions `const`, `constexpr`, `noexcept` when appropriate
- Use brace initialization `{}` instead of parentheses
- Capture lambda variables explicitly, avoid `[=]` and `[&]`
- Use `std::atomic` for thread-safe operations, not `volatile`
- Prefer `emplace` over `insert`/`push_back`

### Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes following our code style
4. Run the pre-commit workflow (format, build, test, lint)
5. Commit with descriptive messages
6. Push to your fork: `git push origin feature/amazing-feature`
7. Create a Pull Request
