#pragma once

#include "llm/core/logger.h"
#include "llm/core/provider.h"
#include "llm/core/types.h"
#include "llm/providers/openai/transport.h"
#include "llm/tools/validation.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace llm {
namespace openai {

namespace models {
// Current GPT-5 series
inline constexpr char kGpt56[] = "gpt-5.6";
inline constexpr char kGpt56Sol[] = "gpt-5.6-sol";
inline constexpr char kGpt56Terra[] = "gpt-5.6-terra";
inline constexpr char kGpt56Luna[] = "gpt-5.6-luna";
inline constexpr char kGpt55[] = "gpt-5.5";
inline constexpr char kGpt54[] = "gpt-5.4";
inline constexpr char kGpt54Pro[] = "gpt-5.4-pro";
inline constexpr char kGpt54Mini[] = "gpt-5.4-mini";
inline constexpr char kGpt54Nano[] = "gpt-5.4-nano";
inline constexpr char kGpt5Mini[] = "gpt-5-mini";
inline constexpr char kGpt5Nano[] = "gpt-5-nano";

// GPT-4.1 series
inline constexpr char kGpt41[] = "gpt-4.1";
inline constexpr char kGpt41Mini[] = "gpt-4.1-mini";

inline constexpr const char* kDefaultModel = kGpt56;
}  // namespace models

/// How often a retryable failure is retried, and how long to wait in between.
struct RetryPolicy {
  int max_attempts = 3;  // 1 disables retrying
  int initial_delay_ms = 500;
  double backoff_factor = 2.0;
  int max_delay_ms = 8000;
};

/// Which Chat Completions parameter dialect an endpoint speaks.
enum class ParameterDialect {
  /// Pick from the model id (see DialectForModel).
  Auto,
  /// `max_tokens` plus temperature/top_p/seed (gpt-4.x and older).
  LegacyTokens,
  /// `max_completion_tokens` and no sampling controls (the gpt-5 and
  /// o-series reasoning families reject them on Chat Completions).
  ReasoningTokens,
};

/// The dialect assumed for a model id when ParameterDialect::Auto is used.
ParameterDialect DialectForModel(const std::string& model);

/// Capabilities of OpenAI's own API. Pointing this provider at an
/// OpenAI-compatible server (OpenRouter, a local gateway) means overriding
/// them: a server that ignores `tool_choice: "required"` or `strict` schemas
/// must not be credited with enforcing them.
constexpr ProviderCapabilities DefaultCapabilities() {
  ProviderCapabilities capabilities;
  capabilities.multiple_command_calls = true;
  capabilities.forced_command_choice = true;
  capabilities.strict_command_schema = true;
  capabilities.command_result_history = true;
  return capabilities;
}

struct ProviderOptions {
  std::string api_key;
  std::string base_url = "https://api.openai.com";
  std::string organization;               // optional OpenAI-Organization header
  std::vector<HttpHeader> extra_headers;  // e.g. OpenRouter attribution headers
  int timeout_seconds = 60;
  RetryPolicy retry;

  ProviderCapabilities capabilities = DefaultCapabilities();

  /// Override the token-limit/sampling parameter mapping for an endpoint that
  /// differs from OpenAI's own model families.
  ParameterDialect parameter_dialect = ParameterDialect::Auto;

  /// When a request demands something the provider cannot enforce (a forced
  /// tool choice, a strict schema), report Unsupported by default. Setting this
  /// lets the call continue with the SDK's own validation instead — never with
  /// a claim that the provider enforced anything.
  bool allow_validation_downgrade = false;

  /// Defaults to the cpp-httplib transport; tests inject a fake.
  std::shared_ptr<ITransport> transport;

  /// Defaults to the JSON-Schema validator. Injecting one lets an application
  /// add its own policy (for example a per-game allowed tool list) without
  /// touching the provider.
  std::shared_ptr<const CommandValidator> validator;

  /// Optional diagnostics sink; the same logger can be shared with the
  /// sessions using this provider. Without it the provider stays silent.
  std::shared_ptr<ILogger> logger;
};

/// OpenAI-compatible Chat Completions provider.
///
/// One Complete() call performs exactly one (retried) HTTP request and returns
/// the parsed result. It never executes tools and never appends anything to the
/// request's context.
class OpenAIProvider : public IModelProvider {
 public:
  explicit OpenAIProvider(ProviderOptions options);

  std::string Name() const override { return "openai"; }

  ProviderCapabilities Capabilities() const override {
    return options_.capabilities;
  }

  CompletionResult Complete(const ModelRequest& request) override;

 private:
  ProviderOptions options_;
};

/// Convenience factory.
std::shared_ptr<IModelProvider> CreateProvider(ProviderOptions options);

/// Reads OPENAI_API_KEY and OPENAI_BASE_URL. The optional logger is shared by
/// every session using this provider.
std::shared_ptr<IModelProvider> CreateProviderFromEnvironment(
    std::shared_ptr<ILogger> logger = nullptr);

}  // namespace openai
}  // namespace llm
