#include "openai_client.h"

#include "ai/logger.h"
#include "ai/openai.h"
#include "openai_request_builder.h"
#include "openai_response_parser.h"

#include <algorithm>
#include <memory>

namespace ai {
namespace openai {

OpenAIClient::OpenAIClient(const std::string& api_key,
                           const std::string& base_url)
    : BaseProviderClient(
          providers::ProviderConfig{
              .api_key = api_key,
              .base_url = base_url,
              .completions_endpoint_path = "/v1/chat/completions",
              .auth_header_name = "Authorization",
              .auth_header_prefix = "Bearer ",
              .extra_headers = {},
              .retry_config = std::nullopt},
          std::make_unique<OpenAIRequestBuilder>(),
          std::make_unique<OpenAIResponseParser>()) {
  ai::logger::log_debug("OpenAI client initialized with base_url: {}",
                        base_url);
}

OpenAIClient::OpenAIClient(const std::string& api_key,
                           const std::string& base_url,
                           const retry::RetryConfig& retry_config)
    : BaseProviderClient(
          providers::ProviderConfig{
              .api_key = api_key,
              .base_url = base_url,
              .completions_endpoint_path = "/v1/chat/completions",
              .auth_header_name = "Authorization",
              .auth_header_prefix = "Bearer ",
              .extra_headers = {},
              .retry_config = retry_config},
          std::make_unique<OpenAIRequestBuilder>(),
          std::make_unique<OpenAIResponseParser>()) {
  ai::logger::log_debug(
      "OpenAI client initialized with base_url: {} and custom retry config",
      base_url);
}

std::string OpenAIClient::provider_name() const {
  return "openai";
}

std::vector<std::string> OpenAIClient::supported_models() const {
  return {// Current GPT-5 series
          models::kGpt56, models::kGpt56Sol, models::kGpt56Terra,
          models::kGpt56Luna, models::kGpt55, models::kGpt54, models::kGpt54Pro,
          models::kGpt54Mini, models::kGpt54Nano, models::kGpt5Mini,
          models::kGpt5Nano,
          // Current GPT-4.1 series
          models::kGpt41, models::kGpt41Mini};
}

bool OpenAIClient::supports_model(const std::string& model_name) const {
  auto models = supported_models();
  return std::find(models.begin(), models.end(), model_name) != models.end();
}

std::string OpenAIClient::config_info() const {
  return "OpenAI API (base_url: " + config_.base_url + ")";
}

std::string OpenAIClient::default_model() const {
  return models::kDefaultModel;
}

}  // namespace openai
}  // namespace ai
