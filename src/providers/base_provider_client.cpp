#include "base_provider_client.h"

#include "ai/logger.h"

namespace ai {
namespace providers {

BaseProviderClient::BaseProviderClient(
    const ProviderConfig& config,
    std::unique_ptr<RequestBuilder> request_builder,
    std::unique_ptr<ResponseParser> response_parser)
    : config_(config),
      request_builder_(std::move(request_builder)),
      response_parser_(std::move(response_parser)) {
  // Initialize HTTP handler with parsed config
  auto http_config = http::HttpRequestHandler::parse_base_url(config.base_url);

  // Apply custom retry config if provided
  if (config.retry_config.has_value()) {
    http_config.retry_config = config.retry_config.value();
  }

  http_handler_ = std::make_unique<http::HttpRequestHandler>(http_config);

  ai::logger::log_debug(
      "BaseProviderClient initialized - base_url: {}, completions_endpoint: {}",
      config.base_url, config.completions_endpoint_path);
}

/// One completion request in, one parsed response out. The SDK never runs the
/// tools the model asked for: the caller decides what to execute.
GenerateResult BaseProviderClient::generate_text(
    const GenerateOptions& options) {
  ai::logger::log_debug(
      "Starting text generation - model: {}, prompt length: {}, tools: {}",
      options.model, options.prompt.length(), options.tools.size());
  return generate_text_single_step(options);
}

GenerateResult BaseProviderClient::generate_text_single_step(
    const GenerateOptions& options) {
  try {
    // Build request JSON using the provider-specific builder
    auto request_json = request_builder_->build_request_json(options);
    std::string json_body = request_json.dump();
    ai::logger::log_debug("Request JSON built: {}", json_body);

    // Build headers
    auto headers = request_builder_->build_headers(config_);

    // Make the request
    auto result = http_handler_->post(config_.completions_endpoint_path,
                                      headers, json_body);

    if (!result.is_success()) {
      // Parse error response using provider-specific parser
      if (result.provider_metadata.has_value()) {
        int status_code = std::stoi(result.provider_metadata.value());
        return response_parser_->parse_error_completion_response(
            status_code, result.error.value_or(""));
      }
      return result;
    }

    // Parse the response JSON from result.text
    nlohmann::json json_response;
    try {
      json_response = nlohmann::json::parse(result.text);
    } catch (const nlohmann::json::exception& e) {
      ai::logger::log_error("Failed to parse response JSON: {}", e.what());
      ai::logger::log_debug("Raw response text: {}", result.text);
      return GenerateResult("Failed to parse response: " +
                            std::string(e.what()));
    }

    ai::logger::log_info(
        "Text generation successful - model: {}, response_id: {}",
        options.model, json_response.value("id", "unknown"));

    // Parse using provider-specific parser
    auto parsed_result =
        response_parser_->parse_success_completion_response(json_response);

    // The parsed tool calls are handed back untouched: no tool runs here, and
    // nothing is appended to the conversation.
    if (parsed_result.has_tool_calls()) {
      ai::logger::log_debug("Model asked for {} tool calls",
                            parsed_result.tool_calls.size());
    }

    return parsed_result;
  } catch (const std::exception& e) {
    ai::logger::log_error("Exception during text generation: {}", e.what());
    return GenerateResult(std::string("Exception: ") + e.what());
  }
}
}  // namespace providers
}  // namespace ai
