#include "ai/logger.h"

#include <iostream>

#include <ai/ai.h>

int main() {
  try {
    // Enable debug logging
    ai::logger::install_logger(std::make_shared<ai::logger::ConsoleLogger>(
        ai::logger::LogLevel::kLogLevelInfo));

    // Create OpenAI client
    auto client = ai::openai::create_client();

    // Test simple generation
    std::cout << "Testing OpenAI text generation...\n\n";

    ai::GenerateOptions options(ai::openai::models::kGpt56Terra,
                                "You are a friendly assistant!",
                                "Why is the sky blue? Give a short answer.");

    auto result = client.generate_text(options);

    if (result) {
      std::cout << "Response: " << result.text << "\n";
      std::cout << "Model: " << result.model.value_or("unknown") << "\n";
      std::cout << "Tokens used: " << result.usage.total_tokens << "\n";
      std::cout << "Finish reason: " << result.finishReasonToString() << "\n";
    } else {
      std::cout << "Error: " << result.error_message() << "\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 1;
  }

  return 0;
}