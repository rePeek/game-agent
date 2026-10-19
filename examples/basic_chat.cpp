/**
 * Basic Chat Example - AI SDK C++
 *
 * This example demonstrates basic text generation using the AI SDK.
 * It shows how to:
 * - Use the simple generate_text API
 * - Handle OpenAI-compatible providers
 * - Check for errors and display results
 *
 * Usage:
 *   export OPENAI_API_KEY=your_key_here
 *   ./basic_chat
 */

#include <iostream>
#include <string>

#include <ai/openai.h>

int main() {
  std::cout << "AI SDK C++ - Basic Chat Example\n";
  std::cout << "================================\n\n";

  // Example 1: Simple text generation with OpenAI
  std::cout << "1. Generating text with OpenAI GPT-5.4:\n";
  std::cout << "Question: What is the capital of France?\n\n";

  auto client1 = ai::openai::create_client();
  ai::GenerateOptions options1;
  options1.model = ai::openai::models::kGpt56;
  options1.prompt =
      "What is the capital of France? Please provide a brief answer.";

  auto result1 = client1.generate_text(options1);

  if (result1) {
    std::cout << "Answer: " << result1.text << "\n";
    std::cout << "Usage: " << result1.usage.total_tokens << " tokens\n";
    std::cout << "Finish reason: " << result1.finishReasonToString() << "\n\n";
  } else {
    std::cout << "Error: " << result1.error_message() << "\n\n";
  }

  // Example 2: Text generation with system prompt
  std::cout << "2. Generating text with system prompt:\n";
  std::cout << "System: You are a helpful math tutor who explains concepts "
               "clearly.\n";
  std::cout << "Question: Explain what a prime number is.\n\n";

  ai::GenerateOptions options2;
  options2.model = ai::openai::models::kGpt56;
  options2.system =
      "You are a helpful math tutor who explains concepts clearly.";
  options2.prompt =
      "Explain what a prime number is in simple terms with an example.";

  auto result2 = client1.generate_text(options2);

  if (result2) {
    std::cout << "Answer: " << result2.text << "\n";
    std::cout << "Usage: " << result2.usage.total_tokens << " tokens\n\n";
  } else {
    std::cout << "Error: " << result2.error_message() << "\n\n";
  }

  // Example 3: Conversation with messages
  std::cout << "3. Multi-turn conversation:\n";

  ai::Messages conversation = {
      ai::Message::system("You are a friendly assistant who likes to help with "
                          "coding questions."),
      ai::Message::user(
          "What is the difference between a vector and a list in C++?"),
      ai::Message::assistant(
          "Great question! In C++, std::vector and std::list are both "
          "containers but with different characteristics..."),
      ai::Message::user(
          "Which one should I use for frequent insertions in the middle?")};

  ai::GenerateOptions options3;
  options3.model = ai::openai::models::kGpt56;
  options3.messages = conversation;

  auto result3 = client1.generate_text(options3);

  if (result3) {
    std::cout << "Assistant: " << result3.text << "\n";
    std::cout << "Usage: " << result3.usage.total_tokens << " tokens\n\n";
  } else {
    std::cout << "Error: " << result3.error_message() << "\n\n";
  }

  // Example 4: Using GenerateOptions for more control
  std::cout << "5. Using GenerateOptions for fine control:\n";

  ai::GenerateOptions options;
  options.model = ai::openai::models::kGpt56;
  options.prompt = "List 3 benefits of using C++ for systems programming.";
  options.max_tokens = 150;
  options.temperature = 0.7;

  auto result5 = client1.generate_text(options);

  if (result5) {
    std::cout << "Response: " << result5.text << "\n";
    std::cout << "Usage: " << result5.usage.total_tokens
              << " tokens (max: 150)\n";
    std::cout << "Temperature: 0.7\n\n";
  } else {
    std::cout << "Error: " << result5.error_message() << "\n\n";
  }

  std::cout << "Example completed!\n";
  std::cout << "\nTip: Make sure to set your API keys:\n";
  std::cout << "  export OPENAI_API_KEY=your_openai_key\n";

  return 0;
}