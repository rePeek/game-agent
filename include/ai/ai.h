#pragma once

/// Main convenience header for AI SDK C++
/// Include this header to get access to all public APIs

// Provider-specific clients (conditionally included)
#ifdef AI_SDK_HAS_OPENAI
#include "openai.h"
#endif

// Type definitions
#include "types/client.h"
#include "types/enums.h"
#include "types/generate_options.h"
#include "types/message.h"
#include "types/model.h"
#include "types/tool.h"
#include "types/usage.h"

// Tool functionality
#include "tools.h"

// Error handling
#include "errors.h"

/// AI SDK C++ - Modern C++ toolkit for AI-powered applications
///
/// Usage Examples:
///
/// OpenAI Integration:
/// ```cpp
/// #include <ai/ai.h>
/// #include <iostream>
///
/// // Ensure OPENAI_API_KEY environment variable is set
/// auto client = ai::openai::create_client();
///
/// ai::GenerateOptions options(ai::openai::models::kGpt56,
///                             "Why is the sky blue?");
/// options.system = "You are a friendly assistant!";
/// auto result = client.generate_text(options);
///
/// if (result) {
///     std::cout << result->text << std::endl;
/// }
/// ```
namespace ai {}
