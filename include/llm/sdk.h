#pragma once

/// Everything the SDK ships, in one header.
///
/// Convenience only: each module's headers can be included on their own, and
/// llm::core, llm::commands and llm::context pull in no third-party dependency
/// even through this file — the transport interface is standard-library-only
/// and the HTTP implementation stays inside the provider.

// The agent: one decision stream, its context and its provider.
#include "llm/agent.hpp"

// Core: protocol vocabulary and the provider interface.
#include "llm/core/logger.h"
#include "llm/core/message.h"
#include "llm/core/provider.h"
#include "llm/core/request.h"
#include "llm/core/result.h"
#include "llm/core/types.h"

// Commands: typed registration, schema generation and batch validation.
#include "llm/commands/validation.h"

#include "llm/commands/arguments.hpp"
#include "llm/commands/parameter.hpp"

// Context: the history the agent projects into a request.
#include "llm/context/context.h"

// Providers.
#include "llm/providers/openai/openai.h"
#include "llm/providers/openai/transport.h"

namespace llm {}
