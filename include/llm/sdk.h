#pragma once

/// Everything the SDK ships, in one header.
///
/// Convenience only: each module's headers can be included on their own, and
/// llm::core, llm::tools and llm::context pull in no third-party dependency
/// even through this file — the transport interface is standard-library-only
/// and the HTTP implementation stays inside the provider.

// Core: protocol vocabulary and the provider interface.
#include "llm/core/logger.h"
#include "llm/core/message.h"
#include "llm/core/provider.h"
#include "llm/core/request.h"
#include "llm/core/result.h"
#include "llm/core/types.h"

// Tools: schema and batch validation.
#include "llm/tools/validation.h"

// Context: per-session history and asynchronous decisions.
#include "llm/context/context.h"
#include "llm/context/session.h"

// Providers.
#include "llm/providers/openai/openai.h"
#include "llm/providers/openai/transport.h"

namespace llm {}
