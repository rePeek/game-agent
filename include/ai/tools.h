#pragma once

/// Tool vocabulary of the legacy ai:: API.
///
/// The SDK describes the tools a request may use and reports the calls the
/// model made. It does not execute them: dispatch belongs to the caller, which
/// owns the game state, permissions and side effects.
#include "types/generate_options.h"
#include "types/message.h"
#include "types/tool.h"

namespace ai {}  // namespace ai
