#pragma once

#include "transport.h"

#include <memory>

namespace llm {

/// HTTP/HTTPS transport backed by cpp-httplib, with OpenSSL for TLS.
///
/// One transport instance is safe to share: every Send() builds its own
/// client, so concurrent completions do not share connection state.
std::shared_ptr<ITransport> CreateHttpsTransport();

}  // namespace llm
