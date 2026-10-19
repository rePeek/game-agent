#pragma once

/// Transport seam of the llm:: core.
///
/// Providers depend on this interface rather than on an HTTP library, so a
/// provider can be exercised in tests with a scripted transport and no
/// network. It stays standard-library-only: bodies and headers cross it as
/// strings.

#include <string>
#include <utility>
#include <vector>

namespace llm {

struct HttpHeader {
  std::string name;
  std::string value;
};

struct HttpRequest {
  std::string method = "POST";
  std::string url;
  std::vector<HttpHeader> headers;
  std::string body;
  int timeout_seconds = 60;
};

/// Why a transport call could not produce an HTTP response.
enum class TransportError {
  None,
  Timeout,
  Connection,
  Tls,
  Io,
  Other,
};

struct HttpResponse {
  /// false when no HTTP response was received at all (DNS, TLS, timeout...).
  bool transport_ok = false;
  int status = 0;
  std::string body;
  /// Human-readable transport failure, empty on success.
  std::string error;
  TransportError error_kind = TransportError::None;

  bool IsSuccess() const {
    return transport_ok && status >= 200 && status < 300;
  }
};

class ITransport {
 public:
  virtual ~ITransport() = default;

  /// Sends one request. Must not throw: failures are reported through
  /// HttpResponse::transport_ok == false plus `error`/`error_kind`.
  virtual HttpResponse Send(const HttpRequest& request) = 0;
};

}  // namespace llm
