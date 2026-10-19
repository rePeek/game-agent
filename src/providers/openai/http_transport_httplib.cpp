#include "http_transport.h"

#include <httplib.h>
#include <memory>
#include <string>

namespace llm {
namespace {

struct ParsedUrl {
  std::string scheme;
  std::string host;
  int port = 0;
  std::string path;
};

bool ParseUrl(const std::string& url, ParsedUrl& parsed, std::string& error) {
  std::string rest;
  if (url.starts_with("https://")) {
    parsed.scheme = "https";
    parsed.port = 443;
    rest = url.substr(8);
  } else if (url.starts_with("http://")) {
    parsed.scheme = "http";
    parsed.port = 80;
    rest = url.substr(7);
  } else {
    error = "url must start with http:// or https://";
    return false;
  }

  const std::size_t slash = rest.find('/');
  std::string authority =
      slash == std::string::npos ? rest : rest.substr(0, slash);
  parsed.path = slash == std::string::npos ? "/" : rest.substr(slash);

  const std::size_t colon = authority.rfind(':');
  if (colon != std::string::npos) {
    try {
      parsed.port = std::stoi(authority.substr(colon + 1));
    } catch (const std::exception&) {
      error = "invalid port in url";
      return false;
    }
    authority = authority.substr(0, colon);
  }

  parsed.host = authority;
  if (parsed.host.empty()) {
    error = "url has no host";
    return false;
  }
  if (parsed.path.empty()) {
    parsed.path = "/";
  }
  return true;
}

TransportError Classify(httplib::Error error) {
  switch (error) {
    case httplib::Error::ConnectionTimeout:
      return TransportError::Timeout;
    case httplib::Error::Connection:
    case httplib::Error::ProxyConnection:
      return TransportError::Connection;
    case httplib::Error::SSLConnection:
    case httplib::Error::SSLLoadingCerts:
    case httplib::Error::SSLServerVerification:
    case httplib::Error::SSLServerHostnameVerification:
      return TransportError::Tls;
    case httplib::Error::Read:
    case httplib::Error::Write:
      return TransportError::Io;
    default:
      return TransportError::Other;
  }
}

class HttpsTransport : public ITransport {
 public:
  HttpResponse Send(const HttpRequest& request) override {
    HttpResponse response;

    ParsedUrl parsed;
    if (!ParseUrl(request.url, parsed, response.error)) {
      response.error_kind = TransportError::Other;
      return response;
    }

    httplib::Headers headers;
    for (const auto& header : request.headers) {
      headers.emplace(header.name, header.value);
    }

    try {
      // A client per call: httplib::Client is not thread-safe, and concurrent
      // completions must not share connection state.
      if (parsed.scheme == "https") {
        httplib::SSLClient client(parsed.host, parsed.port);
        client.enable_server_certificate_verification(true);
        return Perform(client, request, parsed.path, headers);
      }
      httplib::Client client(parsed.host, parsed.port);
      return Perform(client, request, parsed.path, headers);
    } catch (const std::exception& e) {
      response.error = e.what();
      response.error_kind = TransportError::Other;
      return response;
    }
  }

 private:
  template <typename ClientType>
  static HttpResponse Perform(ClientType& client,
                              const HttpRequest& request,
                              const std::string& path,
                              const httplib::Headers& headers) {
    HttpResponse response;

    client.set_connection_timeout(request.timeout_seconds, 0);
    client.set_read_timeout(request.timeout_seconds, 0);
    client.set_write_timeout(request.timeout_seconds, 0);

    httplib::Result result;
    if (request.method == "POST") {
      result = client.Post(path, headers, request.body, "application/json");
    } else if (request.method == "GET") {
      result = client.Get(path, headers);
    } else {
      response.error = "unsupported method: " + request.method;
      response.error_kind = TransportError::Other;
      return response;
    }

    if (!result) {
      response.error_kind = Classify(result.error());
      response.error = httplib::to_string(result.error());
      if (response.error.empty()) {
        response.error = "no HTTP response";
      }
      return response;
    }

    response.transport_ok = true;
    response.status = result->status;
    response.body = result->body;
    return response;
  }
};

}  // namespace

std::shared_ptr<ITransport> CreateHttpsTransport() {
  return std::make_shared<HttpsTransport>();
}

}  // namespace llm
