#pragma once

#include <cstddef>
#include <string_view>

namespace llm {
namespace detail {

/// Number of Unicode code points in a UTF-8 string.
///
/// JSON Schema's `minLength`/`maxLength` count code points, not bytes, so
/// "中" is one character even though it is three UTF-8 bytes. Invalid bytes are
/// counted as one code point each so a malformed string cannot loop forever.
inline std::size_t Utf8Length(std::string_view text) {
  std::size_t count = 0;
  for (const char raw : text) {
    if ((static_cast<unsigned char>(raw) & 0xC0) != 0x80) {
      ++count;  // not a continuation byte: the start of a code point
    }
  }
  return count;
}

}  // namespace detail
}  // namespace llm
