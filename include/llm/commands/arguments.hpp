#pragma once

/// Decoding of a model-produced argument object into typed C++ parameters.
///
/// The model's arguments arrive as JSON text. This layer parses them once into
/// a `CommandArguments` handle, then reads each declared parameter with
/// `ReadArgument<T>`. The public header stays standard-library-only: the JSON
/// parser and the concrete handle live in src/commands/arguments.cpp.

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>

namespace llm {

/// A parsed argument object, opaque to callers.
///
/// It is move-only and cheap: the parsed JSON lives behind a pimpl so this
/// header does not expose the JSON library. `valid()` reports whether the text
/// was a JSON object at all; a missing property is not an error here, only a
/// type mismatch is.
class CommandArguments {
 public:
  CommandArguments();
  ~CommandArguments();
  CommandArguments(CommandArguments&&) noexcept;
  CommandArguments& operator=(CommandArguments&&) noexcept;
  CommandArguments(const CommandArguments&) = delete;
  CommandArguments& operator=(const CommandArguments&) = delete;

  bool valid() const;
  /// Parse failure description, empty when valid().
  const std::string& error() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  friend CommandArguments ParseCommandArguments(std::string_view text);
  friend bool ReadArgumentAs(const CommandArguments& args,
                             const std::string& name,
                             bool& out,
                             bool& present,
                             std::string& error);
  friend bool ReadArgumentAs(const CommandArguments& args,
                             const std::string& name,
                             std::int64_t& out,
                             bool& present,
                             std::string& error);
  friend bool ReadArgumentAs(const CommandArguments& args,
                             const std::string& name,
                             double& out,
                             bool& present,
                             std::string& error);
  friend bool ReadArgumentAs(const CommandArguments& args,
                             const std::string& name,
                             std::string& out,
                             bool& present,
                             std::string& error);
};

/// Parses an argument object. On failure the result is !valid().
CommandArguments ParseCommandArguments(std::string_view arguments_json);

// --- scalar readers (implemented in arguments.cpp) -------------------------
//
// Each returns false only on a type error; `present` distinguishes "absent"
// from "read". Callers that need a required parameter check `present`.
bool ReadArgumentAs(const CommandArguments& args,
                    const std::string& name,
                    bool& out,
                    bool& present,
                    std::string& error);
bool ReadArgumentAs(const CommandArguments& args,
                    const std::string& name,
                    std::int64_t& out,
                    bool& present,
                    std::string& error);
bool ReadArgumentAs(const CommandArguments& args,
                    const std::string& name,
                    double& out,
                    bool& present,
                    std::string& error);
bool ReadArgumentAs(const CommandArguments& args,
                    const std::string& name,
                    std::string& out,
                    bool& present,
                    std::string& error);

/// Type-directed read used by the registration templates.
///
/// Integer parameters are read through std::int64_t and narrowed, so the JSON
/// number must be an integer. Floating point parameters accept any JSON number.
template <typename T>
bool ReadArgument(const CommandArguments& args,
                  const std::string& name,
                  T& out,
                  bool& present,
                  std::string& error) {
  using Value = std::decay_t<T>;
  if constexpr (std::is_same_v<Value, bool>) {
    return ReadArgumentAs(args, name, out, present, error);
  } else if constexpr (std::is_integral_v<Value>) {
    std::int64_t value = 0;
    if (!ReadArgumentAs(args, name, value, present, error)) {
      return false;
    }
    if (present) {
      out = static_cast<T>(value);
    }
    return true;
  } else if constexpr (std::is_floating_point_v<Value>) {
    double value = 0.0;
    if (!ReadArgumentAs(args, name, value, present, error)) {
      return false;
    }
    if (present) {
      out = static_cast<T>(value);
    }
    return true;
  } else if constexpr (std::is_same_v<Value, std::string>) {
    return ReadArgumentAs(args, name, out, present, error);
  } else {
    static_assert(sizeof(T) == 0, "unsupported command parameter type");
  }
}

}  // namespace llm
