#pragma once

/// The typed parameter DSL a game uses to register a command.
///
/// A game describes each parameter once, with a `Param<T>`, and the SDK derives
/// everything else from that description: the JSON Schema it offers the model,
/// the validation the provider applies, and the C++ argument decoding used when
/// a returned call is bound to the callback.
///
///     agent.RegisterCommand(
///         "set_pressing",
///         "Change team pressing intensity",
///         llm::Params(llm::Param<int>("intensity")
///                         .Description("Pressing intensity")
///                         .Range(0, 5)
///                         .Required()),
///         [&coach](int intensity) { return coach.SetPressing(intensity); });
///
/// `Param<T>` carries the C++ type, so `RegisterCommand` can check at compile
/// time that the callback actually accepts the parameters being described.
/// This header stays standard-library-only: the JSON library lives in the
/// implementation (src/commands/schema.cpp, src/commands/arguments.cpp).

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace llm {

/// The JSON/parameter kinds this SDK can describe and decode.
enum class CommandValueType {
  Boolean,
  Integer,
  Number,
  String,
  Array,
  Object,
};

/// The JSON spelling of a value type, as used in generated schemas.
const char* ToString(CommandValueType type);

/// Non-template description of one command parameter.
///
/// This is the single source of truth: both the generated JSON Schema and the
/// argument decoder are derived from it, so a parameter can never be described
/// one way for the model and handled another way at execution time.
struct ParameterDescriptor {
  std::string name;
  CommandValueType type = CommandValueType::String;
  std::string description;
  bool required = false;
  /// Default as JSON text (already serialized), when one was declared.
  std::optional<std::string> default_json;

  // Numeric bounds (Integer/Number only).
  std::optional<double> minimum;
  std::optional<double> maximum;

  // String bounds (String only).
  std::optional<std::int64_t> min_length;
  std::optional<std::int64_t> max_length;

  // Array bounds (Array only).
  std::optional<std::int64_t> min_items;
  std::optional<std::int64_t> max_items;

  // Allowed values, each already serialized as JSON text.
  std::vector<std::string> enum_json;

  bool IsValid() const { return !name.empty(); }
};

/// Escapes a string so it can be embedded in JSON text (RFC 8259).
inline std::string EscapeJsonString(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 2);
  for (const char raw : value) {
    const unsigned char byte = static_cast<unsigned char>(raw);
    switch (byte) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (byte < 0x20) {
          static const char kHex[] = "0123456789abcdef";
          out += "\\u00";
          out += kHex[(byte >> 4) & 0x0f];
          out += kHex[byte & 0x0f];
        } else {
          out += static_cast<char>(byte);
        }
        break;
    }
  }
  return out;
}

/// Serializes the parameter types this SDK supports to JSON text.
template <typename T>
std::string CommandJsonText(const T& value) {
  if constexpr (std::is_same_v<std::decay_t<T>, bool>) {
    return value ? "true" : "false";
  } else if constexpr (std::is_same_v<std::decay_t<T>, std::string>) {
    return "\"" + EscapeJsonString(value) + "\"";
  } else if constexpr (std::is_convertible_v<T, std::string>) {
    return "\"" + EscapeJsonString(std::string(value)) + "\"";
  } else if constexpr (std::is_integral_v<std::decay_t<T>>) {
    return std::to_string(value);
  } else if constexpr (std::is_floating_point_v<std::decay_t<T>>) {
    return std::to_string(value);
  } else {
    static_assert(sizeof(T) == 0,
                  "unsupported command parameter type: use bool, an integer, "
                  "a floating point type or std::string");
  }
}

/// Maps a supported C++ parameter type to its JSON kind.
template <typename T>
constexpr CommandValueType CommandTypeOf() {
  using Value = std::decay_t<T>;
  if constexpr (std::is_same_v<Value, bool>) {
    return CommandValueType::Boolean;
  } else if constexpr (std::is_integral_v<Value>) {
    return CommandValueType::Integer;
  } else if constexpr (std::is_floating_point_v<Value>) {
    return CommandValueType::Number;
  } else if constexpr (std::is_convertible_v<Value, std::string>) {
    return CommandValueType::String;
  } else {
    static_assert(sizeof(T) == 0,
                  "unsupported command parameter type: use bool, an integer, "
                  "a floating point type or std::string");
  }
}

/// One parameter of a command, typed by the C++ value the callback takes.
///
/// The fluent setters return `*this` so a declaration reads top to bottom. A
/// setter that does not apply to the parameter's type (for example Range on a
/// string) is disabled at compile time rather than silently ignored.
template <typename T>
class Param {
 public:
  explicit Param(std::string name) : name_(std::move(name)) {}

  Param& Description(std::string text) {
    description_ = std::move(text);
    return *this;
  }

  Param& Required(bool value = true) {
    required_ = value;
    return *this;
  }

  Param& Optional() { return Required(false); }

  Param& Default(T value) {
    default_value_ = std::move(value);
    default_json_ = CommandJsonText(*default_value_);
    required_ = false;
    return *this;
  }

  /// Inclusive numeric bounds.
  template <typename U = T,
            typename = std::enable_if_t<std::is_arithmetic_v<U> &&
                                        !std::is_same_v<U, bool>>>
  Param& Range(T minimum, T maximum) {
    minimum_ = static_cast<double>(minimum);
    maximum_ = static_cast<double>(maximum);
    return *this;
  }

  template <typename U = T,
            typename = std::enable_if_t<std::is_arithmetic_v<U> &&
                                        !std::is_same_v<U, bool>>>
  Param& Min(T value) {
    minimum_ = static_cast<double>(value);
    return *this;
  }

  template <typename U = T,
            typename = std::enable_if_t<std::is_arithmetic_v<U> &&
                                        !std::is_same_v<U, bool>>>
  Param& Max(T value) {
    maximum_ = static_cast<double>(value);
    return *this;
  }

  template <typename U = T,
            typename = std::enable_if_t<std::is_same_v<U, std::string>>>
  Param& MinLength(std::int64_t value) {
    min_length_ = value;
    return *this;
  }

  template <typename U = T,
            typename = std::enable_if_t<std::is_same_v<U, std::string>>>
  Param& MaxLength(std::int64_t value) {
    max_length_ = value;
    return *this;
  }

  /// The allowed values, for a closed set (string enumerations in the first
  /// version; numeric enums work because the values are serialized as JSON).
  template <typename... Values>
  Param& OneOf(Values... values) {
    enum_json_ = {CommandJsonText(T(std::forward<Values>(values)))...};
    return *this;
  }

  const std::string& name() const { return name_; }

  /// The declared default, if any. Used by argument binding.
  const std::optional<T>& default_value() const { return default_value_; }

  ParameterDescriptor ToDescriptor() const {
    ParameterDescriptor descriptor;
    descriptor.name = name_;
    descriptor.type = CommandTypeOf<T>();
    descriptor.description = description_;
    descriptor.required = required_;
    descriptor.default_json = default_json_;
    descriptor.minimum = minimum_;
    descriptor.maximum = maximum_;
    descriptor.min_length = min_length_;
    descriptor.max_length = max_length_;
    descriptor.min_items = min_items_;
    descriptor.max_items = max_items_;
    descriptor.enum_json = enum_json_;
    return descriptor;
  }

 private:
  std::string name_;
  std::string description_;
  bool required_ = false;
  std::optional<std::string> default_json_;
  std::optional<T> default_value_;
  std::optional<double> minimum_;
  std::optional<double> maximum_;
  std::optional<std::int64_t> min_length_;
  std::optional<std::int64_t> max_length_;
  std::optional<std::int64_t> min_items_;
  std::optional<std::int64_t> max_items_;
  std::vector<std::string> enum_json_;
};

/// The ordered parameter list of one command, typed by the callback's
/// parameters.
template <typename... Args>
class CommandParameters {
 public:
  explicit CommandParameters(Param<Args>... params)
      : params_(std::move(params)...) {}

  std::tuple<Param<Args>...> params_;

  /// The declared default of each parameter, in declaration order.
  std::tuple<std::optional<Args>...> Defaults() const {
    return std::apply(
        [](const Param<Args>&... param) {
          return std::make_tuple(param.default_value()...);
        },
        params_);
  }

  std::vector<ParameterDescriptor> Descriptors() const {
    return std::apply(
        [](const Param<Args>&... param) {
          return std::vector<ParameterDescriptor>{param.ToDescriptor()...};
        },
        params_);
  }
};

/// Builds the parameter list a typed command is registered with.
template <typename... Args>
CommandParameters<Args...> Params(Param<Args>... params) {
  return CommandParameters<Args...>(std::move(params)...);
}

/// Renders the JSON Schema for a registered command from its descriptors.
///
/// The schema is the object a provider sends as the command's parameters; it
/// only ever uses constraints the SDK validator implements, so a schema the SDK
/// generates is always one it can enforce. Optional parameters are nullable,
/// and stay out of `required`.
std::string BuildCommandSchema(
    const std::vector<ParameterDescriptor>& parameters);

/// Rewrites a generated schema into OpenAI's strict dialect: every property
/// of every object is added to `required` and `additionalProperties` is
/// disabled. Optional parameters are already nullable, so this is safe.
/// A schema that does not parse is returned unchanged.
std::string ToStrictSchema(const std::string& schema_json);

}  // namespace llm
