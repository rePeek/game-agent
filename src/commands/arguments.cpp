#include "llm/commands/arguments.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

namespace llm {

struct CommandArguments::Impl {
  bool parsed = false;
  std::string error;
  nlohmann::json object = nlohmann::json::object();
};

CommandArguments::CommandArguments() : impl_(std::make_unique<Impl>()) {}

CommandArguments::~CommandArguments() = default;

CommandArguments::CommandArguments(CommandArguments&&) noexcept = default;

CommandArguments& CommandArguments::operator=(CommandArguments&&) noexcept =
    default;

bool CommandArguments::valid() const {
  return impl_ != nullptr && impl_->parsed;
}

const std::string& CommandArguments::error() const {
  static const std::string kEmpty;
  return impl_ != nullptr ? impl_->error : kEmpty;
}

CommandArguments ParseCommandArguments(std::string_view arguments_json) {
  CommandArguments args;
  args.impl_->object = nlohmann::json::parse(arguments_json, nullptr, false);
  if (args.impl_->object.is_discarded()) {
    args.impl_->error = "arguments are not valid JSON";
    return args;
  }
  if (!args.impl_->object.is_object()) {
    args.impl_->error = "arguments must be a JSON object";
    return args;
  }
  args.impl_->parsed = true;
  return args;
}

namespace {

/// Looks a property up. Returns nullptr when it is absent or null, which the
/// callers treat as "not supplied" so required checks are handled in one place.
const nlohmann::json* Find(const nlohmann::json& object,
                           const std::string& name) {
  const auto it = object.find(name);
  if (it == object.end() || it->is_null()) {
    return nullptr;
  }
  return &(*it);
}

bool TypeMismatch(const std::string& name,
                  const char* expected,
                  std::string& error) {
  error = "argument '" + name + "' must be " + expected;
  return false;
}

}  // namespace

bool ReadArgumentAs(const CommandArguments& args,
                    const std::string& name,
                    bool& out,
                    bool& present,
                    std::string& error) {
  present = false;
  if (!args.valid()) {
    error = args.error();
    return false;
  }
  const nlohmann::json* value = Find(args.impl_->object, name);
  if (value == nullptr) {
    return true;
  }
  if (!value->is_boolean()) {
    return TypeMismatch(name, "a boolean", error);
  }
  out = value->get<bool>();
  present = true;
  return true;
}

bool ReadArgumentAs(const CommandArguments& args,
                    const std::string& name,
                    std::int64_t& out,
                    bool& present,
                    std::string& error) {
  present = false;
  if (!args.valid()) {
    error = args.error();
    return false;
  }
  const nlohmann::json* value = Find(args.impl_->object, name);
  if (value == nullptr) {
    return true;
  }
  if (!value->is_number_integer()) {
    return TypeMismatch(name, "an integer", error);
  }
  try {
    out = value->get<std::int64_t>();
  } catch (const nlohmann::json::exception&) {
    error = "argument '" + name + "' is out of range for an integer";
    return false;
  }
  present = true;
  return true;
}

bool ReadArgumentAs(const CommandArguments& args,
                    const std::string& name,
                    double& out,
                    bool& present,
                    std::string& error) {
  present = false;
  if (!args.valid()) {
    error = args.error();
    return false;
  }
  const nlohmann::json* value = Find(args.impl_->object, name);
  if (value == nullptr) {
    return true;
  }
  if (!value->is_number()) {
    return TypeMismatch(name, "a number", error);
  }
  out = value->get<double>();
  present = true;
  return true;
}

bool ReadArgumentAs(const CommandArguments& args,
                    const std::string& name,
                    std::string& out,
                    bool& present,
                    std::string& error) {
  present = false;
  if (!args.valid()) {
    error = args.error();
    return false;
  }
  const nlohmann::json* value = Find(args.impl_->object, name);
  if (value == nullptr) {
    return true;
  }
  if (!value->is_string()) {
    return TypeMismatch(name, "a string", error);
  }
  out = value->get<std::string>();
  present = true;
  return true;
}

bool CheckArgumentConstraints(const CommandArguments& args,
                              const ParameterDescriptor& descriptor,
                              std::string& error) {
  if (!args.valid()) {
    error = args.error();
    return false;
  }
  const std::string& name = descriptor.name;
  const nlohmann::json* value = Find(args.impl_->object, name);
  if (value == nullptr) {
    // Absent or explicitly null: only a required parameter is a violation.
    if (descriptor.required) {
      error = "missing required argument '" + name + "'";
      return false;
    }
    return true;
  }
  const nlohmann::json& v = *value;

  switch (descriptor.type) {
    case CommandValueType::Boolean:
      if (!v.is_boolean()) {
        return TypeMismatch(name, "a boolean", error);
      }
      break;
    case CommandValueType::Integer:
      if (!v.is_number_integer()) {
        return TypeMismatch(name, "an integer", error);
      }
      break;
    case CommandValueType::Number:
      if (!v.is_number()) {
        return TypeMismatch(name, "a number", error);
      }
      break;
    case CommandValueType::String:
      if (!v.is_string()) {
        return TypeMismatch(name, "a string", error);
      }
      break;
    case CommandValueType::Array:
      if (!v.is_array()) {
        return TypeMismatch(name, "an array", error);
      }
      break;
    case CommandValueType::Object:
      if (!v.is_object()) {
        return TypeMismatch(name, "an object", error);
      }
      break;
  }

  if (v.is_number()) {
    const double number = v.get<double>();
    if (descriptor.minimum.has_value() && number < *descriptor.minimum) {
      error = "argument '" + name + "' must be at least " +
              std::to_string(*descriptor.minimum);
      return false;
    }
    if (descriptor.maximum.has_value() && number > *descriptor.maximum) {
      error = "argument '" + name + "' must be at most " +
              std::to_string(*descriptor.maximum);
      return false;
    }
  }
  if (v.is_string()) {
    const auto length = static_cast<std::int64_t>(v.get<std::string>().size());
    if (descriptor.min_length.has_value() && length < *descriptor.min_length) {
      error = "argument '" + name + "' is too short";
      return false;
    }
    if (descriptor.max_length.has_value() && length > *descriptor.max_length) {
      error = "argument '" + name + "' is too long";
      return false;
    }
  }
  if (v.is_array()) {
    const auto size = static_cast<std::int64_t>(v.size());
    if (descriptor.min_items.has_value() && size < *descriptor.min_items) {
      error = "argument '" + name + "' has too few items";
      return false;
    }
    if (descriptor.max_items.has_value() && size > *descriptor.max_items) {
      error = "argument '" + name + "' has too many items";
      return false;
    }
  }

  if (!descriptor.enum_json.empty()) {
    bool allowed = false;
    for (const std::string& text : descriptor.enum_json) {
      const nlohmann::json candidate =
          nlohmann::json::parse(text, nullptr, false);
      if (!candidate.is_discarded() && candidate == v) {
        allowed = true;
        break;
      }
    }
    if (!allowed) {
      error = "argument '" + name + "' is not one of the allowed values";
      return false;
    }
  }
  return true;
}

}  // namespace llm
