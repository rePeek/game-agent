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

bool CommandArguments::valid() const { return impl_ != nullptr && impl_->parsed; }

const std::string& CommandArguments::error() const {
  static const std::string kEmpty;
  return impl_ != nullptr ? impl_->error : kEmpty;
}

CommandArguments ParseCommandArguments(std::string_view arguments_json) {
  CommandArguments args;
  args.impl_->object =
      nlohmann::json::parse(arguments_json, nullptr, false);
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

}  // namespace llm
