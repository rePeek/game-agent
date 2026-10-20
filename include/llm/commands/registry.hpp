#pragma once

/// The command registry: what a game registers, and how a model call is bound.
///
/// A game registers an existing C++ function (or lambda) together with a typed
/// parameter list:
///
///     agent.RegisterCommand(
///         "set_pressing",
///         "Change team pressing intensity",
///         llm::Params(llm::Param<int>("intensity").Range(0, 5).Required()),
///         [&coach](int intensity) { return coach.SetPressing(intensity); });
///
/// Everything else — the JSON Schema offered to the model, the type check and
/// the argument binding into a zero-argument callable — is derived from that
/// one declaration, so the schema and the decoder can never disagree.

#include "llm/core/request.h"
#include "llm/core/types.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "llm/commands/arguments.hpp"
#include "llm/commands/parameter.hpp"

namespace llm {

/// Why a registration did or did not happen.
enum class RegistrationStatus {
  Registered,
  NotRegistered,
  EmptyName,
  DuplicateName,
  InvalidParameter,
  AgentBusy,
  AgentClosed,
};

struct CommandRegistrationResult {
  RegistrationStatus status = RegistrationStatus::Registered;
  std::string error;

  bool ok() const { return status == RegistrationStatus::Registered; }
};

/// The decoded, still-unexecuted form of one command call: calling `invoke()`
/// runs the game callback with the model's arguments already bound.
struct CommandBinding {
  bool ok = false;
  std::string error;
  std::function<CommandOutcome()> invoke;
};

/// One registered command: the definition offered to the model plus the
/// machinery that turns a model argument object into a bound callable.
struct RegisteredCommand {
  CommandDefinition definition;
  std::vector<ParameterDescriptor> parameters;

  /// Decodes `arguments_json` and binds it to the callback. It never runs the
  /// callback: the returned `invoke` does that when the game chooses to.
  std::function<CommandBinding(const std::string& arguments_json)> bind;
};

namespace command_detail {

/// Calls the callback with the decoded values and normalizes the return type.
///
/// A callback returning CommandOutcome reports its own success/failure; a
/// callback returning void is treated as a successful, resultless command.
/// Anything else is a compile-time error.
template <typename Fn, typename... Args>
CommandOutcome InvokeCallback(Fn& fn, std::tuple<Args...>& values) {
  static_assert(std::is_invocable_v<Fn&, Args...>,
                "the registered callback cannot be called with the parameters "
                "described by CommandParameters");
  using Result = std::invoke_result_t<Fn&, Args...>;
  return std::apply(
      [&fn](Args&... args) -> CommandOutcome {
        if constexpr (std::is_void_v<Result>) {
          fn(args...);
          return CommandOutcome{true, {}, {}};
        } else {
          static_assert(
              std::is_same_v<std::decay_t<Result>, CommandOutcome>,
              "a command callback must return llm::CommandOutcome or void");
          return fn(args...);
        }
      },
      values);
}

template <typename T>
bool DecodeOne(const CommandArguments& args,
               const ParameterDescriptor& descriptor,
               const std::optional<T>& fallback,
               T& value,
               std::string& error) {
  // The declared contract is enforced here, not only by the provider's own
  // schema validation: preparing is the SDK's last gate before a callback.
  if (!CheckArgumentConstraints(args, descriptor, error)) {
    return false;
  }
  bool present = false;
  if (!ReadArgument<T>(args, descriptor.name, value, present, error)) {
    return false;
  }
  if (present) {
    return true;
  }
  if (fallback.has_value()) {
    value = *fallback;
  } else {
    // A non-required parameter must have a default; the registry rejects a
    // definition that does not, so this is unreachable for registered
    // commands and the value stays value-initialized otherwise.
    value = T{};
  }
  return true;
}

template <typename... Args, std::size_t... I>
bool DecodeAll(const CommandArguments& args,
               const std::vector<ParameterDescriptor>& descriptors,
               const std::tuple<std::optional<Args>...>& defaults,
               std::tuple<Args...>& values,
               std::string& error,
               std::index_sequence<I...>) {
  bool ok = true;
  ((ok = ok && DecodeOne<Args>(args, descriptors[I], std::get<I>(defaults),
                               std::get<I>(values), error)),
   ...);
  return ok;
}

}  // namespace command_detail

/// Builds the type-erased registered command for a callback and its typed
/// parameter list. Called by Agent::RegisterCommand.
template <typename Fn, typename... Args>
RegisteredCommand MakeRegisteredCommand(std::string name,
                                        std::string description,
                                        CommandParameters<Args...> parameters,
                                        Fn&& callback) {
  static_assert(std::is_invocable_v<std::decay_t<Fn>&, Args...>,
                "the registered callback cannot be called with the parameters "
                "described by CommandParameters");

  RegisteredCommand command;
  command.parameters = parameters.Descriptors();
  command.definition.name = std::move(name);
  command.definition.description = std::move(description);
  command.definition.parameters_schema_json =
      BuildCommandSchema(command.parameters);
  command.definition.strict = true;

  const std::vector<ParameterDescriptor> descriptors = command.parameters;
  auto defaults = parameters.Defaults();
  command.bind = [callback = std::forward<Fn>(callback), descriptors,
                  defaults =
                      std::move(defaults)](const std::string& arguments_json) {
    CommandBinding binding;
    const CommandArguments args = ParseCommandArguments(arguments_json);
    if (!args.valid()) {
      binding.error = args.error();
      return binding;
    }
    if (descriptors.size() != sizeof...(Args)) {
      binding.error = "internal error: parameter description mismatch";
      return binding;
    }
    std::tuple<Args...> values{};
    if (!command_detail::DecodeAll(args, descriptors, defaults, values,
                                   binding.error,
                                   std::index_sequence_for<Args...>{})) {
      return binding;
    }
    binding.ok = true;
    binding.invoke = [callback, values = std::move(values)]() mutable {
      return command_detail::InvokeCallback<Fn, Args...>(callback, values);
    };
    return binding;
  };
  return command;
}

/// An ordered set of registered commands, addressed by name.
///
/// The registry is deliberately not thread-safe: the Agent owns it and only
/// touches it from its owner thread. `version()` changes whenever the set
/// changes, so a pending decision can detect that the commands it was taken
/// with are no longer the current ones.
class CommandRegistry {
 public:
  /// Adds a command. The command is rejected when its name is empty, already
  /// taken, or when it describes an unusable parameter.
  CommandRegistrationResult Add(RegisteredCommand command);

  /// Removes the named command. Returns false when it was not registered.
  bool Remove(std::string_view name);

  /// Removes every command.
  void Clear();

  bool Contains(std::string_view name) const;
  std::size_t size() const { return commands_.size(); }
  std::uint64_t version() const { return version_; }

  /// The definitions to offer the model, in registration order.
  std::vector<CommandDefinition> Definitions() const;

  /// The registered command, or nullptr. The pointer is invalidated by any
  /// mutation.
  const RegisteredCommand* Find(std::string_view name) const;

 private:
  std::vector<RegisteredCommand> commands_;
  std::uint64_t version_ = 0;
};

}  // namespace llm
