#pragma once

/// Command-call validation.
///
/// CompletionStatus::Success means "every command call in this batch passed the
/// checks below". The checks are deliberately explicit rather than implied:
/// a provider that cannot enforce something says so (see
/// ProviderCapabilities), and a schema this validator cannot enforce is
/// refused instead of waved through.
///
/// The public surface stays standard-library-only: arguments and schemas are
/// JSON text here, and the JSON parser lives in the implementation.

#include "llm/core/request.h"
#include "llm/core/types.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace llm {

enum class ValidationIssueKind {
  /// The model's output does not satisfy the request (bad type, missing
  /// property, unknown command, duplicate id, ...).
  Arguments,
  /// The request itself cannot be checked: the command schema uses a
  /// constraint this validator does not implement. Refusing is the honest
  /// answer, because Success would otherwise claim a guarantee that was never
  /// verified.
  Schema,
};

struct ValidationIssue {
  std::size_t index = 0;  // position in the batch
  std::string command_name;
  ValidationIssueKind kind = ValidationIssueKind::Arguments;
  std::string message;

  /// "command call #1 (set_pressing_intensity): level must be an integer"
  std::string ToString() const;
};

/// Which checks a validator applies. The defaults are the SDK's contract: at
/// least one call, only known commands, unique ids, parseable arguments and a
/// matching schema.
struct ValidationLimits {
  std::size_t min_command_calls = 1;
  /// Overrides the request's own max_command_calls when set.
  std::optional<std::size_t> max_command_calls;
  bool require_ids = true;
  bool require_unique_ids = true;
  bool require_known_commands = true;
  bool require_arguments_object = true;
  bool validate_schema = true;
};

struct ValidationReport {
  bool ok = false;
  std::vector<ValidationIssue> issues;

  /// True when at least one issue is about the request rather than the model's
  /// output. Providers report those as Unsupported.
  bool HasSchemaIssues() const;

  /// All issues joined into one line, for error messages.
  std::string Summary() const;
};

class CommandValidator {
 public:
  virtual ~CommandValidator() = default;

  /// Checks the request's command schemas *before* anything is sent: every
  /// constraint present in a schema must be one this validator implements. An
  /// empty report means "these schemas can be enforced".
  virtual ValidationReport ValidateCommandSchemas(
      const ModelRequest& request) const {
    (void)request;
    return ValidationReport{true, {}};
  }

  /// Validates a batch against the request that produced it.
  virtual ValidationReport Validate(
      const ModelRequest& request,
      const std::vector<CommandCall>& command_calls) const = 0;
};

/// Validates arguments against the command's JSON Schema.
///
/// Supported constraints: type (single or union), enum, const, properties,
/// required, additionalProperties, items with minItems/maxItems, minimum,
/// maximum, minLength, maxLength and anyOf/oneOf/allOf. Anything else
/// (`pattern`, `format`, `multipleOf`, `$ref`, `not`, ...) is reported as a
/// schema issue instead of being ignored.
class JsonSchemaCommandValidator : public CommandValidator {
 public:
  explicit JsonSchemaCommandValidator(ValidationLimits limits = {});

  ValidationReport ValidateCommandSchemas(
      const ModelRequest& request) const override;

  ValidationReport Validate(
      const ModelRequest& request,
      const std::vector<CommandCall>& command_calls) const override;

 private:
  ValidationLimits limits_;
};

/// The validator the built-in providers use when none is injected.
std::shared_ptr<const CommandValidator> DefaultValidator();

}  // namespace llm
