#pragma once

/// Tool-call validation.
///
/// CompletionStatus::Success means "every tool call in this batch passed the
/// checks below". The checks are deliberately explicit rather than implied:
/// a provider that cannot enforce something says so (see
/// ProviderCapabilities), and the SDK verifies what it promised.
///
/// The public surface stays standard-library-only: arguments and schemas are
/// JSON text here, and the JSON parser lives in the implementation.

#include "request.h"
#include "types.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace llm {

struct ValidationIssue {
  std::size_t index = 0;  // position in the batch
  std::string tool_name;
  std::string message;

  /// "tool call #1 (set_pressing_intensity): level must be an integer"
  std::string ToString() const;
};

/// Which checks a validator applies. The defaults are the SDK's contract: at
/// least one call, only known tools, ids, parseable arguments and a matching
/// schema.
struct ValidationLimits {
  std::size_t min_tool_calls = 1;
  /// Overrides the request's own max_tool_calls when set.
  std::optional<std::size_t> max_tool_calls;
  bool require_ids = true;
  bool require_known_tools = true;
  bool require_arguments_object = true;
  bool validate_schema = true;
};

struct ValidationReport {
  bool ok = false;
  std::vector<ValidationIssue> issues;

  /// All issues joined into one line, for error messages.
  std::string Summary() const;
};

class ToolCallValidator {
 public:
  virtual ~ToolCallValidator() = default;

  /// Validates a batch against the request that produced it.
  virtual ValidationReport Validate(
      const ModelRequest& request,
      const std::vector<ToolCall>& tool_calls) const = 0;
};

/// Validates arguments against the tool's JSON Schema (the subset generative
/// models actually use: type, enum/const, properties, required,
/// additionalProperties, items, min/max for numbers and strings, and
/// anyOf/oneOf/allOf).
class JsonSchemaToolCallValidator : public ToolCallValidator {
 public:
  explicit JsonSchemaToolCallValidator(ValidationLimits limits = {});

  ValidationReport Validate(
      const ModelRequest& request,
      const std::vector<ToolCall>& tool_calls) const override;

 private:
  ValidationLimits limits_;
};

/// The validator the built-in providers use when none is injected.
std::shared_ptr<const ToolCallValidator> DefaultValidator();

}  // namespace llm
