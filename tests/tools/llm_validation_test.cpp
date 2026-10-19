// Unit tests for the tool-call validator: name, count, id, JSON shape and
// argument schemas.

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <llm/tools/validation.h>

namespace llm {
namespace {

CommandDefinition Tool(const std::string& name, const std::string& schema) {
  CommandDefinition tool;
  tool.name = name;
  tool.description = name + " description";
  tool.parameters_schema_json = schema;
  return tool;
}

ModelRequest RequestWith(std::vector<CommandDefinition> tools) {
  ModelRequest request;
  request.model = "test-model";
  request.messages = {Message::User("decide")};
  request.commands = std::move(tools);
  return request;
}

const char* kPressingSchema = R"({
  "type": "object",
  "properties": {"level": {"type": "integer", "minimum": 1, "maximum": 5}},
  "required": ["level"],
  "additionalProperties": false
})";

const char* kLineSchema = R"({
  "type": "object",
  "properties": {
    "height": {"type": "string", "enum": ["high", "medium", "low"]},
    "width": {"type": "integer"}
  },
  "required": ["height"],
  "additionalProperties": false
})";

ModelRequest DefaultRequest() {
  return RequestWith({Tool("set_pressing_intensity", kPressingSchema),
                      Tool("set_defensive_line", kLineSchema)});
}

ValidationReport Validate(const ModelRequest& request,
                          const std::vector<CommandCall>& calls,
                          ValidationLimits limits = {}) {
  const JsonSchemaCommandValidator validator(limits);
  return validator.Validate(request, calls);
}

TEST(LlmValidationTest, AcceptsAValidBatch) {
  const std::vector<CommandCall> calls = {
      CommandCall{"call_1", "set_pressing_intensity", R"({"level": 3})"},
      CommandCall{"call_2", "set_defensive_line", R"({"height":"high"})"},
  };

  const ValidationReport report = Validate(DefaultRequest(), calls);

  EXPECT_TRUE(report.ok) << report.Summary();
  EXPECT_TRUE(report.issues.empty());
}

TEST(LlmValidationTest, RejectsUnknownTools) {
  const ValidationReport report = Validate(
      DefaultRequest(), {CommandCall{"call_1", "delete_everything", R"({})"}});

  EXPECT_FALSE(report.ok);
  ASSERT_EQ(report.issues.size(), 1u);
  EXPECT_EQ(report.issues[0].command_name, "delete_everything");
  EXPECT_NE(report.issues[0].ToString().find("not one of the commands"),
            std::string::npos);
}

TEST(LlmValidationTest, EnforcesTheBatchSizeWindow) {
  const std::vector<CommandCall> two = {
      CommandCall{"call_1", "set_pressing_intensity", R"({"level":1})"},
      CommandCall{"call_2", "set_defensive_line", R"({"height":"low"})"},
  };

  // "at least one" by default: an empty batch is not a success.
  EXPECT_FALSE(Validate(DefaultRequest(), {}).ok);

  // A caller that wants at least two.
  ValidationLimits two_minimum;
  two_minimum.min_command_calls = 2;
  EXPECT_TRUE(Validate(DefaultRequest(), two, two_minimum).ok);
  EXPECT_FALSE(
      Validate(DefaultRequest(),
               {CommandCall{"call_1", "set_pressing_intensity", R"({"level":1})"}},
               two_minimum)
          .ok);

  // Upper bound from the request itself.
  ModelRequest capped = DefaultRequest();
  capped.max_command_calls = 1;
  const ValidationReport report = Validate(capped, two);
  EXPECT_FALSE(report.ok);
  EXPECT_NE(report.Summary().find("at most 1"), std::string::npos);

  // ... or from the limits, which win over the request.
  ValidationLimits one_maximum;
  one_maximum.max_command_calls = 1;
  EXPECT_FALSE(Validate(DefaultRequest(), two, one_maximum).ok);
}

TEST(LlmValidationTest, RequiresIdsAndToolNames) {
  EXPECT_FALSE(
      Validate(DefaultRequest(),
               {CommandCall{"", "set_pressing_intensity", R"({"level":1})"}})
          .ok);
  EXPECT_FALSE(
      Validate(DefaultRequest(), {CommandCall{"call_1", "", R"({})"}}).ok);
}

TEST(LlmValidationTest, RequiresParseableObjectArguments) {
  EXPECT_FALSE(Validate(DefaultRequest(),
                        {CommandCall{"call_1", "set_pressing_intensity", "{oops"}})
                   .ok);
  EXPECT_FALSE(Validate(DefaultRequest(),
                        {CommandCall{"call_1", "set_pressing_intensity", "[1,2]"}})
                   .ok);
  EXPECT_FALSE(Validate(DefaultRequest(),
                        {CommandCall{"call_1", "set_pressing_intensity", "null"}})
                   .ok);
}

TEST(LlmValidationTest, ValidatesTypesAndRequiredProperties) {
  // level must be an integer
  const ValidationReport wrong_type = Validate(
      DefaultRequest(),
      {CommandCall{"call_1", "set_pressing_intensity", R"({"level":"high"})"}});
  EXPECT_FALSE(wrong_type.ok);
  EXPECT_NE(wrong_type.Summary().find("level must be 'integer'"),
            std::string::npos);

  // level is required
  const ValidationReport missing = Validate(
      DefaultRequest(), {CommandCall{"call_1", "set_pressing_intensity", "{}"}});
  EXPECT_FALSE(missing.ok);
  EXPECT_NE(missing.Summary().find("missing required property 'level'"),
            std::string::npos);
}

TEST(LlmValidationTest, ValidatesEnumAndAdditionalProperties) {
  const ValidationReport bad_enum = Validate(
      DefaultRequest(),
      {CommandCall{"call_1", "set_defensive_line", R"({"height":"sideways"})"}});
  EXPECT_FALSE(bad_enum.ok);
  EXPECT_NE(bad_enum.Summary().find("not one of the allowed values"),
            std::string::npos);

  const ValidationReport extra_key =
      Validate(DefaultRequest(), {CommandCall{"call_1", "set_pressing_intensity",
                                           R"({"level":1,"urgency":"now"})"}});
  EXPECT_FALSE(extra_key.ok);
  EXPECT_NE(extra_key.Summary().find("unexpected property 'urgency'"),
            std::string::npos);
}

TEST(LlmValidationTest, ValidatesNumericAndStringBounds) {
  const ValidationReport too_low = Validate(
      DefaultRequest(),
      {CommandCall{"call_1", "set_pressing_intensity", R"({"level":0})"}});
  EXPECT_FALSE(too_low.ok);
  EXPECT_NE(too_low.Summary().find("level must be >= 1"), std::string::npos);

  const ValidationReport too_high = Validate(
      DefaultRequest(),
      {CommandCall{"call_1", "set_pressing_intensity", R"({"level":9})"}});
  EXPECT_FALSE(too_high.ok);
  EXPECT_NE(too_high.Summary().find("level must be <= 5"), std::string::npos);

  const ModelRequest request = RequestWith({Tool(
      "note",
      R"({"type":"object","properties":{"text":{"type":"string","minLength":3,"maxLength":5}},"required":["text"]})")});
  EXPECT_TRUE(
      Validate(request, {CommandCall{"call_1", "note", R"({"text":"abc"})"}}).ok);
  EXPECT_FALSE(
      Validate(request, {CommandCall{"call_1", "note", R"({"text":"ab"})"}}).ok);
  EXPECT_FALSE(
      Validate(request, {CommandCall{"call_1", "note", R"({"text":"abcdef"})"}})
          .ok);
}

TEST(LlmValidationTest, ValidatesNestedObjectsListsAndAlternatives) {
  const ModelRequest request = RequestWith({Tool("match_context",
                                                 R"({
        "type": "object",
        "properties": {
          "board": {
            "type": "object",
            "properties": {"pressing": {"type": "integer"}},
            "required": ["pressing"],
            "additionalProperties": false
          },
          "players": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {"type": "string"}
          },
          "reason": {
            "anyOf": [{"type": "string"}, {"type": "null"}]
          }
        },
        "required": ["board", "players"],
        "additionalProperties": false
      })")});

  const ValidationReport ok = Validate(
      request,
      {CommandCall{"call_1", "match_context",
                R"({"board":{"pressing":3},"players":["a"],"reason":null})"}});
  EXPECT_TRUE(ok.ok) << ok.Summary();

  const ValidationReport nested = Validate(
      request, {CommandCall{"call_1", "match_context",
                         R"({"board":{"pressing":"high"},"players":["a"]})"}});
  EXPECT_FALSE(nested.ok);
  EXPECT_NE(nested.Summary().find("board.pressing must be 'integer'"),
            std::string::npos);

  const ValidationReport list =
      Validate(request, {CommandCall{"call_1", "match_context",
                                  R"({"board":{"pressing":3},"players":[]})"}});
  EXPECT_FALSE(list.ok);
  EXPECT_NE(list.Summary().find("players needs at least 1 items"),
            std::string::npos);

  const ValidationReport items = Validate(
      request, {CommandCall{"call_1", "match_context",
                         R"({"board":{"pressing":3},"players":[7]})"}});
  EXPECT_FALSE(items.ok);
  EXPECT_NE(items.Summary().find("players[0] must be 'string'"),
            std::string::npos);

  const ValidationReport any_of = Validate(
      request,
      {CommandCall{"call_1", "match_context",
                R"({"board":{"pressing":3},"players":["a"],"reason":5})"}});
  EXPECT_FALSE(any_of.ok);
  EXPECT_NE(any_of.Summary().find("matches no allowed shape"),
            std::string::npos);
}

TEST(LlmValidationTest, LimitsCanBeRelaxed) {
  ValidationLimits permissive;
  permissive.min_command_calls = 0;
  permissive.require_ids = false;
  permissive.require_known_commands = false;
  permissive.validate_schema = false;

  // Unknown tool, no id, arguments that are not an object: only the JSON
  // requirement is left, because the caller turned the rest off.
  const ValidationReport report =
      Validate(DefaultRequest(), {CommandCall{"", "whatever", "[1]"}}, permissive);
  EXPECT_FALSE(report.ok);

  ValidationLimits only_json;
  only_json.min_command_calls = 0;
  only_json.require_ids = false;
  only_json.require_known_commands = false;
  only_json.require_arguments_object = false;
  only_json.validate_schema = false;
  EXPECT_TRUE(
      Validate(DefaultRequest(), {CommandCall{"", "whatever", "[1]"}}, only_json)
          .ok);
}

TEST(LlmValidationTest, ReportsUnusableToolSchema) {
  const ModelRequest request = RequestWith({Tool("broken", "{not json")});

  const ValidationReport report =
      Validate(request, {CommandCall{"call_1", "broken", R"({})"}});

  EXPECT_FALSE(report.ok);
  EXPECT_NE(report.Summary().find("schema is not valid JSON"),
            std::string::npos);
}

TEST(LlmValidationTest, SummaryListsEveryIssueWithPosition) {
  const ValidationReport report =
      Validate(DefaultRequest(),
               {CommandCall{"call_1", "set_pressing_intensity", R"({"level":0})"},
                CommandCall{"call_2", "nope", R"({})"}});

  EXPECT_FALSE(report.ok);
  ASSERT_EQ(report.issues.size(), 2u);
  EXPECT_EQ(report.issues[0].index, 0u);
  EXPECT_EQ(report.issues[1].index, 1u);
  EXPECT_NE(report.Summary().find("command call #0 (set_pressing_intensity)"),
            std::string::npos);
  EXPECT_NE(report.Summary().find("command call #1 (nope)"), std::string::npos);
}

TEST(LlmValidationTest, DefaultValidatorIsSharedAndUsable) {
  const std::shared_ptr<const CommandValidator> validator = DefaultValidator();
  ASSERT_NE(validator, nullptr);
  EXPECT_EQ(validator, DefaultValidator());

  const ValidationReport report = validator->Validate(
      DefaultRequest(),
      {CommandCall{"call_1", "set_pressing_intensity", R"({"level":2})"}});
  EXPECT_TRUE(report.ok) << report.Summary();
}

// ---------------------------------------------------------------------------
// CTX-0: a schema the validator cannot enforce is refused, not waved through
// ---------------------------------------------------------------------------
TEST(LlmValidationTest, UnknownSchemaTypeIsASchemaIssue) {
  const ModelRequest request = RequestWith({Tool(
      "odd",
      R"({"type":"object","properties":{"when":{"type":"timestamp"}}})")});

  const ValidationReport schemas =
      Validate(request, {CommandCall{"call_1", "odd", R"({"when":"now"})"}},
               ValidationLimits{});

  // Argument-level validation stays quiet: 'timestamp' simply is not checked.
  EXPECT_FALSE(schemas.ok) << schemas.Summary();
  EXPECT_TRUE(schemas.HasSchemaIssues());
}

TEST(LlmValidationTest, UnsupportedKeywordsAreReportedNotIgnored) {
  struct Case {
    const char* schema;
    const char* keyword;
  };
  const Case cases[] = {
      {R"({"type":"object","properties":{"name":{"type":"string","pattern":"^a"}}})",
       "pattern"},
      {R"({"type":"object","properties":{"when":{"type":"string","format":"date"}}})",
       "format"},
      {R"({"type":"object","properties":{"n":{"type":"number","multipleOf":2}}})",
       "multipleOf"},
      {R"({"type":"object","properties":{"n":{"not":{"type":"null"}}}})",
       "not"},
      {R"({"type":"object","properties":{"a":{"$ref":"#/definitions/x"}}})",
       "$ref"},
  };

  const JsonSchemaCommandValidator validator;
  for (const Case& test_case : cases) {
    const ModelRequest request = RequestWith({Tool("t", test_case.schema)});
    const ValidationReport report = validator.ValidateCommandSchemas(request);
    EXPECT_FALSE(report.ok) << test_case.keyword;
    EXPECT_TRUE(report.HasSchemaIssues()) << test_case.keyword;
    EXPECT_NE(report.Summary().find(test_case.keyword), std::string::npos)
        << report.Summary();
    // The provider refuses such a request before sending it, which is what
    // makes the keyword check worth having separately from the batch check.
  }
}

TEST(LlmValidationTest, AnnotationsAreAllowed) {
  const ModelRequest request = RequestWith({Tool(
      "t",
      R"({"type":"object","title":"Params","description":"d","properties":{"level":{"type":"integer","description":"1-5","default":1,"examples":[3]}},"required":["level"],"$comment":"fine"})")});

  const ValidationReport report =
      Validate(request, {CommandCall{"call_1", "t", R"({"level":3})"}});

  EXPECT_TRUE(report.ok) << report.Summary();
  EXPECT_FALSE(report.HasSchemaIssues());
}

TEST(LlmValidationTest, ValidateCommandSchemasChecksWithoutABatch) {
  const JsonSchemaCommandValidator validator;

  const ValidationReport ok = validator.ValidateCommandSchemas(DefaultRequest());
  EXPECT_TRUE(ok.ok) << ok.Summary();

  const ValidationReport broken = validator.ValidateCommandSchemas(RequestWith({Tool(
      "t",
      R"({"type":"object","properties":{"a":{"type":"string","pattern":"x"}}})")}));
  EXPECT_FALSE(broken.ok);
  EXPECT_TRUE(broken.HasSchemaIssues());
  EXPECT_NE(broken.Summary().find("pattern"), std::string::npos);

  // A schema that is not even JSON is refused here too.
  const ValidationReport not_json =
      validator.ValidateCommandSchemas(RequestWith({Tool("t", "{oops")}));
  EXPECT_FALSE(not_json.ok);
  EXPECT_TRUE(not_json.HasSchemaIssues());
}

TEST(LlmValidationTest, DeeplyNestedSchemasAreRefusedNotSkipped) {
  // Built by wrapping, so the result is balanced by construction.
  std::string schema = R"({"type":"integer"})";
  constexpr int kDepth = 40;
  for (int i = 0; i < kDepth; ++i) {
    schema = R"({"type":"object","properties":{"a":)" + schema + "}}";
  }

  const ModelRequest request = RequestWith({Tool("deep", schema)});
  const JsonSchemaCommandValidator validator;
  const ValidationReport report = validator.ValidateCommandSchemas(request);

  EXPECT_FALSE(report.ok) << report.Summary();
  EXPECT_TRUE(report.HasSchemaIssues());
  EXPECT_NE(report.Summary().find("deeper than"), std::string::npos)
      << report.Summary();
}

TEST(LlmValidationTest, DuplicateCommandCallIdsAreRejected) {
  const std::vector<CommandCall> calls = {
      CommandCall{"call_1", "set_pressing_intensity", R"({"level":1})"},
      CommandCall{"call_1", "set_defensive_line", R"({"height":"low"})"},
  };

  const ValidationReport report = Validate(DefaultRequest(), calls);

  EXPECT_FALSE(report.ok);
  EXPECT_NE(report.Summary().find("reuses command call id"), std::string::npos);

  // Callers can ask for the check to be relaxed, but not by accident.
  ValidationLimits relaxed;
  relaxed.require_unique_ids = false;
  EXPECT_TRUE(Validate(DefaultRequest(), calls, relaxed).ok);
}

TEST(LlmValidationTest, KnownKeywordsWithTheWrongValueShapeAreRefused) {
  const JsonSchemaCommandValidator validator;

  // An 'enum' that is not an array would otherwise be silently ignored, and
  // Success would claim a check that never happened.
  const ValidationReport bad_enum = validator.ValidateCommandSchemas(RequestWith(
      {Tool("t", R"({"type":"object","properties":{"a":{"enum":"x"}}})")}));
  EXPECT_FALSE(bad_enum.ok);
  EXPECT_TRUE(bad_enum.HasSchemaIssues());
  EXPECT_NE(bad_enum.Summary().find("non-empty array"), std::string::npos);

  // A bound expressed as a string cannot be enforced numerically.
  const ValidationReport bad_minimum = validator.ValidateCommandSchemas(RequestWith(
      {Tool("t",
            R"({"type":"object","properties":{"a":{"type":"integer","minimum":"1"}}})")}));
  EXPECT_FALSE(bad_minimum.ok);
  EXPECT_NE(bad_minimum.Summary().find("must be a number"),
            std::string::npos);

  // minItems must be a non-negative integer, not a string or a float.
  const ValidationReport bad_min_items = validator.ValidateCommandSchemas(RequestWith(
      {Tool("t",
            R"({"type":"object","properties":{"a":{"type":"array","minItems":"1"}}})")}));
  EXPECT_FALSE(bad_min_items.ok);
  EXPECT_NE(bad_min_items.Summary().find("non-negative integer"),
            std::string::npos);

  // required must be an array of names.
  const ValidationReport bad_required = validator.ValidateCommandSchemas(RequestWith(
      {Tool("t", R"({"type":"object","required":"a"})")}));
  EXPECT_FALSE(bad_required.ok);
  EXPECT_NE(bad_required.Summary().find("array of property names"),
            std::string::npos);
}

TEST(LlmValidationTest, DuplicateCommandDefinitionsAreRefused) {
  const JsonSchemaCommandValidator validator;
  const ModelRequest request =
      RequestWith({Tool("set_pressing_intensity", kPressingSchema),
                   Tool("set_pressing_intensity", kPressingSchema)});

  const ValidationReport report = validator.ValidateCommandSchemas(request);
  EXPECT_FALSE(report.ok);
  EXPECT_TRUE(report.HasSchemaIssues());
  EXPECT_NE(report.Summary().find("more than once"), std::string::npos)
      << report.Summary();
}


}  // namespace
}  // namespace llm
