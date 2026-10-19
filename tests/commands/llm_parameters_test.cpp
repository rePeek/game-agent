// Parameter DSL tests: descriptors, generated schemas and typed decoding. The
// generated schema is checked through the validator rather than by parsing JSON
// here, so this binary needs no JSON library.

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <llm/commands/arguments.hpp>
#include <llm/commands/parameter.hpp>
#include <llm/commands/validation.h>

namespace llm {
namespace {

CommandDefinition DefinitionWith(const std::string& name,
                                 const std::vector<ParameterDescriptor>& p) {
  CommandDefinition definition;
  definition.name = name;
  definition.description = name;
  definition.parameters_schema_json = BuildCommandSchema(p);
  definition.strict = true;
  return definition;
}

ModelRequest RequestWith(std::vector<CommandDefinition> commands) {
  ModelRequest request;
  request.model = "test-model";
  request.messages = {Message::User("decide")};
  request.commands = std::move(commands);
  return request;
}

TEST(LlmParametersTest, DescriptorsCarryTypeNameAndConstraints) {
  const auto params =
      Params(Param<int>("intensity")
                 .Description("Pressing intensity")
                 .Range(0, 5)
                 .Required(),
             Param<std::string>("formation")
                 .OneOf("4-3-3", "4-4-2", "3-5-2")
                 .Optional(),
             Param<double>("risk"),
             Param<bool>("aggressive"));

  const std::vector<ParameterDescriptor> descriptors = params.Descriptors();
  ASSERT_EQ(descriptors.size(), 4u);

  EXPECT_EQ(descriptors[0].name, "intensity");
  EXPECT_EQ(descriptors[0].type, CommandValueType::Integer);
  EXPECT_TRUE(descriptors[0].required);
  ASSERT_TRUE(descriptors[0].minimum.has_value());
  ASSERT_TRUE(descriptors[0].maximum.has_value());
  EXPECT_DOUBLE_EQ(*descriptors[0].minimum, 0.0);
  EXPECT_DOUBLE_EQ(*descriptors[0].maximum, 5.0);

  EXPECT_EQ(descriptors[1].type, CommandValueType::String);
  EXPECT_FALSE(descriptors[1].required);
  EXPECT_EQ(descriptors[1].enum_json.size(), 3u);

  EXPECT_EQ(descriptors[2].type, CommandValueType::Number);
  EXPECT_EQ(descriptors[3].type, CommandValueType::Boolean);
}

TEST(LlmParametersTest, DefaultsImplyOptionalAndAreSerialized) {
  const auto params = Params(Param<int>("level").Default(2));
  const ParameterDescriptor descriptor = params.Descriptors()[0];
  EXPECT_FALSE(descriptor.required);
  ASSERT_TRUE(descriptor.default_json.has_value());
  EXPECT_EQ(*descriptor.default_json, "2");
}

TEST(LlmParametersTest, GeneratedSchemaIsOneTheValidatorCanEnforce) {
  const auto params =
      Params(Param<int>("intensity").Range(0, 5).Required(),
             Param<std::string>("formation")
                 .OneOf("4-3-3", "4-4-2")
                 .Optional());

  const std::string schema = BuildCommandSchema(params.Descriptors());
  EXPECT_NE(schema.find("\"required\":[\"intensity\"]"), std::string::npos)
      << schema;
  EXPECT_NE(schema.find("\"minimum\":0"), std::string::npos) << schema;
  EXPECT_NE(schema.find("\"maximum\":5"), std::string::npos) << schema;
  EXPECT_NE(schema.find("\"enum\":[\"4-3-3\",\"4-4-2\"]"), std::string::npos)
      << schema;
  EXPECT_NE(schema.find("\"additionalProperties\":false"), std::string::npos)
      << schema;

  // A schema the SDK generates must be one the SDK validator accepts.
  const ModelRequest request =
      RequestWith({DefinitionWith("set_pressing", params.Descriptors())});
  const JsonSchemaCommandValidator validator;
  const ValidationReport report = validator.ValidateCommandSchemas(request);
  EXPECT_TRUE(report.ok) << report.Summary();

  // ... and it must enforce the declared rules.
  EXPECT_TRUE(validator
                  .Validate(request, {CommandCall{"c1", "set_pressing",
                                                  R"({"intensity":4})"}})
                  .ok);
  EXPECT_FALSE(validator
                   .Validate(request, {CommandCall{"c1", "set_pressing",
                                                   R"({"intensity":9})"}})
                   .ok);
  EXPECT_FALSE(validator
                   .Validate(request, {CommandCall{"c1", "set_pressing",
                                                   R"({})"}})
                   .ok);
  EXPECT_FALSE(
      validator
          .Validate(request, {CommandCall{"c1", "set_pressing",
                                          R"({"intensity":1,"formation":"x"})"}})
          .ok);
}

TEST(LlmParametersTest, DecodesTypedArguments) {
  const CommandArguments args = ParseCommandArguments(
      R"({"intensity":4,"risk":0.5,"formation":"4-4-2","aggressive":true})");
  ASSERT_TRUE(args.valid());
  EXPECT_TRUE(args.error().empty());

  int intensity = 0;
  bool present = false;
  std::string error;

  ASSERT_TRUE(ReadArgument(args, "intensity", intensity, present, error))
      << error;
  EXPECT_TRUE(present);
  EXPECT_EQ(intensity, 4);

  std::string formation;
  ASSERT_TRUE(ReadArgument(args, "formation", formation, present, error));
  EXPECT_TRUE(present);
  EXPECT_EQ(formation, "4-4-2");

  double risk = 0.0;
  ASSERT_TRUE(ReadArgument(args, "risk", risk, present, error));
  EXPECT_TRUE(present);
  EXPECT_DOUBLE_EQ(risk, 0.5);

  bool aggressive = false;
  ASSERT_TRUE(ReadArgument(args, "aggressive", aggressive, present, error));
  EXPECT_TRUE(present);
  EXPECT_TRUE(aggressive);

  // An absent optional parameter is not an error; presence says so.
  ASSERT_TRUE(ReadArgument(args, "missing", intensity, present, error));
  EXPECT_FALSE(present);
}

TEST(LlmParametersTest, RejectsMismatchedTypesAndMalformedArguments) {
  const CommandArguments args =
      ParseCommandArguments(R"({"intensity":"high","risk":1})");
  ASSERT_TRUE(args.valid());

  int intensity = 0;
  bool present = false;
  std::string error;
  EXPECT_FALSE(ReadArgument(args, "intensity", intensity, present, error));
  EXPECT_NE(error.find("must be an integer"), std::string::npos);

  // An integer is not a floating point value in the other direction either.
  double risk = 0.0;
  ASSERT_TRUE(ReadArgument(args, "risk", risk, present, error));
  EXPECT_DOUBLE_EQ(risk, 1.0);

  const CommandArguments not_an_object = ParseCommandArguments("[1,2,3]");
  EXPECT_FALSE(not_an_object.valid());
  EXPECT_FALSE(not_an_object.error().empty());

  const CommandArguments not_json = ParseCommandArguments("{oops");
  EXPECT_FALSE(not_json.valid());
  EXPECT_FALSE(ReadArgument(not_json, "x", intensity, present, error));
}

TEST(LlmParametersTest, ValueTypeNamesMatchJsonSchema) {
  EXPECT_STREQ(ToString(CommandValueType::Integer), "integer");
  EXPECT_STREQ(ToString(CommandValueType::String), "string");
  EXPECT_STREQ(ToString(CommandValueType::Boolean), "boolean");
  EXPECT_STREQ(ToString(CommandValueType::Number), "number");
  EXPECT_STREQ(ToString(CommandValueType::Array), "array");
  EXPECT_STREQ(ToString(CommandValueType::Object), "object");
}

}  // namespace
}  // namespace llm
