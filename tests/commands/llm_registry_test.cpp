// Command registry tests: registration rules, definitions and argument binding.

#include <functional>
#include <string>

#include <gtest/gtest.h>
#include <llm/commands/registry.hpp>

namespace llm {
namespace {

TEST(LlmRegistryTest, RegistersLooksUpAndRemoves) {
  CommandRegistry registry;
  EXPECT_EQ(registry.version(), 0u);

  const CommandRegistrationResult added = registry.Add(MakeRegisteredCommand(
      "set_pressing", "Pressing intensity",
      Params(Param<int>("level").Range(1, 5).Required()),
      [](int level) -> CommandOutcome {
        return CommandOutcome{true, std::to_string(level), {}};
      }));
  ASSERT_TRUE(added.ok()) << added.error;
  EXPECT_EQ(registry.size(), 1u);
  EXPECT_EQ(registry.version(), 1u);

  const RegisteredCommand* command = registry.Find("set_pressing");
  ASSERT_NE(command, nullptr);
  EXPECT_EQ(command->definition.name, "set_pressing");
  EXPECT_EQ(command->definition.description, "Pressing intensity");
  EXPECT_FALSE(command->definition.parameters_schema_json.empty());
  EXPECT_EQ(command->parameters.size(), 1u);
  EXPECT_TRUE(registry.Contains("set_pressing"));

  const std::vector<CommandDefinition> definitions = registry.Definitions();
  ASSERT_EQ(definitions.size(), 1u);
  EXPECT_EQ(definitions[0].name, "set_pressing");

  EXPECT_TRUE(registry.Remove("set_pressing"));
  EXPECT_FALSE(registry.Remove("set_pressing"));
  EXPECT_FALSE(registry.Contains("set_pressing"));
  EXPECT_EQ(registry.size(), 0u);
  EXPECT_EQ(registry.version(), 2u);
}

TEST(LlmRegistryTest, RefusesEmptyAndDuplicateNames) {
  CommandRegistry registry;
  const auto make = [](const std::string& name) {
    return MakeRegisteredCommand(
        name, "", Params(Param<int>("level").Required()),
        [](int) { return CommandOutcome{true, {}, {}}; });
  };

  EXPECT_EQ(registry.Add(make("")).status, RegistrationStatus::EmptyName);

  ASSERT_TRUE(registry.Add(make("a")).ok());
  EXPECT_EQ(registry.Add(make("a")).status, RegistrationStatus::DuplicateName);
  EXPECT_EQ(registry.size(), 1u);
}

TEST(LlmRegistryTest, RefusesADuplicatedParameterName) {
  CommandRegistry registry;
  const CommandRegistrationResult result = registry.Add(MakeRegisteredCommand(
      "bad", "", Params(Param<int>("level"), Param<std::string>("level")),
      [](int, const std::string&) { return CommandOutcome{true, {}, {}}; }));
  EXPECT_EQ(result.status, RegistrationStatus::InvalidParameter);
  EXPECT_NE(result.error.find("more than once"), std::string::npos);
}

TEST(LlmRegistryTest, BindingHasNoSideEffectsUntilInvoked) {
  int calls = 0;
  CommandRegistry registry;
  ASSERT_TRUE(registry
                  .Add(MakeRegisteredCommand(
                      "set_pressing", "",
                      Params(Param<int>("level").Range(1, 5).Required()),
                      [&calls](int level) -> CommandOutcome {
                        ++calls;
                        return CommandOutcome{true, std::to_string(level), {}};
                      }))
                  .ok());

  const RegisteredCommand* command = registry.Find("set_pressing");
  ASSERT_NE(command, nullptr);
  const CommandBinding binding = command->bind(R"({"level":4})");
  ASSERT_TRUE(binding.ok) << binding.error;
  EXPECT_EQ(calls, 0) << "binding must not run the callback";

  const CommandOutcome outcome = binding.invoke();
  EXPECT_EQ(calls, 1);
  EXPECT_TRUE(outcome.success);
  EXPECT_EQ(outcome.content_json, "4");
}

TEST(LlmRegistryTest, RejectsMissingWrongTypedAndMalformedArguments) {
  int calls = 0;
  CommandRegistry registry;
  ASSERT_TRUE(registry
                  .Add(MakeRegisteredCommand(
                      "set_pressing", "",
                      Params(Param<int>("level").Range(1, 5).Required()),
                      [&calls](int) {
                        ++calls;
                        return CommandOutcome{true, {}, {}};
                      }))
                  .ok());
  const RegisteredCommand* command = registry.Find("set_pressing");

  const CommandBinding missing = command->bind("{}");
  EXPECT_FALSE(missing.ok);
  EXPECT_NE(missing.error.find("missing required argument"), std::string::npos);

  const CommandBinding wrong_type = command->bind(R"({"level":"high"})");
  EXPECT_FALSE(wrong_type.ok);
  EXPECT_NE(wrong_type.error.find("must be an integer"), std::string::npos);

  const CommandBinding malformed = command->bind("{oops");
  EXPECT_FALSE(malformed.ok);

  EXPECT_EQ(calls, 0) << "a rejected binding never runs the callback";
}

TEST(LlmRegistryTest, DefaultsFillAbsentArguments) {
  CommandRegistry registry;
  ASSERT_TRUE(
      registry
          .Add(MakeRegisteredCommand(
              "set_pressing", "", Params(Param<int>("level").Default(3)),
              [](int level) -> CommandOutcome {
                return CommandOutcome{true, std::to_string(level), {}};
              }))
          .ok());
  const RegisteredCommand* command = registry.Find("set_pressing");

  const CommandBinding binding = command->bind("{}");
  ASSERT_TRUE(binding.ok) << binding.error;
  EXPECT_EQ(binding.invoke().content_json, "3");

  const CommandBinding explicit_value = command->bind(R"({"level":5})");
  ASSERT_TRUE(explicit_value.ok) << explicit_value.error;
  EXPECT_EQ(explicit_value.invoke().content_json, "5");
}

TEST(LlmRegistryTest, VoidCallbacksBecomeSuccessfulResultlessCommands) {
  bool ran = false;
  CommandRegistry registry;
  ASSERT_TRUE(registry
                  .Add(MakeRegisteredCommand("no_args", "", Params(),
                                             [&ran]() { ran = true; }))
                  .ok());
  const CommandBinding binding = registry.Find("no_args")->bind("{}");
  ASSERT_TRUE(binding.ok) << binding.error;
  EXPECT_FALSE(ran);
  const CommandOutcome outcome = binding.invoke();
  EXPECT_TRUE(ran);
  EXPECT_TRUE(outcome.success);
  EXPECT_TRUE(outcome.content_json.empty());
}

TEST(LlmRegistryTest, RefusesAnOptionalParameterWithoutADefault) {
  CommandRegistry registry;
  const CommandRegistrationResult result = registry.Add(
      MakeRegisteredCommand("c", "", Params(Param<int>("x")),
                            [](int) { return CommandOutcome{true, {}, {}}; }));
  EXPECT_EQ(result.status, RegistrationStatus::InvalidParameter);
  EXPECT_NE(result.error.find("default"), std::string::npos);
}

TEST(LlmRegistryTest, RefusesInconsistentOrInvalidDeclarations) {
  CommandRegistry registry;

  // minimum > maximum
  EXPECT_EQ(registry
                .Add(MakeRegisteredCommand(
                    "c", "", Params(Param<int>("x").Range(5, 1).Required()),
                    [](int) { return CommandOutcome{true, {}, {}}; }))
                .status,
            RegistrationStatus::InvalidParameter);

  // a default that violates its own range
  EXPECT_EQ(registry
                .Add(MakeRegisteredCommand(
                    "d", "", Params(Param<int>("x").Range(0, 5).Default(99)),
                    [](int) { return CommandOutcome{true, {}, {}}; }))
                .status,
            RegistrationStatus::InvalidParameter);

  // a default outside the allowed set
  EXPECT_EQ(
      registry
          .Add(MakeRegisteredCommand(
              "e", "",
              Params(Param<std::string>("x").OneOf("a", "b").Default("z")),
              [](const std::string&) { return CommandOutcome{true, {}, {}}; }))
          .status,
      RegistrationStatus::InvalidParameter);
}

TEST(LlmRegistryTest, RejectsBindingContractsWhenNoProviderValidated) {
  CommandRegistry registry;
  ASSERT_TRUE(registry
                  .Add(MakeRegisteredCommand(
                      "set_pressing", "",
                      Params(Param<int>("level").Range(1, 5).Required(),
                             Param<std::string>("code")
                                 .OneOf("chasing", "stable")
                                 .Required()),
                      [](int, const std::string&) {
                        return CommandOutcome{true, {}, {}};
                      }))
                  .ok());
  const RegisteredCommand* command = registry.Find("set_pressing");

  const CommandBinding too_high =
      command->bind(R"({"level":9,"code":"chasing"})");
  EXPECT_FALSE(too_high.ok);
  EXPECT_NE(too_high.error.find("at most"), std::string::npos);
  EXPECT_TRUE(command->bind(R"({"level":1,"code":"chasing"})").ok);
  EXPECT_FALSE(command->bind(R"({"level":1,"code":"x"})").ok);
}

TEST(LlmRegistryTest, BindingUsesExactIntegerBoundsAndUnicodeLength) {
  constexpr std::int64_t kMax = 9007199254740993LL;
  CommandRegistry registry;
  ASSERT_TRUE(registry
                  .Add(MakeRegisteredCommand(
                      "big", "",
                      Params(Param<std::int64_t>("v").Range(0, kMax).Required(),
                             Param<std::string>("s").MaxLength(1).Required()),
                      [](std::int64_t, const std::string&) {
                        return CommandOutcome{true, {}, {}};
                      }))
                  .ok());
  const RegisteredCommand* command = registry.Find("big");

  EXPECT_TRUE(command->bind(R"({"v":9007199254740993,"s":"o"})").ok);
  EXPECT_FALSE(command->bind(R"({"v":9007199254740994,"s":"o"})").ok);
  // One Unicode code point is one character, whatever its UTF-8 width.
  EXPECT_TRUE(command->bind(R"({"v":0,"s":"中"})").ok);
  EXPECT_FALSE(command->bind(R"({"v":0,"s":"中文"})").ok);
}

}  // namespace
}  // namespace llm
