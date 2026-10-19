# Cross-component tests: Agent + Context + the OpenAI provider over a scripted
# transport. They assert the wire bodies the SDK builds for consecutive decision
# rounds, so no network and no API key are involved.

add_llm_test(llm_integration_tests llm_agent_openai_test.cpp)
target_link_libraries(llm_integration_tests
    PRIVATE
        llm::sdk
        nlohmann_json::nlohmann_json
)

# Manual debugging aid: it prints the SDK's event stream through the test-tree
# ConsoleLogger. It asserts nothing, so it is only there to be watched:
#   ctest -R console_logger_demo -V
add_executable(console_logger_demo console_logger_demo.cpp)
target_link_libraries(console_logger_demo PRIVATE llm::sdk)
target_include_directories(console_logger_demo PRIVATE ${LLM_TEST_SUPPORT_DIR})
set_target_properties(console_logger_demo PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
add_test(NAME console_logger_demo COMMAND console_logger_demo)
