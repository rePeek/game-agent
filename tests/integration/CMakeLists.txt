# Cross-component tests: Session + Context + the OpenAI provider over a
# scripted transport. They assert the wire bodies the SDK builds for
# consecutive decision rounds, so no network and no API key are involved.

add_llm_test(llm_integration_tests llm_session_openai_test.cpp)
target_link_libraries(llm_integration_tests
    PRIVATE
        llm::sdk
        nlohmann_json::nlohmann_json
)
