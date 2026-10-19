# Agent tests: decisions, staging, correlation and isolation. The provider is
# scripted in the tests themselves, so this binary links llm::context (which
# contains the Agent) and no concrete provider.

add_llm_test(llm_agent_tests
    llm_agent_test.cpp
    llm_agent_async_test.cpp
    llm_agent_logging_test.cpp
    llm_agent_commands_test.cpp
    llm_agent_prepare_test.cpp
)
target_link_libraries(llm_agent_tests PRIVATE llm::context)
