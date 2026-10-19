# Tool validation tests. JSON stays inside llm::tools: this binary only needs
# the public header.

add_llm_test(llm_tools_tests llm_validation_test.cpp)
target_link_libraries(llm_tools_tests PRIVATE llm::tools)
