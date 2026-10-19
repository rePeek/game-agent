# Context tests: the history, its projection and its budget. This binary links
# llm::context and no provider at all — which is exactly the dependency the
# module promises. (The Agent tests live in tests/agent/.)

add_llm_test(llm_context_tests llm_context_test.cpp)
target_link_libraries(llm_context_tests PRIVATE llm::context)
