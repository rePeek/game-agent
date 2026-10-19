# Context and Session tests. They drive a scripted IModelProvider that lives in
# the test files, so this binary links llm::context and no provider at all —
# which is exactly the dependency the module promises.

add_llm_test(llm_context_tests
    llm_context_test.cpp
    llm_session_test.cpp
    llm_logging_test.cpp
)
target_link_libraries(llm_context_tests PRIVATE llm::context)
