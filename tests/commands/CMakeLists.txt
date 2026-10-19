# Command tests: parameter descriptors, schema generation, argument decoding and
# batch validation. JSON stays inside llm::commands: this binary only needs the
# public headers.

add_llm_test(llm_commands_tests
    llm_validation_test.cpp
    llm_parameters_test.cpp
    llm_registry_test.cpp
)
target_link_libraries(llm_commands_tests PRIVATE llm::commands)
