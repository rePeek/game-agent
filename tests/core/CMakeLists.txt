# Core tests: the protocol types, and a target that links nothing else.

add_llm_test(llm_core_tests
    llm_core_test.cpp
    llm_logger_test.cpp
)
target_link_libraries(llm_core_tests PRIVATE llm::core)

# ---------------------------------------------------------------------------
# The standard-library-only check.
#
# This executable includes only the core headers and links only llm::core: no
# JSON, no HTTP, no TLS, not even a test framework. If the core ever grows a
# third-party dependency, this target stops building.
# ---------------------------------------------------------------------------
add_executable(llm_core_smoke llm_core_smoke.cpp)
target_link_libraries(llm_core_smoke PRIVATE llm::core)
target_compile_options(llm_core_smoke PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME llm_core_smoke COMMAND llm_core_smoke)
