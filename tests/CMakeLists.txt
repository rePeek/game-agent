# Test executable for AI SDK C++
add_executable(ai_tests
    unit/llm_core_test.cpp
    unit/llm_context_test.cpp
    unit/llm_session_test.cpp
    unit/llm_openai_test.cpp
    unit/llm_validation_test.cpp

    # Cross-component: Session + Context + the OpenAI provider over a
    # scripted transport (still no network and no API key).
    integration/llm_session_openai_test.cpp
)

# Link against the SDK and the test framework. The internal request builder and
# response parser are exercised through the provider, but the tests also
# include their headers directly (see the source include directory below), which
# is why nlohmann/json and cpp-httplib appear here.
target_link_libraries(ai_tests
    PRIVATE
        llm::core
        llm::openai
        llm::tools
        llm::context
        nlohmann_json::nlohmann_json
        httplib::httplib
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
)

target_include_directories(ai_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src  # Access to the provider's internal headers
)

set_target_properties(ai_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

if(MSVC)
    target_compile_options(ai_tests PRIVATE /W3)
else()
    target_compile_options(ai_tests PRIVATE -Wall -Wextra)
endif()

# Discover tests for CTest
include(GoogleTest)
gtest_discover_tests(ai_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTIES
        TIMEOUT 30
)

# ---------------------------------------------------------------------------
# llm:: core standalone check.
#
# This executable includes only the core headers and links only llm::core: no
# JSON,
# no HTTP, no TLS, not even a test framework. If the core ever grows a
# third-party dependency, this target stops building.
# ---------------------------------------------------------------------------
add_executable(llm_core_smoke standalone/llm_core_smoke.cpp)
target_link_libraries(llm_core_smoke PRIVATE llm::core)
set_target_properties(llm_core_smoke PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
if(NOT MSVC)
    target_compile_options(llm_core_smoke PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME llm_core_smoke COMMAND llm_core_smoke)
