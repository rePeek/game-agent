# Test executable for AI SDK C++
add_executable(ai_tests
    # Unit tests
    unit/llm_core_test.cpp
    unit/llm_openai_test.cpp
    unit/openai_client_test.cpp
    unit/types_test.cpp

    # Integration tests
    integration/openai_integration_test.cpp
    integration/tool_calling_integration_test.cpp

    # Utility classes
    utils/test_fixtures.cpp
)

# Link against the main library and test dependencies
target_link_libraries(ai_tests
    PRIVATE
        ai-sdk-cpp
        llm::core  # the STL-only protocol core
        llm::openai  # provider + transport
        nlohmann_json::nlohmann_json  # assertions on built request JSON
        httplib::httplib  # Need direct access to httplib for internal headers
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
)

# Include directories for tests
target_include_directories(ai_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
        ${CMAKE_SOURCE_DIR}/src  # Access to private headers
)

# Set test properties
set_target_properties(ai_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Compiler options for tests (less strict than main library)
if(MSVC)
    target_compile_options(ai_tests PRIVATE /W3)
else()
    target_compile_options(ai_tests PRIVATE -Wall -Wextra)
endif()

# Define preprocessor macros for tests
target_compile_definitions(ai_tests PRIVATE
    AI_SDK_TESTING=1
    $<$<CONFIG:Debug>:AI_SDK_DEBUG_TESTS=1>
    # httplib configuration (must match main library)
    CPPHTTPLIB_OPENSSL_SUPPORT=1
)

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
# This executable includes only <llm/core.h> and links only llm::core: no JSON,
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
