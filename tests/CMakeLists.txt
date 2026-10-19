# Tests for game-agent.
#
# Each module has its own test binary so the dependency boundary is checked by
# the linker: the core tests link llm::core and nothing else, the context tests
# need no provider and no TLS, and only the provider tests pull in a JSON
# library.
include(GoogleTest)

function(add_llm_test name)
    add_executable(${name} ${ARGN})
    set_target_properties(${name} PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )
    if(MSVC)
        target_compile_options(${name} PRIVATE /W3)
    else()
        target_compile_options(${name} PRIVATE -Wall -Wextra)
    endif()
    target_link_libraries(${name}
        PRIVATE
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
    )
    gtest_discover_tests(${name}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        PROPERTIES
            TIMEOUT 30
    )
endfunction()

add_subdirectory(core)
add_subdirectory(tools)
add_subdirectory(context)
add_subdirectory(providers/openai)

# Cross-component tests: Session + Context + the OpenAI provider, driven through
# a scripted transport (no network, no API key).
add_subdirectory(integration)
