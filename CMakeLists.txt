cmake_minimum_required(VERSION 3.20)

project(game-agent
    VERSION 0.1.0
    DESCRIPTION "Structured tool-calling C++ SDK for OpenAI-compatible models"
    LANGUAGES CXX
)

# C++20 requirement
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_EXAMPLES "Build example applications" ON)
option(BUILD_SHARED_LIBS "Build shared libraries instead of static ones" OFF)

# Third-party dependencies: a vendored cpp-httplib header, nlohmann/json
# (reused, found or fetched) and GoogleTest when tests are enabled. OpenSSL is
# looked up by the module that needs it (src/providers/openai).
add_subdirectory(third_party)

# ---------------------------------------------------------------------------
# Module helpers
#
# Public headers live in include/, implementation details in src/. A module
# target is created, configured and added to the install list by its own
# src/<module>/CMakeLists.txt.
# ---------------------------------------------------------------------------
function(configure_llm_component target_name)
    target_include_directories(${target_name}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
    )
endfunction()

function(configure_llm_warnings target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4)
        target_compile_definitions(${target_name} PRIVATE
            WIN32_LEAN_AND_MEAN NOMINMAX _WIN32_WINNT=0x0601)
    else()
        target_compile_options(${target_name} PRIVATE
            -Wall -Wextra -Wpedantic
            $<$<CONFIG:Debug>:-g -O0>
            $<$<CONFIG:Release>:-O3>
        )
        if(APPLE)
            target_compile_options(${target_name} PRIVATE -stdlib=libc++)
        endif()
    endif()

    target_compile_definitions(${target_name} PRIVATE
        $<$<CONFIG:Debug>:AI_SDK_DEBUG=1>
        $<$<CONFIG:Release>:AI_SDK_RELEASE=1;NDEBUG>
    )
endfunction()

# Targets that are installed and exported, filled in by the modules.
define_property(GLOBAL PROPERTY LLM_INSTALL_TARGETS
    BRIEF_DOCS "Targets installed by game-agent"
    FULL_DOCS "Appended by each module's CMakeLists.txt, consumed by cmake/Install.cmake")

# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------
add_subdirectory(src/core)
add_subdirectory(src/tools)
add_subdirectory(src/context)
add_subdirectory(src/providers)

# ---------------------------------------------------------------------------
# llm::sdk — convenience target for everything the SDK ships.
#
# An INTERFACE library: it produces no extra archive, it only aggregates.
# ---------------------------------------------------------------------------
add_library(llm-sdk INTERFACE)
add_library(llm::sdk ALIAS llm-sdk)
set_target_properties(llm-sdk PROPERTIES EXPORT_NAME sdk)

target_include_directories(llm-sdk
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(llm-sdk
    INTERFACE
        llm::core
        llm::tools
        llm::context
        llm::openai
)

set_property(GLOBAL APPEND PROPERTY LLM_INSTALL_TARGETS llm-sdk)

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation, export and package configuration
include(cmake/Install.cmake)

# Print configuration summary
message(STATUS "")
message(STATUS "game-agent ${PROJECT_VERSION} Configuration Summary")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build examples: ${BUILD_EXAMPLES}")
message(STATUS "  Build tests: ${BUILD_TESTS}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
