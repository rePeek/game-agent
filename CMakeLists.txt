cmake_minimum_required(VERSION 3.24)

project(game-agent
    VERSION 0.1.0
    DESCRIPTION "C++ Agent SDK"
    LANGUAGES CXX
)

# Only GCC compiles this project. clang-format/clang-tidy/clangd are still used
# as tools, but the C++ compiler is GCC so the toolchain matches the games this
# SDK is embedded in.
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(FATAL_ERROR
        "game-agent requires GCC, but CMAKE_CXX_COMPILER_ID is "
        "'${CMAKE_CXX_COMPILER_ID}'. Configure with a GCC toolchain.")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Defines BUILD_TESTING (and enable_testing()); the standard switch that
# replaces the project-local BUILD_TESTS.
include(CTest)

option(BUILD_EXAMPLES "Build example applications" ON)
option(BUILD_SHARED_LIBS "Build shared libraries instead of static ones" OFF)

# ---------------------------------------------------------------------------
# Module helpers
#
# Public headers live in include/, implementation details in src/. A module
# target is created, configured and added to the install list by its own
# src/<module>/CMakeLists.txt. cxx_std_23 is PUBLIC so an installed consumer
# inherits the language requirement.
# ---------------------------------------------------------------------------
function(configure_llm_component target_name)
    target_include_directories(${target_name}
        PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${PROJECT_SOURCE_DIR}/src
    )
    target_compile_features(${target_name} PUBLIC cxx_std_23)
endfunction()

function(configure_llm_warnings target_name)
    target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_definitions(${target_name} PRIVATE
        $<$<CONFIG:Debug>:AI_SDK_DEBUG=1>
        $<$<CONFIG:Release>:AI_SDK_RELEASE=1;NDEBUG>
    )
endfunction()

# Targets that are installed and exported, filled in by the modules.
define_property(GLOBAL PROPERTY LLM_INSTALL_TARGETS
    BRIEF_DOCS "Targets installed by game-agent"
    FULL_DOCS "Appended by each module's CMakeLists.txt, consumed by cmake/Install.cmake")

# Third-party C++ dependencies are fetched by CPM (nlohmann/json, cpp-httplib
# and, with tests on, GoogleTest); OpenSSL and Threads are system libraries
# found with find_package(). See cmake/Dependencies.cmake.
include(cmake/Dependencies.cmake)

# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------
add_subdirectory(src/core)
add_subdirectory(src/commands)
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
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(llm-sdk
    INTERFACE
        llm::core
        llm::commands
        llm::context
        llm::openai
)

set_property(GLOBAL APPEND PROPERTY LLM_INSTALL_TARGETS llm-sdk)

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests (CTest's BUILD_TESTING switch)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# clang-format targets: `format` and `format-check`
include(cmake/Formatting.cmake)

# Installation, export and package configuration
include(cmake/Install.cmake)

# Print configuration summary
message(STATUS "")
message(STATUS "game-agent ${PROJECT_VERSION} Configuration Summary")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build examples: ${BUILD_EXAMPLES}")
message(STATUS "  Build tests: ${BUILD_TESTING}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
