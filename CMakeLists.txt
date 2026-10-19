cmake_minimum_required(VERSION 3.16)

# Project definition
project(ai-sdk-cpp
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
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# Third-party dependencies: a vendored cpp-httplib header, nlohmann/json
# (reused, found or fetched) and GoogleTest when tests are enabled.
add_subdirectory(third_party)

find_package(OpenSSL REQUIRED)

# Public headers live in include/, implementation details in src/.
function(configure_llm_component target_name)
    target_include_directories(${target_name}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
endfunction()

# ---------------------------------------------------------------------------
# llm::core — request, message, tool-call and completion vocabulary.
#
# Deliberately standard-library-only: the core has no JSON, HTTP or TLS
# dependency, so it can be embedded anywhere and providers translate to it.
# ---------------------------------------------------------------------------
add_library(llm-core)
add_library(llm::core ALIAS llm-core)
set_target_properties(llm-core PROPERTIES EXPORT_NAME core)
configure_llm_component(llm-core)

target_sources(llm-core
    PRIVATE
        src/llm/core.cpp
)

# ---------------------------------------------------------------------------
# llm::validation — tool-call validation.
#
# The public header is standard-library-only (JSON text in, report out); the
# JSON parser is an implementation detail.
# ---------------------------------------------------------------------------
add_library(llm-validation)
add_library(llm::validation ALIAS llm-validation)
set_target_properties(llm-validation PROPERTIES EXPORT_NAME validation)
configure_llm_component(llm-validation)

target_sources(llm-validation
    PRIVATE
        src/llm/validation.cpp
)

target_link_libraries(llm-validation
    PUBLIC
        llm::core
    PRIVATE
        $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>
)

# ---------------------------------------------------------------------------
# llm::http — HTTP/HTTPS transport implementation (cpp-httplib + OpenSSL).
# ---------------------------------------------------------------------------
add_library(llm-http)
add_library(llm::http ALIAS llm-http)
set_target_properties(llm-http PROPERTIES EXPORT_NAME http)
configure_llm_component(llm-http)

target_sources(llm-http
    PRIVATE
        src/llm/http_transport_httplib.cpp
)

# httplib is internal: only OpenSSL and Threads have to be found by a consumer.
target_link_libraries(llm-http
    PUBLIC
        llm::core
    PRIVATE
        $<BUILD_INTERFACE:httplib::httplib>
        $<BUILD_INTERFACE:OpenSSL::SSL>
        $<BUILD_INTERFACE:OpenSSL::Crypto>
        $<INSTALL_INTERFACE:OpenSSL::SSL>
        $<INSTALL_INTERFACE:OpenSSL::Crypto>
        $<INSTALL_INTERFACE:Threads::Threads>
)

# ---------------------------------------------------------------------------
# llm::openai — OpenAI-compatible Chat Completions provider.
# ---------------------------------------------------------------------------
add_library(llm-openai)
add_library(llm::openai ALIAS llm-openai)
set_target_properties(llm-openai PROPERTIES EXPORT_NAME openai)
configure_llm_component(llm-openai)

target_sources(llm-openai
    PRIVATE
        src/llm/openai/openai_provider.cpp
        src/llm/openai/openai_request_builder.cpp
        src/llm/openai/openai_response_parser.cpp
)

target_link_libraries(llm-openai
    PUBLIC
        llm::core
        llm::validation
    PRIVATE
        llm::http
        $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>
)

# ---------------------------------------------------------------------------
# llm::sdk — everything the SDK ships.
# ---------------------------------------------------------------------------
add_library(llm-sdk INTERFACE)
add_library(llm::sdk ALIAS llm-sdk)
set_target_properties(llm-sdk PROPERTIES EXPORT_NAME sdk)

target_include_directories(llm-sdk
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(llm-sdk
    INTERFACE
        llm::core
        llm::validation
        llm::http
        llm::openai
)

# Concrete (non-INTERFACE) targets: these take compile options and definitions.
set(LLM_CONCRETE_TARGETS
    llm-core
    llm-validation
    llm-http
    llm-openai
)

# Everything that is installed and exported.
set(LLM_TARGETS ${LLM_CONCRETE_TARGETS} llm-sdk)

# Common compile options
if(MSVC)
    set(COMMON_WARNINGS /W4)
    set(COMMON_PLATFORM_DEFS WIN32_LEAN_AND_MEAN NOMINMAX _WIN32_WINNT=0x0601)
    set(DEBUG_FLAGS)
    set(RELEASE_FLAGS)
else()
    set(COMMON_WARNINGS -Wall -Wextra -Wpedantic)
    set(DEBUG_FLAGS -g -O0)
    set(RELEASE_FLAGS -O3)
    if(APPLE)
        list(APPEND COMMON_WARNINGS -stdlib=libc++)
    endif()
endif()

foreach(target ${LLM_CONCRETE_TARGETS})
    target_compile_options(${target} PRIVATE
        ${COMMON_WARNINGS}
        $<$<CONFIG:Debug>:${DEBUG_FLAGS}>
        $<$<CONFIG:Release>:${RELEASE_FLAGS}>
    )

    target_compile_definitions(${target} PRIVATE
        $<$<CONFIG:Debug>:AI_SDK_DEBUG=1>
        $<$<CONFIG:Release>:AI_SDK_RELEASE=1;NDEBUG>
        $<$<BOOL:${MSVC}>:${COMMON_PLATFORM_DEFS}>
    )
endforeach()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation rules
include(GNUInstallDirs)

install(TARGETS ${LLM_TARGETS}
    EXPORT ai-sdk-cpp-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)

# Export targets under the llm:: namespace (llm::core, llm::openai, ...).
install(EXPORT ai-sdk-cpp-targets
    FILE ai-sdk-cpp-targets.cmake
    NAMESPACE llm::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ai-sdk-cpp
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ai-sdk-cpp-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ai-sdk-cpp-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ai-sdk-cpp
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ai-sdk-cpp-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ai-sdk-cpp-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ai-sdk-cpp-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ai-sdk-cpp
)

# Print configuration summary
message(STATUS "")
message(STATUS "ai-sdk-cpp ${PROJECT_VERSION} Configuration Summary")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build examples: ${BUILD_EXAMPLES}")
message(STATUS "  Build tests: ${BUILD_TESTS}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
