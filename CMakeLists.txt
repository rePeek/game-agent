cmake_minimum_required(VERSION 3.16)

# Project definition
project(ai-sdk-cpp 
    VERSION 0.1.0
    DESCRIPTION "A unified C++ SDK for multiple AI/LLM providers"
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

# Add third party directory (vendored and CMake-fetched dependencies)
add_subdirectory(third_party)

# Find system OpenSSL
find_package(OpenSSL REQUIRED)

# Helper function to configure component libraries
function(configure_ai_component target_name)
    target_include_directories(${target_name}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
endfunction()

# Common compile definitions for HTTP support
set(HTTPLIB_COMPILE_DEFS
    CPPHTTPLIB_OPENSSL_SUPPORT=1
)

# Create libraries
add_library(ai-sdk-cpp-core)
add_library(ai-sdk-cpp-openai)
add_library(ai-sdk-cpp INTERFACE)

# Keep installed target names consistent with the build-tree aliases.
set_target_properties(ai-sdk-cpp-core PROPERTIES EXPORT_NAME core)
set_target_properties(ai-sdk-cpp-openai PROPERTIES EXPORT_NAME openai)
set_target_properties(ai-sdk-cpp PROPERTIES EXPORT_NAME sdk)

# Set library aliases
add_library(ai::sdk ALIAS ai-sdk-cpp)
add_library(ai::core ALIAS ai-sdk-cpp-core)
add_library(ai::openai ALIAS ai-sdk-cpp-openai)

# Configure all component libraries
configure_ai_component(ai-sdk-cpp-core)
configure_ai_component(ai-sdk-cpp-openai)

# Configure main interface library
target_include_directories(ai-sdk-cpp 
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Core component sources
target_sources(ai-sdk-cpp-core
    PRIVATE
        # Type implementations
        src/types/generate_options.cpp
        src/types/message.cpp
        
        # HTTP and base provider infrastructure
        src/http/http_request_handler.cpp
        src/providers/base_provider_client.cpp
)

# Provider-specific sources
target_sources(ai-sdk-cpp-openai
    PRIVATE
        src/providers/openai/openai_client.cpp
        src/providers/openai/openai_request_builder.cpp
        src/providers/openai/openai_response_parser.cpp
        src/providers/openai/openai_factory.cpp
)

# ---------------------------------------------------------------------------
# llm:: core — the protocol surface.
#
# Deliberately STL-only: no JSON, no HTTP, no TLS. It is built next to the
# legacy ai:: components while the migration is in progress, and is not part of
# the installed export set yet (that is finalised once the legacy API is gone).
# ---------------------------------------------------------------------------
add_library(llm-core)
add_library(llm::core ALIAS llm-core)
configure_ai_component(llm-core)

target_sources(llm-core
    PRIVATE
        src/llm/core.cpp
)

# HTTP transport implementation (customed by the providers).
add_library(llm-http)
add_library(llm::http ALIAS llm-http)
configure_ai_component(llm-http)

target_sources(llm-http
    PRIVATE
        src/llm/http_transport_httplib.cpp
)

# OpenAI-compatible provider on top of the core and the HTTP transport.
add_library(llm-openai)
add_library(llm::openai ALIAS llm-openai)
configure_ai_component(llm-openai)

target_sources(llm-openai
    PRIVATE
        src/llm/openai/openai_provider.cpp
        src/llm/openai/openai_request_builder.cpp
        src/llm/openai/openai_response_parser.cpp
)

# Tool-call validation: standard-library-only interface, JSON in the
# implementation only.
add_library(llm-validation)
add_library(llm::validation ALIAS llm-validation)
configure_ai_component(llm-validation)

target_sources(llm-validation
    PRIVATE
        src/llm/validation.cpp
)

# Link dependencies

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

# JSON stays an implementation detail of the provider: its public header is
# standard-library-only.
target_link_libraries(llm-validation
    PUBLIC
        llm::core
    PRIVATE
        $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>
)

# JSON stays an implementation detail of the provider: its public header is
# standard-library-only.
target_link_libraries(llm-openai
    PUBLIC
        llm::core
        llm::validation
    PRIVATE
        llm::http
        $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>
)
target_link_libraries(ai-sdk-cpp-core
    PUBLIC
        nlohmann_json::nlohmann_json
    PRIVATE
        $<BUILD_INTERFACE:httplib::httplib>
        $<BUILD_INTERFACE:OpenSSL::SSL>
        $<BUILD_INTERFACE:OpenSSL::Crypto>
        $<INSTALL_INTERFACE:OpenSSL::SSL>
        $<INSTALL_INTERFACE:OpenSSL::Crypto>
        $<INSTALL_INTERFACE:Threads::Threads>
)

# OpenAI component links to core and shares the common dependencies
target_link_libraries(ai-sdk-cpp-openai
    PUBLIC
        ai::core
        nlohmann_json::nlohmann_json
    PRIVATE
        $<BUILD_INTERFACE:httplib::httplib>
        $<INSTALL_INTERFACE:OpenSSL::SSL>
        $<INSTALL_INTERFACE:OpenSSL::Crypto>
        $<INSTALL_INTERFACE:Threads::Threads>
)
target_compile_definitions(ai-sdk-cpp-openai
    PUBLIC
        AI_SDK_HAS_OPENAI=1
    PRIVATE
        ${HTTPLIB_COMPILE_DEFS}
)

# Core needs HTTP definitions too
target_compile_definitions(ai-sdk-cpp-core
    PRIVATE
        ${HTTPLIB_COMPILE_DEFS}
)

# Main library links all components
target_link_libraries(ai-sdk-cpp
    INTERFACE
        ai::core
        ai::openai
)

# Define all component availability for main library
target_compile_definitions(ai-sdk-cpp
    INTERFACE
        AI_SDK_HAS_OPENAI=1
)

# List of all concrete component targets
set(COMPONENT_TARGETS
    ai-sdk-cpp-core
    ai-sdk-cpp-openai
)

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

# Apply common options to all component targets
foreach(target ${COMPONENT_TARGETS} llm-core llm-http llm-openai llm-validation)
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

# Install all targets (httplib is internal only, not installed)
set(ALL_TARGETS ${COMPONENT_TARGETS} ai-sdk-cpp)
install(TARGETS ${ALL_TARGETS}
    EXPORT ai-sdk-cpp-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install headers
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)

# Export targets
install(EXPORT ai-sdk-cpp-targets
    FILE ai-sdk-cpp-targets.cmake
    NAMESPACE ai::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ai-sdk-cpp
)

# Create and install config file
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
