include_guard(GLOBAL)

# All third-party C++ sources are fetched by CPM; only system libraries use
# find_package(). See cmake/CPM.cmake for the pinned CPM version.

# Keep fetched sources out of the source tree (and out of the Nix store). A
# parent project that already set a cache, or a top-level build that embeds
# game-agent, keeps its own value.
if(NOT DEFINED CPM_SOURCE_CACHE)
    set(CPM_SOURCE_CACHE
        "${CMAKE_SOURCE_DIR}/.cache/CPM"
        CACHE PATH "CPM dependency source cache")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/CPM.cmake")

# ---------------------------------------------------------------------------
# System dependencies (platform libraries, not CPM)
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)
find_package(OpenSSL 3.0 REQUIRED COMPONENTS SSL Crypto)

# ---------------------------------------------------------------------------
# C++ dependencies
# ---------------------------------------------------------------------------

# JSON: argument decoding, schema generation and the request/response bodies.
# Version is kept in step with Football Sim so both share one package.
if(NOT TARGET nlohmann_json::nlohmann_json)
    CPMAddPackage(
        NAME nlohmann_json
        GITHUB_REPOSITORY nlohmann/json
        GIT_TAG v3.11.3
        OPTIONS
            "JSON_BuildTests OFF"
            "JSON_Install OFF"
    )
endif()

# HTTP/HTTPS client. Header-only on purpose: the compiled-library mode runs
# Python code generation, and this project has no Python toolchain.
if(NOT TARGET httplib::httplib)
    CPMAddPackage(
        NAME httplib
        GITHUB_REPOSITORY yhirose/cpp-httplib
        GIT_TAG v0.22.0
        OPTIONS
            "HTTPLIB_COMPILE OFF"
            "HTTPLIB_INSTALL OFF"
            "HTTPLIB_TEST OFF"
            "HTTPLIB_REQUIRE_OPENSSL ON"
            "HTTPLIB_USE_ZLIB_IF_AVAILABLE OFF"
            "HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF"
            "HTTPLIB_USE_ZSTD_IF_AVAILABLE OFF"
    )
endif()

# Unit tests only. The revision is pinned.
if(BUILD_TESTING AND NOT TARGET GTest::gtest)
    CPMAddPackage(
        NAME googletest
        GITHUB_REPOSITORY google/googletest
        GIT_TAG 3983f67e32fb3e9294487b9d4f9586efa6e5d088
        OPTIONS
            "INSTALL_GTEST OFF"
    )
endif()
