# nlohmann/json — the SDK exposes nlohmann::json in its public headers, so the
# dependency is PUBLIC and must be visible to anyone linking ai::core.
#
# Resolution order:
#   1. a target the enclosing project already provides (ClickHouse and other
#      consumers vendor their own copy, and reuse avoids a second copy),
#   2. an installed package found by find_package(),
#   3. the official upstream release, fetched by CMake.
#
# This used to be a vendored copy of nlohmann/json 3.12.0 with localeconv()
# removed from the parser and serializer. Upstream 3.12.0 already normalizes
# the decimal separator when dumping, so the patched tree only removed the
# (narrow, setlocale-related) data race; the SDK now uses the official release
# so the dependency can be shared instead of duplicated.
if(NOT TARGET nlohmann_json::nlohmann_json)
    find_package(nlohmann_json 3.11.3 CONFIG QUIET)
endif()

if(TARGET nlohmann_json::nlohmann_json)
    message(STATUS "Using the available nlohmann_json package")
else()
    include(FetchContent)

    set(JSON_BuildTests OFF CACHE INTERNAL "Disable nlohmann/json tests")
    set(JSON_Install ON CACHE INTERNAL "Install nlohmann/json with the SDK")

    FetchContent_Declare(nlohmann_json
        URL https://github.com/nlohmann/json/archive/refs/tags/v3.12.0.tar.gz
        URL_HASH SHA256=4b92eb0c06d10683f7447ce9406cb97cd4b453be18d7279320f7b2f025c10187
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(nlohmann_json)

    message(STATUS "Fetched nlohmann/json ${nlohmann_json_VERSION}")
endif()
