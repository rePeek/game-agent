cmake_minimum_required(VERSION 3.16)

project(ai_sdk_third_party)

# Set the third party directory variable for all subdirectories
set(AI_SDK_THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "Third party directory root")

# Add httplib (header-only, vendored)
add_subdirectory(httplib-cmake)

# Add nlohmann_json (reused, found, or fetched from upstream)
add_subdirectory(nlohmann_json-cmake)

# Add googletest (only if BUILD_TESTS is ON)
if(BUILD_TESTS)
    add_subdirectory(googletest-cmake)
endif()
