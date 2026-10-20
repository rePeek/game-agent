# clang-format integration.
#
# The repository formats C++ with clang-format directly through two CMake
# targets, so no Python wrapper is needed:
#
#   cmake --build --preset debug --target format        # rewrite in place
#   cmake --build --preset debug --target format-check  # check only (CI)
#
# Only include/, src/, tests/ and examples/ are scanned, so third_party/,
# build/ and FetchContent sources are never touched.

find_program(CLANG_FORMAT_EXECUTABLE
    NAMES clang-format
    REQUIRED
)

set(FORMAT_FILES)

foreach(dir include src tests examples)
    file(GLOB_RECURSE files
        CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/${dir}/*.h"
        "${PROJECT_SOURCE_DIR}/${dir}/*.hpp"
        "${PROJECT_SOURCE_DIR}/${dir}/*.cpp"
        "${PROJECT_SOURCE_DIR}/${dir}/*.cc"
        "${PROJECT_SOURCE_DIR}/${dir}/*.cxx"
    )
    list(APPEND FORMAT_FILES ${files})
endforeach()

list(REMOVE_DUPLICATES FORMAT_FILES)

add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXECUTABLE} -i ${FORMAT_FILES}
    COMMENT "Formatting ${PROJECT_NAME} sources with clang-format"
    VERBATIM
)

add_custom_target(format-check
    COMMAND ${CLANG_FORMAT_EXECUTABLE} --dry-run --Werror ${FORMAT_FILES}
    COMMENT "Checking ${PROJECT_NAME} formatting with clang-format"
    VERBATIM
)
