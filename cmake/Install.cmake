# Installation rules for every module target.
#
# The targets are collected by the modules themselves (LLM_INSTALL_TARGETS) and
# exported together under the llm:: namespace, so the installed interface is
# exactly the build tree's: llm::core, llm::tools, llm::context, llm::openai
# and the llm::sdk umbrella.

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

get_property(LLM_TARGETS GLOBAL PROPERTY LLM_INSTALL_TARGETS)

install(TARGETS ${LLM_TARGETS}
    EXPORT game-agent-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# .hpp for the public agent header, .h for the rest.
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
)

install(EXPORT game-agent-targets
    FILE game-agent-targets.cmake
    NAMESPACE llm::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/game-agent
)

configure_package_config_file(
    "${CMAKE_SOURCE_DIR}/cmake/game-agent-config.cmake.in"
    "${CMAKE_BINARY_DIR}/game-agent-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/game-agent
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR
)

write_basic_package_version_file(
    "${CMAKE_BINARY_DIR}/game-agent-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_BINARY_DIR}/game-agent-config.cmake"
    "${CMAKE_BINARY_DIR}/game-agent-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/game-agent
)
