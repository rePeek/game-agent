# llm::commands — command registration vocabulary and argument validation.
#
# Depends on llm::core only; nlohmann/json is an implementation detail (the
# public headers are standard-library-only, so it stays a build-time dependency
# that no consumer has to provide).

add_library(llm-commands)
add_library(llm::commands ALIAS llm-commands)
set_target_properties(llm-commands PROPERTIES EXPORT_NAME commands)

configure_llm_component(llm-commands)
configure_llm_warnings(llm-commands)

target_sources(llm-commands
    PRIVATE
        validation.cpp
        arguments.cpp
        schema.cpp
        registry.cpp
        prepared_command.cpp
)

target_link_libraries(llm-commands
    PUBLIC
        llm::core
    PRIVATE
        $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>
)

set_property(GLOBAL APPEND PROPERTY LLM_INSTALL_TARGETS llm-commands)
