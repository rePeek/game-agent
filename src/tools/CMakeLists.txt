# llm::tools — tool-call validation (JSON Schema subset, batch rules).
#
# Depends on llm::core only; nlohmann/json is an implementation detail (the
# public header is standard-library-only, so it stays a build-time dependency
# that no consumer has to provide).

add_library(llm-tools)
add_library(llm::tools ALIAS llm-tools)
set_target_properties(llm-tools PROPERTIES EXPORT_NAME tools)

configure_llm_component(llm-tools)
configure_llm_warnings(llm-tools)

target_sources(llm-tools
    PRIVATE
        validation.cpp
)

target_link_libraries(llm-tools
    PUBLIC
        llm::core
    PRIVATE
        $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>
)

set_property(GLOBAL APPEND PROPERTY LLM_INSTALL_TARGETS llm-tools)
