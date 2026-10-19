# llm::core — protocol vocabulary: messages, requests, results, provider.
#
# Standard-library only: the core must not pull in JSON, HTTP or TLS, so it can
# be used by anything (and so this module is the boundary every other module
# depends on).

add_library(llm-core)
add_library(llm::core ALIAS llm-core)
set_target_properties(llm-core PROPERTIES EXPORT_NAME core)

configure_llm_component(llm-core)
configure_llm_warnings(llm-core)

target_sources(llm-core
    PRIVATE
        core.cpp
)

set_property(GLOBAL APPEND PROPERTY LLM_INSTALL_TARGETS llm-core)
