# llm::context — per-session history and asynchronous decisions.
#
# Depends on llm::core only: a Session is handed a std::shared_ptr<IModelProvider>
# and never links a concrete provider, so a context-only build needs no HTTP or
# TLS.

add_library(llm-context)
add_library(llm::context ALIAS llm-context)
set_target_properties(llm-context PROPERTIES EXPORT_NAME context)

configure_llm_component(llm-context)
configure_llm_warnings(llm-context)

target_sources(llm-context
    PRIVATE
        context.cpp
        session.cpp
        ../agent.cpp
)

target_link_libraries(llm-context
    PUBLIC
        llm::core
)

set_property(GLOBAL APPEND PROPERTY LLM_INSTALL_TARGETS llm-context)
