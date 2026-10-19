# llm::context — the context history and the Agent that owns it.
#
# Depends on llm::core and llm::commands: an Agent owns a command registry and
# is handed a std::shared_ptr<IModelProvider>, so it never links a concrete
# provider and a context-only build needs no HTTP and no TLS. agent.cpp lives
# at the src root because the Agent is the SDK's front door, not a detail of
# Context.

add_library(llm-context)
add_library(llm::context ALIAS llm-context)
set_target_properties(llm-context PROPERTIES EXPORT_NAME context)

configure_llm_component(llm-context)
configure_llm_warnings(llm-context)

target_sources(llm-context
    PRIVATE
        context.cpp
        ../agent.cpp
)

target_link_libraries(llm-context
    PUBLIC
        llm::core
        llm::commands
)

set_property(GLOBAL APPEND PROPERTY LLM_INSTALL_TARGETS llm-context)
