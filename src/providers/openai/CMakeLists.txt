# llm::openai — OpenAI-compatible Chat Completions provider.
#
# It owns its HTTP transport: cpp-httplib, nlohmann/json and OpenSSL are
# implementation details of this provider, not modules of their own. OpenSSL
# and Threads must still reach a consumer that links the static archive, which
# is what the INSTALL_INTERFACE entries below export.

find_package(OpenSSL REQUIRED)

add_library(llm-openai)
add_library(llm::openai ALIAS llm-openai)
set_target_properties(llm-openai PROPERTIES EXPORT_NAME openai)

configure_llm_component(llm-openai)
configure_llm_warnings(llm-openai)

target_sources(llm-openai
    PRIVATE
        openai_provider.cpp
        request_builder.cpp
        response_parser.cpp
        http_transport_httplib.cpp
)

target_link_libraries(llm-openai
    PUBLIC
        llm::core
        llm::commands
    PRIVATE
        $<BUILD_INTERFACE:httplib::httplib>
        $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>
        $<BUILD_INTERFACE:OpenSSL::SSL>
        $<BUILD_INTERFACE:OpenSSL::Crypto>
        $<INSTALL_INTERFACE:OpenSSL::SSL>
        $<INSTALL_INTERFACE:OpenSSL::Crypto>
        $<INSTALL_INTERFACE:Threads::Threads>
)

set_property(GLOBAL APPEND PROPERTY LLM_INSTALL_TARGETS llm-openai)
