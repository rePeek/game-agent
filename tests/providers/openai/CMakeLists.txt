# OpenAI provider tests: request building, response parsing and status mapping
# against a scripted transport. They include the provider's internal headers
# (hence the src/ include directory) and assert on the built request JSON
# (hence nlohmann/json).

add_llm_test(llm_openai_tests llm_openai_test.cpp)
target_include_directories(llm_openai_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(llm_openai_tests
    PRIVATE
        llm::openai
        nlohmann_json::nlohmann_json
)
