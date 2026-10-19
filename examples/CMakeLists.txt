# Examples for AI SDK C++

# Helper function to create examples
function(add_llm_example name source_file)
    add_executable(${name} ${source_file})
    target_link_libraries(${name} PRIVATE llm::sdk)

    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples
    )

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(EXAMPLE_SUFFIX "_debug")
    else()
        set(EXAMPLE_SUFFIX "")
    endif()

    set_target_properties(${name} PROPERTIES
        OUTPUT_NAME "${name}${EXAMPLE_SUFFIX}"
    )
endfunction()

# The Session flow: context in, validated decision batch out, history managed.
add_llm_example(coach_session coach_session.cpp)

# The provider flow on its own: one completion, caller-side dispatch.
add_llm_example(coach_decision coach_decision.cpp)

# The same flow without a network, using a scripted transport.
add_llm_example(offline_transport offline_transport.cpp)

message(STATUS "Examples will be built in: ${CMAKE_BINARY_DIR}/examples/")
message(STATUS "  ./examples/coach_session      (scripted without OPENAI_API_KEY)")
message(STATUS "  ./examples/coach_decision     (needs OPENAI_API_KEY)")
message(STATUS "  ./examples/offline_transport  (no key, no network)")
