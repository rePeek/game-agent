# Providers: one independently built library per model backend, so a project
# that only needs a local model never has to deal with HTTP or OpenSSL.

add_subdirectory(openai)
