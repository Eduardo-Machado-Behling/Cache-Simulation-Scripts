cmake_minimum_required(VERSION 3.10.0)
project(cache-sim VERSION 0.1.0 LANGUAGES C)

file(GLOB SRC "${CMAKE_SOURCE_DIR}/src/*.c")

add_executable(cache-sim ${SRC})

target_include_directories(cache-sim PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_directories(cache-sim PRIVATE "${CMAKE_SOURCE_DIR}/lib")
target_link_libraries(cache-sim PRIVATE pcre2-8)

set_target_properties(cache-sim PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")