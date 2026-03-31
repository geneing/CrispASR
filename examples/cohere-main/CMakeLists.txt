set(TARGET cohere-main)
add_executable(${TARGET} cohere-main.cpp)

include(DefaultTargetOptions)

target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(${TARGET} PRIVATE cohere common Threads::Threads)

install(TARGETS ${TARGET} RUNTIME)
