set(TARGET cohere-main)
add_executable(${TARGET} cohere-main.cpp)

include(DefaultTargetOptions)

target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(${TARGET} PRIVATE cohere common Threads::Threads)

set(TARGET_Q cohere-quantize)
add_executable(${TARGET_Q} cohere-quantize.cpp)
target_include_directories(${TARGET_Q} PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(${TARGET_Q} PRIVATE cohere common Threads::Threads)

install(TARGETS ${TARGET} ${TARGET_Q} RUNTIME)
