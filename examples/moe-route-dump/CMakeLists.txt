set(TARGET llama-moe-route-dump)
add_executable(${TARGET} moe_route_dump.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
