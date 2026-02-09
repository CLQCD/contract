add_subdirectory(baryon)
add_subdirectory(meson)

add_library(runtime_api STATIC "target/runtime_api_cuda.cu")
set_property(TARGET runtime_api PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(contract SHARED contract.cpp)
target_link_libraries(contract PRIVATE runtime_api)
target_link_libraries(contract PRIVATE baryon meson)