add_subdirectory(baryon)
add_subdirectory(meson)

add_library(runtime_api STATIC "target/runtime_api_cuda.cu")

add_library(contract SHARED contract.cc)
target_link_libraries(contract PRIVATE runtime_api)
target_link_libraries(contract PRIVATE baryon meson)