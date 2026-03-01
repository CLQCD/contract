if(${GPU_TARGET} STREQUAL "cuda")
    add_library(runtime_api STATIC target/runtime_api_cuda.cpp)
    target_link_libraries(runtime_api PUBLIC CUDA::cudart)
elseif(${GPU_TARGET} STREQUAL "hip")
    add_library(runtime_api STATIC target/runtime_api_hip.cpp)
endif()
set_target_properties(runtime_api PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory(baryon)
add_subdirectory(meson)

add_library(contract SHARED contract.cpp)
target_link_libraries(contract PRIVATE baryon meson)
target_link_libraries(contract PRIVATE runtime_api)
