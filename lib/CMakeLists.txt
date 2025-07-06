add_subdirectory(baryon)
add_subdirectory(meson)

add_library(contract SHARED contract.cu)
target_link_libraries(contract PRIVATE baryon meson)
target_link_libraries(contract PUBLIC CUDA::cudart)