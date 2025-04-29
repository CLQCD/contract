add_subdirectory(baryon)

add_library(contract SHARED contract.cu)
target_link_libraries(contract PRIVATE baryon)
target_link_libraries(contract PUBLIC CUDA::cudart)