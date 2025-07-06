add_executable(test_baryon test_baryon.cpp)
target_link_libraries(test_baryon PUBLIC contract)

add_executable(test_meson test_meson.cpp)
target_link_libraries(test_meson PUBLIC contract)
