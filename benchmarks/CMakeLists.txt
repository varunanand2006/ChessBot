# Timing harnesses. Meaningful only in Release (-O3 -march=native), which is
# the default build type for this project.

add_executable(bench bench.cpp)
target_link_libraries(bench PRIVATE chess_engine chess_project_options)
