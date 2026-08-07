# CUDA scaffold (Phase 0). This subdirectory is added by the top-level
# CMakeLists ONLY when a CUDA compiler is found, so a CPU-only machine with no
# toolkit configures and builds the engine exactly as before — nothing here runs
# on the dev laptop.

# --- Hello / equality-harness smoke test -----------------------------------
add_executable(cuda_hello hello.cu)

# Pull in include/ so device code can later include the CH_HD-annotated shared
# headers (combinatorial.hpp, tb_symmetry.hpp, ...). Phase 0 only needs
# cuda_compat.hpp, but wiring the path now means Phase 1 adds no CMake churn.
target_include_directories(cuda_hello PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Match the engine's C++20 host dialect on nvcc's host-compiler pass so shared
# headers compile identically on both sides. Separable compilation is left OFF
# for now (single .cu, no cross-TU device linking yet); enable it in Phase 2+
# when device code spans multiple .cu files.
set_target_properties(cuda_hello PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON)

# Register the harness as a ctest. Its exit code is the Phase 0 gate (0 == host
# and device agree). On a CPU-only machine this whole subdirectory is skipped,
# so the test simply doesn't exist there — it only appears on a box with a GPU.
add_test(NAME cuda_hello COMMAND cuda_hello)

# --- Phase 1 (primitives): device combinatorics + D4 vs host ---------------
# Runs the CH_HD-annotated combinatorial.hpp / tb_symmetry.hpp on the device and
# diffs bit-exact against the host. The prerequisite gate for the device
# CombIndex port.
add_executable(cuda_index_check index_check.cu)
target_include_directories(cuda_index_check PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(cuda_index_check PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON)
add_test(NAME cuda_index_check COMMAND cuda_index_check)

# --- Phase 1c: device CombIndex encode/decode vs host CombIndex ------------
# Runs comb_encode/comb_decode (tb_comb_index_device.hpp) on the device and
# diffs against the host tb::CombIndex over whole materials — the stated Phase 1
# gate. Links chess_engine for the host reference. Deliberately does NOT link
# chess_project_options: its -O3 -march=native are bare host flags nvcc would not
# forward through to the host compiler, and this target needs none of them.
add_executable(cuda_comb_index_check comb_index_check.cu)
target_include_directories(cuda_comb_index_check PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(cuda_comb_index_check PRIVATE chess_engine)
set_target_properties(cuda_comb_index_check PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON)
add_test(NAME cuda_comb_index_check COMMAND cuda_comb_index_check)
