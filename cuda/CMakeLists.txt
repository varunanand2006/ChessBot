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
