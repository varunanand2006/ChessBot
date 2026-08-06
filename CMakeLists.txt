# ==========================================================================
#  Chess Engine — C++/CUDA rewrite
#  Phase 1: CPU engine core (board, movegen, make/unmake).
#
#  Toolchain assumed for this project: GCC (MinGW-w64 UCRT) + Ninja.
#  Clang/MSVC are not used, so the release flags below are written for the
#  GCC/Clang flag dialect (-O3 -march=native) per the project spec.
# ==========================================================================
cmake_minimum_required(VERSION 3.20)

project(ChessEngine
        VERSION 0.1.0
        DESCRIPTION "Bitboard chess engine (C++20) with a later CUDA tablebase phase"
        LANGUAGES CXX)

# --- Language standard -----------------------------------------------------
# C++20, no compiler extensions (-std=c++20, not -std=gnu++20). We want the
# same dialect that will later compile as CUDA host code, so keep it standard.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile_commands.json for clangd / tooling (gitignored).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Default build type ----------------------------------------------------
# Single-config generators (Ninja, Makefiles) have no build type by default,
# which silently gives an -O0, un-tuned build. Default to Release so the
# "produces a binary that runs" gate reflects the optimized configuration we
# actually care about for benchmarking.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo)
endif()

# --- Warnings (applied to our targets only, not third-party) ---------------
# -Wconversion/-Wsign-conversion are deliberately NOT enabled: bitboard code
# does constant intentional narrowing (square<->index<->uint8_t, shifts into
# uint64_t) and those warnings would force a cast on nearly every line,
# obscuring the logic. -Wall -Wextra -Wshadow catch the real mistakes.
add_library(chess_warnings INTERFACE)
target_compile_options(chess_warnings INTERFACE
        -Wall -Wextra -Wshadow)

# --- Release flags ---------------------------------------------------------
# -O3        : full optimization; move generation and search are hot loops
#              where the extra inlining/vectorization is the whole point.
# -march=native: emit instructions for THIS machine's CPU. Critical here —
#              magic-bitboard and popcount-heavy code benefits directly from
#              BMI2/POPCNT/LZCNT. Cost: the binary is not portable to older
#              CPUs. Acceptable: this is a benchmark/portfolio build, run on
#              the machine it is compiled on. The portable alternative would
#              be -mpopcnt -mbmi2 (explicit ISA subset) at some tuning loss.
add_library(chess_flags_release INTERFACE)
target_compile_options(chess_flags_release INTERFACE
        $<$<CONFIG:Release>:-O3 -march=native>)

# --- Debug flags -----------------------------------------------------------
# The spec asks for a sanitizer debug build. The WinLibs MinGW-w64 toolchain
# used here ships NEITHER libasan NOR libubsan, so -fsanitize=address and the
# default (runtime-library) -fsanitize=undefined both fail to LINK.
#
# Workaround that keeps real UB detection without those runtimes:
#   -fsanitize=undefined -fsanitize-trap=undefined
#       UBSan in "trap" mode: a violation executes __builtin_trap() (ud2)
#       instead of calling into libubsan. No runtime library required.
#       Trade-off vs. full UBSan: you get a hard crash at the fault site
#       (run under gdb to see where) rather than a formatted diagnostic.
#   -D_GLIBCXX_ASSERTIONS : libstdc++ precondition checks (e.g. container
#       bounds, std::array::at) — cheap correctness net for debug runs.
#   -fstack-protector-strong : catch stack-buffer overruns.
# If this project is later built with a toolchain that ships the sanitizer
# runtimes (MSVC ASan, or a full GCC/Clang), switch to real
# -fsanitize=address,undefined there.
add_library(chess_flags_debug INTERFACE)
target_compile_options(chess_flags_debug INTERFACE
        $<$<CONFIG:Debug>:-Og -g
                          -fsanitize=undefined -fsanitize-trap=undefined
                          -fstack-protector-strong>)
target_compile_definitions(chess_flags_debug INTERFACE
        $<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>)

# Convenience interface bundling the per-config flags + warnings. Every engine
# target links this; it carries no code, only flags.
add_library(chess_project_options INTERFACE)
target_link_libraries(chess_project_options INTERFACE
        chess_warnings chess_flags_release chess_flags_debug)

# --- Shared include directory ----------------------------------------------
# Public headers live in include/. Exposed via an INTERFACE target so every
# consumer (engine lib, tests, benchmarks) picks it up uniformly.
add_library(chess_headers INTERFACE)
target_include_directories(chess_headers INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

# --- Engine library --------------------------------------------------------
# Core engine sources build into a static library that both the main binary
# and the test/benchmark executables link against. Sources are listed
# explicitly (no globbing) so adding a file is a deliberate, reviewable change
# and CMake reliably reconfigures.
add_library(chess_engine STATIC
        src/position.cpp
        src/slider.cpp
        src/movegen.cpp
        src/perft.cpp
        src/tb_index.cpp
        src/tb_solve.cpp)
target_link_libraries(chess_engine
        PUBLIC  chess_headers
        PRIVATE chess_project_options)

# --- Primary executable ----------------------------------------------------
add_executable(chess src/main.cpp)
target_link_libraries(chess PRIVATE chess_engine chess_project_options)

# --- Tests -----------------------------------------------------------------
enable_testing()
add_subdirectory(tests)

# --- Benchmarks ------------------------------------------------------------
add_subdirectory(benchmarks)
