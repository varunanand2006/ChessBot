# Test executables. Dependency-free (no GoogleTest) to keep the build simple
# and portable to the CUDA phase — each test is a small program returning
# nonzero on failure, registered with CTest.

add_executable(test_fen test_fen.cpp)
target_link_libraries(test_fen PRIVATE chess_engine chess_project_options)
add_test(NAME fen COMMAND test_fen)

# Attack tables are header-only (constexpr); no engine sources needed, but link
# the engine lib anyway for a uniform test setup.
add_executable(test_attacks test_attacks.cpp)
target_link_libraries(test_attacks PRIVATE chess_engine chess_project_options)
add_test(NAME attacks COMMAND test_attacks)

# Mandatory gate: magic sliders vs. ray-walk reference, 10k occupancies/square.
add_executable(test_slider test_slider.cpp)
target_link_libraries(test_slider PRIVATE chess_engine chess_project_options)
add_test(NAME slider COMMAND test_slider)

# Move encoding is header-only (constexpr); link engine lib for uniformity.
add_executable(test_move test_move.cpp)
target_link_libraries(test_move PRIVATE chess_engine chess_project_options)
add_test(NAME move COMMAND test_move)

# make/unmake reversibility + Zobrist consistency.
add_executable(test_makemove test_makemove.cpp)
target_link_libraries(test_makemove PRIVATE chess_engine chess_project_options)
add_test(NAME makemove COMMAND test_makemove)

# Legal move generation smoke tests (perft in Step 8 is the real gate).
add_executable(test_movegen test_movegen.cpp)
target_link_libraries(test_movegen PRIVATE chess_engine chess_project_options)
add_test(NAME movegen COMMAND test_movegen)

# Perft — the Phase 1 gate. Fast tier runs in the default suite; deep tier
# (startpos->6, Kiwipete->5, etc.) is a separate entry with a long timeout.
add_executable(test_perft test_perft.cpp)
target_link_libraries(test_perft PRIVATE chess_engine chess_project_options)
add_test(NAME perft COMMAND test_perft)
add_test(NAME perft_deep COMMAND test_perft deep)
set_tests_properties(perft_deep PROPERTIES TIMEOUT 300 LABELS deep)

# Evaluation: structural invariants (start = 0, color-swap antisymmetry).
add_executable(test_eval test_eval.cpp)
target_link_libraries(test_eval PRIVATE chess_engine chess_project_options)
add_test(NAME eval COMMAND test_eval)

# Search: legal best move, mate-in-1 both colors, wins a hanging piece.
add_executable(test_search test_search.cpp)
target_link_libraries(test_search PRIVATE chess_engine chess_project_options)
add_test(NAME search COMMAND test_search)

# Phase 4: tablebase probing — correct WDL + search pruning (3-man, fast).
add_executable(test_tb_probe test_tb_probe.cpp)
target_link_libraries(test_tb_probe PRIVATE chess_engine chess_project_options)
add_test(NAME tb_probe COMMAND test_tb_probe)

# Phase 2 / Step 2.1: tablebase position indexing bijection + symmetry.
# Fast tier = 3-man; slow tier adds 4-man (millions of positions).
add_executable(test_tb_index test_tb_index.cpp)
target_link_libraries(test_tb_index PRIVATE chess_engine chess_project_options)
add_test(NAME tb_index COMMAND test_tb_index)
add_test(NAME tb_index_4man COMMAND test_tb_index slow)
set_tests_properties(tb_index_4man PROPERTIES TIMEOUT 300 LABELS deep)

# Phase 2 / Step 2.2: retrograde DTM solver — two solvers must agree.
# Fast tier = 3-man; slow tier adds the first 4-man material-DAG table (KQKR).
add_executable(test_tb_solve test_tb_solve.cpp)
target_link_libraries(test_tb_solve PRIVATE chess_engine chess_project_options)
add_test(NAME tb_solve COMMAND test_tb_solve)
add_test(NAME tb_solve_4man COMMAND test_tb_solve slow)
set_tests_properties(tb_solve_4man PROPERTIES TIMEOUT 300 LABELS deep)
