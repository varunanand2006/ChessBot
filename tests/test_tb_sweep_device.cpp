// Phase 3 gate (host half) — the DEVICE-shaped retrograde sweep
// (tb_sweep_device.hpp + tb_sweep_setup.hpp) converges to the SAME table as the
// authoritative solve_sweep_comb.
//
// sweep_host_reference runs the device update (CH_HD) Jacobi-style on the HOST to
// a fixpoint. Its value[] must equal solve_sweep_comb's value[] at EVERY index —
// a full local Phase 3 gate with no GPU. The device half (cuda/sweep_check.cu)
// then runs the same update in a kernel and diffs against solve_sweep_comb.
//
// This is a real cross-check, not a tautology: the reference sweep uses a
// different algorithm shape than the oracle — Jacobi ping-pong vs in-place
// Gauss-Seidel, the device movegen/make-unmake, and CombIndex-layout capture
// sub-tables vs the oracle's dense Index sub-tables. Agreement on the fixpoint is
// strong evidence the port is correct.
//
// KRK/KQK run by default (3-man, seconds). KQKR (4-man) is `slow`/manual: the
// host Jacobi sweep over 3.49M positions is minutes — that slowness is exactly
// why the real KQKR run happens on the GPU (cuda/sweep_check.cu).

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "slider.hpp"
#include "tb_comb_index.hpp"
#include "tb_solve.hpp"
#include "tb_sweep_setup.hpp"
#include "types.hpp"

namespace {

int g_failures = 0;
void fail(const char* m) { std::printf("[FAIL] %s\n", m); ++g_failures; }

void gate(std::vector<tb::Piece> extras, const char* name) {
    tb::CombIndex idx(extras);

    // Oracle.
    tb::Table oracle = tb::solve_sweep_comb(idx);

    // Device-shaped sweep on the host.
    int passes = 0;
    std::vector<int16_t> got = tb::sweep_host_reference(idx, &passes);

    if (got.size() != oracle.value.size()) { fail("size mismatch"); return; }

    std::size_t diffs = 0, first = 0;
    for (std::size_t i = 0; i < got.size(); ++i)
        if (got[i] != oracle.value[i]) { if (!diffs) first = i; ++diffs; }

    if (diffs) {
        std::printf("  %s: %zu / %zu values differ (first at index %zu: got=%d oracle=%d)\n",
                    name, diffs, got.size(), first, got[first], oracle.value[first]);
        fail("device-shaped sweep != solve_sweep_comb");
        return;
    }
    std::printf("[ OK ] %-7s  size=%zu  passes(jacobi)=%d  == solve_sweep_comb\n",
                name, got.size(), passes);
}

}  // namespace

int main(int argc, char** argv) {
    slider::init();  // solve_sweep_comb -> Index::legal -> slider attacks need it
    const bool slow = (argc > 1 && std::string(argv[1]) == "slow");

    using PT = PieceType;
    const Color W = Color::White, B = Color::Black;

    gate({{W, PT::Rook}},  "KRK");   // mate-in-16 theory
    gate({{W, PT::Queen}}, "KQK");   // mate-in-10 theory

    if (slow) {
        // 4-man: minutes on the host (the reason the real run is on the GPU).
        gate({{W, PT::Queen}, {B, PT::Rook}}, "KQKR");  // mate-in-35
    }

    if (g_failures == 0) {
        std::printf("Device-shaped sweep matches solve_sweep_comb.\n");
        return 0;
    }
    std::printf("\n%d sweep device check(s) failed.\n", g_failures);
    return 1;
}
