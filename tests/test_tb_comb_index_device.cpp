// Phase 1c gate (host half) — the DEVICE-shaped combinatorial index
// (tb::comb_encode / tb::comb_decode over POD DeviceKingTable/DeviceMaterial,
// in tb_comb_index_device.hpp) computes EXACTLY what the authoritative
// tb::CombIndex does.
//
// These functions are CH_HD, so this runs them on the HOST and diffs them
// against CombIndex over whole materials — a full local gate with no GPU. The
// GPU half (cuda/comb_index_check.cu) then only has to reproduce this host
// result on the device; the primitives underneath are already device-verified
// by cuda/index_check.cu. So the port's correctness is pinned in three places,
// only one of which needs the rented box.
//
// For each index i in [0, size):
//   * comb_decode(i) == CombIndex::decode(i)      (same men squares + stm)
//   * comb_encode(those squares) == CombIndex::encode(those squares) == i
// i.e. both directions of the device mirror match the host reference exactly.
//
// KRK/KQK are swept in full (fast). KQKR is sampled by default and swept in full
// with `slow`. KRRK (duplicate pieces) is sampled — the only duplicate-group
// coverage of the device encode path.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "slider.hpp"                 // slider::init() — parity with sibling tests
#include "tb_comb_index.hpp"
#include "tb_comb_index_device.hpp"
#include "types.hpp"

namespace {

int g_failures = 0;
void fail(const char* m) { std::printf("[FAIL] %s\n", m); ++g_failures; }

// Diff the device-mirror against CombIndex over [0,size), sampling every `step`.
// step==1 sweeps the whole space.
void gate(std::vector<tb::Piece> extras, const char* name, std::size_t step) {
    tb::CombIndex idx(std::move(extras));
    const int men = idx.men();
    const std::size_t N = idx.size();

    tb::DeviceKingTable kt{};
    tb::DeviceMaterial  mat{};
    idx.fill_device(kt, mat);

    if (mat.men != men) { fail("fill_device men mismatch"); return; }

    std::size_t checked = 0;
    for (std::size_t i = 0; i < N; i += step) {
        // Host reference.
        int refsq[tb::kDevMaxMen];
        Color refstm;
        idx.decode(i, refsq, &refstm);
        const int ref_bit = (refstm == Color::Black) ? 1 : 0;

        // Device mirror (run on host here).
        int devsq[tb::kDevMaxMen];
        int dev_bit = -1;
        tb::comb_decode(i, devsq, &dev_bit, kt, mat);

        if (dev_bit != ref_bit) { fail("comb_decode stm != CombIndex::decode"); return; }
        for (int m = 0; m < men; ++m)
            if (devsq[m] != refsq[m]) { fail("comb_decode square != CombIndex::decode"); return; }

        // Both encode directions must return i (CombIndex is a self-bijection,
        // established by test_tb_comb_index; here we require the mirror to agree).
        const std::size_t ref_enc = idx.encode(refsq, refstm);
        const uint64_t    dev_enc = tb::comb_encode(devsq, dev_bit, kt, mat);
        if (ref_enc != i)        { fail("CombIndex::encode(decode(i)) != i"); return; }
        if (dev_enc != i)        { fail("comb_encode(comb_decode(i)) != i"); return; }
        ++checked;
    }
    std::printf("[ OK ] %-7s  men=%d  size=%zu  checked=%zu%s\n",
                name, men, N, checked, step == 1 ? " (full)" : " (sampled)");
}

}  // namespace

int main(int argc, char** argv) {
    slider::init();  // parity with the sibling tablebase tests' setup.
    const bool slow = (argc > 1 && std::string(argv[1]) == "slow");

    using PT = PieceType;
    const Color W = Color::White, B = Color::Black;

    // Full sweeps — small enough to enumerate every index.
    gate({{W, PT::Rook}},                 "KRK",  1);
    gate({{W, PT::Queen}},                "KQK",  1);

    // KQKR: ~3.49M comb indices. Full only in slow mode; sampled otherwise.
    gate({{W, PT::Queen}, {B, PT::Rook}}, "KQKR", slow ? 1 : 97);

    // KRRK: duplicate group — the device encode's insertion-sort / unordered-set
    // path. Sampled (its space is large); full in slow mode.
    gate({{W, PT::Rook}, {W, PT::Rook}},  "KRRK", slow ? 1 : 61);

    if (g_failures == 0) {
        std::printf("All device-mirror index checks match CombIndex.\n");
        return 0;
    }
    std::printf("\n%d device-mirror check(s) failed.\n", g_failures);
    return 1;
}
