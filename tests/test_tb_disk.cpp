// Persistence gate: a solved CombIndex table written to disk and loaded back
// must be identical, and the loaded table must probe real positions correctly.
//
// For KRK and KQK (3-man, solved by solve_sweep_comb in well under a second):
//   (1) full payload equality  — value_at(i) == solved value[i] for every i,
//   (2) probe path             — DiskTable::probe(pos) == solved value at
//                                comb.encode(pos) for every legal (dense) position,
//   (3) integrity              — flipping one payload byte makes open() reject it.
//
// Together these prove the format is lossless and the rebuilt-on-load CombIndex
// probes the same values the solver produced.
//
// No <filesystem>: the rest of the suite avoids it, and on this MinGW an older
// libstdc++ on PATH lacks its runtime entry points. Plain stdio + $TEMP suffices.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "slider.hpp"
#include "tb_comb_index.hpp"
#include "tb_disk.hpp"
#include "tb_index.hpp"
#include "tb_probe.hpp"
#include "tb_solve.hpp"
#include "types.hpp"

namespace {

int g_failures = 0;
void fail(const char* m) { std::printf("[FAIL] %s\n", m); ++g_failures; }

std::string temp_path(const char* nm) {
    const char* dir = std::getenv("TEMP");
    if (!dir) dir = std::getenv("TMP");
    const std::string base = dir ? std::string(dir) : std::string(".");
    return base + "/chess_test_tb_disk_" + nm + ".tb";
}

void check(std::vector<tb::Piece> extras, const char* nm) {
    tb::CombIndex comb(extras);
    tb::Table     t = tb::solve_sweep_comb(comb);

    const std::string p = temp_path(nm);

    if (!tb::write_table(p, comb, t)) { fail("write_table failed"); return; }

    auto loaded = tb::DiskTable::open(p);
    if (!loaded)                           { fail("open failed"); std::remove(p.c_str()); return; }
    if (loaded->size() != comb.size())     { fail("size mismatch"); std::remove(p.c_str()); return; }
    if (loaded->name() != std::string(nm)) { fail("name mismatch"); }

    // (1) Full payload equality.
    for (std::size_t i = 0; i < comb.size(); ++i)
        if (loaded->value_at(i) != t.value[i]) { fail("value mismatch"); std::remove(p.c_str()); return; }

    // (2) Probe path over every legal position (dense enumeration is the oracle
    // of legality here, mirroring test_tb_comb_solve).
    tb::Index dense(extras);
    for (std::size_t d = 0; d < dense.size(); ++d) {
        const Position pos = dense.decode(d);
        const std::size_t c = comb.encode(pos);
        if (c >= comb.size()) { fail("comb index out of range"); std::remove(p.c_str()); return; }
        if (loaded->probe(pos) != t.value[c]) { fail("probe mismatch"); std::remove(p.c_str()); return; }
    }

    // (3) Corruption is rejected: flip one payload byte (past the 128-byte header).
    {
        FILE* rf = std::fopen(p.c_str(), "rb");
        std::fseek(rf, 0, SEEK_END);
        const long sz = std::ftell(rf);
        std::fseek(rf, 0, SEEK_SET);
        std::vector<char> buf(static_cast<std::size_t>(sz));
        if (std::fread(buf.data(), 1, static_cast<std::size_t>(sz), rf) != static_cast<std::size_t>(sz)) fail("reread failed");
        std::fclose(rf);
        buf[200] = static_cast<char>(buf[200] ^ 0xFF);
        FILE* wf = std::fopen(p.c_str(), "wb");
        std::fwrite(buf.data(), 1, static_cast<std::size_t>(sz), wf);
        std::fclose(wf);
    }
    if (tb::DiskTable::open(p)) fail("corrupt file was not rejected");

    std::remove(p.c_str());
    std::printf("[ OK ] %-4s  N=%zu  round-trips + probes + corruption-detected\n", nm, comb.size());
}

// The search's probe path reads the on-disk table: persist KRK under its
// canonical name, point tb::set_table_dir at it, and confirm tb::probe() returns
// the disk-backed result matching the dense oracle for every legal position.
void check_probe_wiring() {
    std::vector<tb::Piece> extras = {{Color::White, PieceType::Rook}};
    tb::CombIndex comb(extras);
    tb::Table     ct = tb::solve_sweep_comb(comb);

    const char* dir = std::getenv("TEMP");
    if (!dir) dir = std::getenv("TMP");
    const std::string d = dir ? std::string(dir) : std::string(".");
    const std::string p = d + "/KRK.tb";  // canonical name so the probe finds it
    if (!tb::write_table(p, comb, ct)) { fail("probe-wiring: write failed"); return; }

    tb::set_table_dir(d);
    tb::Index dense(extras);
    tb::Table dt = tb::solve_sweep(dense);

    int checked = 0;
    for (std::size_t i = 0; i < dense.size(); ++i) {
        const Position pos = dense.decode(i);
        const tb::ProbeResult r = tb::probe(pos);
        const int16_t dv  = dt.value[i];
        const int     wdl = dv > 0 ? 1 : dv < 0 ? -1 : 0;
        const int     dtm = dv > 0 ? tb::win_dtm(dv) : dv < 0 ? tb::loss_dtm(dv) : 0;
        if (!r.found || r.wdl != wdl || r.dtm != dtm) { fail("probe-wiring: mismatch vs dense"); break; }
        ++checked;
    }

    tb::set_table_dir("");  // disable disk probing so other tests are unaffected
    std::remove(p.c_str());
    if (checked == static_cast<int>(dense.size()))
        std::printf("[ OK ] probe wiring: %d KRK positions via disk match dense oracle\n", checked);
}

}  // namespace

int main() {
    slider::init();
    using PT = PieceType;
    const Color W = Color::White;

    check({{W, PT::Rook}},  "KRK");
    check({{W, PT::Queen}}, "KQK");
    check_probe_wiring();

    if (g_failures == 0) { std::printf("\nAll tb_disk persistence tests passed.\n"); return 0; }
    std::printf("\n%d tb_disk test(s) failed.\n", g_failures);
    return 1;
}
