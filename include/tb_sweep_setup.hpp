// Host-side setup for the device retrograde sweep (Phase 3).
//
// The GPU kernel only runs the per-node update (tb_sweep_device.hpp). Everything
// one-time — classifying positions, and solving the capture-exit sub-tables —
// happens here on the host, exactly mirroring solve_sweep_comb's pass 0 and
// sub-table build, then packaged as POD arrays to upload.
//
// build_sweep_setup() produces that package. sweep_host_reference() additionally
// runs the FULL sweep on the host using the CH_HD device functions (Jacobi to a
// fixpoint) — the local Phase 3 gate: its result is diffed against
// solve_sweep_comb with no GPU involved.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "tb_comb_index.hpp"
#include "tb_sweep_device.hpp"

namespace tb {

struct SweepSetup {
    std::size_t          N = 0;
    DeviceKingTable      kt{};      // aliases the CombIndex's KingTable storage
    DeviceMaterial       mat{};
    std::vector<uint8_t> state;     // N — SW_ILLEGAL/SOLVE/MATE/STALE
    std::vector<int16_t> v_init;    // N — MATE nodes pinned to -MATE, else 0

    // One capture-exit sub-table per main group. Owns the value storage; the
    // material descriptor is a self-contained POD copy. sub encode uses the MAIN
    // king table (the 462-config table is universal across materials).
    struct HostSub {
        int                  draw = 0;
        std::vector<int16_t> value;
        DeviceMaterial       mat{};
    };
    std::vector<HostSub> subs;      // indexed by group

    // Fill a DeviceSub[num_groups] viewing this setup's sub storage.
    void device_subs(DeviceSub* out) const {
        for (int g = 0; g < mat.num_groups; ++g) {
            out[g].draw  = subs[static_cast<std::size_t>(g)].draw;
            out[g].value = subs[static_cast<std::size_t>(g)].draw
                               ? nullptr : subs[static_cast<std::size_t>(g)].value.data();
            out[g].mat   = subs[static_cast<std::size_t>(g)].mat;
        }
    }
};

// Classify every index + solve the capture sub-tables (distinct pieces only).
SweepSetup build_sweep_setup(const CombIndex& idx);

// Run the device-shaped sweep entirely on the host (Jacobi ping-pong to a
// fixpoint). Returns value[] aligned with idx; *passes_out gets the iteration
// count if non-null. This is the local reference for the Phase 3 gate.
std::vector<int16_t> sweep_host_reference(const CombIndex& idx, int* passes_out = nullptr);

}  // namespace tb
