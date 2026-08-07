// Device-side magic sliders — the GPU port of slider.cpp's fast attack path.
//
// The magic SEARCH (finding collision-free multipliers) stays on the host: it's
// a one-time, PRNG-driven build, not device work. Its RESULT — per-square
// {mask, magic, shift, table offset} + the two flat attack tables — is uploaded
// to the device once, and these CH_HD functions do the O(1) query (one masked
// multiply, one shift, one table load) identically on host and device. This is
// exactly why the engine uses multiply-shift magics and not BMI2 PEXT: PEXT has
// no CUDA instruction, so magics port verbatim (see slider.hpp).
//
// Lean by design: pulls in only types.hpp (POD Bitboard) + cuda_compat.hpp, so
// a .cu including it stays a small device-clean TU. The host fills a DeviceSliders
// with slider::get_device_sliders() (pointers into its internal tables); the .cu
// copies those arrays to device memory and rebinds the pointers.

#pragma once

#include "cuda_compat.hpp"
#include "types.hpp"

namespace slider {

// One square's magic entry, self-contained (offset into the shared attack table
// instead of an absolute pointer, so it survives being copied to the device).
struct DeviceMagic {
    Bitboard mask;
    Bitboard magic;
    int      offset;   // base index into the rook/bishop attack table
    unsigned shift;    // 64 - popcount(mask)
};

// The complete device slider dataset. On the host these pointers alias slider.cpp
// internals; the .cu replaces them with device pointers after cudaMemcpy.
struct DeviceSliders {
    const DeviceMagic* rook;          // [64]
    const DeviceMagic* bishop;        // [64]
    const Bitboard*    rook_table;    // [rook_table_size]
    const Bitboard*    bishop_table;  // [bishop_table_size]
    int rook_table_size;
    int bishop_table_size;
};

CH_HD inline unsigned magic_index(const DeviceMagic& m, Bitboard occ) {
    return static_cast<unsigned>(((occ & m.mask) * m.magic) >> m.shift);
}

CH_HD inline Bitboard rook_attacks_dev(const DeviceSliders& s, int sq, Bitboard occ) {
    const DeviceMagic& m = s.rook[sq];
    return s.rook_table[m.offset + magic_index(m, occ)];
}

CH_HD inline Bitboard bishop_attacks_dev(const DeviceSliders& s, int sq, Bitboard occ) {
    const DeviceMagic& m = s.bishop[sq];
    return s.bishop_table[m.offset + magic_index(m, occ)];
}

CH_HD inline Bitboard queen_attacks_dev(const DeviceSliders& s, int sq, Bitboard occ) {
    return rook_attacks_dev(s, sq, occ) | bishop_attacks_dev(s, sq, occ);
}

}  // namespace slider
