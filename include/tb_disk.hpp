// Tablebase persistence — on-disk format + loader for solved CombIndex tables.
//
// The GPU-generated 5-man tables are computed, verified, then freed; nothing
// keeps them. This module gives them a durable home: a solved Table.value[] is
// written to a `.tb` file and loaded back into something the engine can probe,
// so a 5-man table generated once (on the GPU) is reusable without regenerating.
//
// The format targets tb::CombIndex — the same arithmetic index the GPU solve
// uses — because those are the expensive tables worth persisting. The dense
// tb::Index tops out at 4 men and rebuilds in seconds, so it isn't a target.
//
// A table is fully described by (extras, N, value[]): the extra-piece list
// reconstructs the CombIndex on load (radices are functions of the material), N
// is its size, and value[] is the int16 mate-score array (tb_solve.hpp scale).
// So the file is a small self-describing header + the raw int16 payload.
//
// v1 reads the payload into RAM rather than mmap'ing it: MinGW-w64 ships no
// <sys/mman.h>, a 5-man table (~419 MB int16) fits in RAM, and the in-RAM array
// probe is already O(1). The on-disk layout is mmap-friendly (payload is raw
// int16 at a fixed offset), so a memory-mapped loader is a drop-in later.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "position.hpp"
#include "tb_comb_index.hpp"
#include "tb_index.hpp"   // tb::Piece
#include "tb_solve.hpp"   // tb::Table

namespace tb {

// Write `tbl` (a CombIndex-aligned solved table) for material `idx` to `path`.
// Returns false (and prints a reason to stderr) on a size mismatch or I/O error.
bool write_table(const std::string& path, const CombIndex& idx, const Table& tbl);

// A loaded table: the value payload plus a CombIndex rebuilt from the stored
// material, so it can turn a Position into an exact result. Move-only (holds the
// index + a large value buffer); constructed only via open().
class DiskTable {
public:
    // Load + validate `path` (magic, version, checksum, and that the rebuilt
    // CombIndex size matches the stored N). Returns nullopt on any failure.
    static std::optional<DiskTable> open(const std::string& path);

    std::size_t               size()   const { return value_.size(); }
    const std::vector<Piece>& extras() const { return index_.extras(); }
    const std::string&        name()   const { return name_; }

    int16_t value_at(std::size_t i) const { return value_[i]; }

    // Exact result for `pos` (side-to-move relative, tb_solve.hpp scale). The
    // caller guarantees `pos` is this material with a legal king pair.
    int16_t probe(const Position& pos) const;

private:
    DiskTable(CombIndex idx, std::string name, std::vector<int16_t> value)
        : index_(std::move(idx)), name_(std::move(name)), value_(std::move(value)) {}

    CombIndex            index_;
    std::string          name_;
    std::vector<int16_t> value_;
};

}  // namespace tb
