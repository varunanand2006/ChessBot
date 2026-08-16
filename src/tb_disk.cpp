#include "tb_disk.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "tb_material.hpp"  // type_letter
#include "types.hpp"

// I/O uses C stdio (fopen/fwrite/fread), not <fstream>: it matches the rest of
// the codebase's C-style, avoids pulling in iostream/locale, and dodges a
// libstdc++ DLL-ABI mismatch that crashes iostream at runtime on this MinGW.

namespace tb {

namespace {

// Fixed 128-byte little-endian header. Field order is chosen so every member is
// naturally aligned and the struct is exactly 128 bytes with no padding, so the
// int16 payload begins at offset 128 (2-byte aligned) — mmap-friendly for later.
constexpr char     kMagic[8]  = {'C', 'H', 'E', 'S', 'S', 'T', 'B', '\0'};
constexpr uint32_t kVersion   = 1;
constexpr uint32_t kSchemeComb = 0;  // CombIndex layout (the only one for now)
constexpr int      kMaxExtras  = 6;

struct Header {
    char     magic[8];
    uint32_t version;
    uint32_t scheme;
    uint64_t n_entries;
    uint32_t men;
    uint32_t extras_count;
    uint8_t  extras[2 * kMaxExtras];  // (color, type) byte pairs
    char     name[20];                // human-readable, null-padded (e.g. "KQKR")
    uint64_t checksum;                // FNV-1a of the value payload
    uint8_t  reserved[56];
};
static_assert(sizeof(Header) == 128, "on-disk header must be exactly 128 bytes");

// FNV-1a 64: a non-cryptographic integrity check, enough to catch truncation or
// bit-rot on load. Not a security boundary.
uint64_t fnv1a(const void* data, std::size_t len) {
    const auto* p = static_cast<const uint8_t*>(data);
    uint64_t h = 1469598103934665603ull;
    for (std::size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

}  // namespace

bool write_table(const std::string& path, const CombIndex& idx, const Table& tbl) {
    if (tbl.value.size() != idx.size()) {
        std::fprintf(stderr, "write_table: value size %zu != index size %zu\n",
                     tbl.value.size(), idx.size());
        return false;
    }
    const std::vector<Piece>& extras = idx.extras();
    if (extras.size() > static_cast<std::size_t>(kMaxExtras)) {
        std::fprintf(stderr, "write_table: %zu extras exceeds max %d\n", extras.size(), kMaxExtras);
        return false;
    }

    Header h{};
    std::memcpy(h.magic, kMagic, sizeof(h.magic));
    h.version      = kVersion;
    h.scheme       = kSchemeComb;
    h.n_entries    = idx.size();
    h.men          = static_cast<uint32_t>(idx.men());
    h.extras_count = static_cast<uint32_t>(extras.size());
    for (std::size_t i = 0; i < extras.size(); ++i) {
        h.extras[2 * i]     = static_cast<uint8_t>(color_index(extras[i].color));
        h.extras[2 * i + 1] = static_cast<uint8_t>(type_index(extras[i].type));
    }
    const std::string name = material_name_canonical(extras);
    std::memcpy(h.name, name.c_str(), std::min(name.size(), sizeof(h.name) - 1));

    const std::size_t n = tbl.value.size();
    h.checksum = fnv1a(tbl.value.data(), n * sizeof(int16_t));

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "write_table: cannot open %s for writing\n", path.c_str()); return false; }
    const bool ok = std::fwrite(&h, sizeof(h), 1, f) == 1 &&
                    std::fwrite(tbl.value.data(), sizeof(int16_t), n, f) == n;
    std::fclose(f);
    if (!ok) { std::fprintf(stderr, "write_table: write failed for %s\n", path.c_str()); return false; }
    return true;
}

std::optional<DiskTable> DiskTable::open(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "DiskTable: cannot open %s\n", path.c_str()); return std::nullopt; }

    Header h{};
    if (std::fread(&h, sizeof(h), 1, f) != 1) {
        std::fprintf(stderr, "DiskTable: %s too short for header\n", path.c_str());
        std::fclose(f);
        return std::nullopt;
    }
    if (std::memcmp(h.magic, kMagic, sizeof(kMagic)) != 0) {
        std::fprintf(stderr, "DiskTable: %s bad magic\n", path.c_str());
        std::fclose(f);
        return std::nullopt;
    }
    if (h.version != kVersion) {
        std::fprintf(stderr, "DiskTable: %s unsupported version %u\n", path.c_str(), h.version);
        std::fclose(f);
        return std::nullopt;
    }
    if (h.scheme != kSchemeComb) {
        std::fprintf(stderr, "DiskTable: %s unsupported index scheme %u\n", path.c_str(), h.scheme);
        std::fclose(f);
        return std::nullopt;
    }
    if (h.extras_count > static_cast<uint32_t>(kMaxExtras)) {
        std::fprintf(stderr, "DiskTable: %s bad extras_count %u\n", path.c_str(), h.extras_count);
        std::fclose(f);
        return std::nullopt;
    }

    std::vector<Piece> extras(h.extras_count);
    for (uint32_t i = 0; i < h.extras_count; ++i) {
        extras[i].color = static_cast<Color>(h.extras[2 * i]);
        extras[i].type  = static_cast<PieceType>(h.extras[2 * i + 1]);
    }

    std::vector<int16_t> value(h.n_entries);
    const std::size_t bytes = static_cast<std::size_t>(h.n_entries) * sizeof(int16_t);
    const bool read_ok = std::fread(value.data(), sizeof(int16_t), h.n_entries, f) == h.n_entries;
    std::fclose(f);
    if (!read_ok) {
        std::fprintf(stderr, "DiskTable: %s payload truncated\n", path.c_str());
        return std::nullopt;
    }
    if (fnv1a(value.data(), bytes) != h.checksum) {
        std::fprintf(stderr, "DiskTable: %s checksum mismatch (corrupt)\n", path.c_str());
        return std::nullopt;
    }

    // Rebuild the index from the stored material and confirm it agrees with the
    // stored size — a mismatch means the file's material and payload disagree.
    CombIndex idx(extras);
    if (idx.size() != h.n_entries) {
        std::fprintf(stderr, "DiskTable: %s index size %zu != stored N %llu\n",
                     path.c_str(), idx.size(), static_cast<unsigned long long>(h.n_entries));
        return std::nullopt;
    }

    std::size_t nlen = 0;
    while (nlen < sizeof(h.name) && h.name[nlen] != '\0') ++nlen;
    return DiskTable(std::move(idx), std::string(h.name, nlen), std::move(value));
}

int16_t DiskTable::probe(const Position& pos) const {
    const std::size_t i = index_.encode(pos);
    return i < value_.size() ? value_[i] : int16_t{0};
}

}  // namespace tb
