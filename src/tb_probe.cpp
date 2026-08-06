#include "tb_probe.hpp"

#include <memory>
#include <unordered_map>
#include <vector>

#include "bitboard.hpp"
#include "tb_index.hpp"
#include "tb_solve.hpp"

namespace tb {

namespace {

// A built + solved table for one material, kept alive in the cache. Index has no
// default constructor, so the cache stores these by unique_ptr.
struct Entry {
    Index index;
    Table table;
};

std::unordered_map<uint64_t, std::unique_ptr<Entry>> g_cache;
uint64_t g_tables_built = 0;

// Decompose a position into the tablebase's extra-piece list, in a canonical
// order (piece type ascending, then White before Black). Returns false — meaning
// "not a supported material, fall back to the heuristic" — for pawns, more than
// four men, or duplicate (color,type) pieces (identical-piece indexing is a
// later step). `extras` empty on success means bare KK.
bool extract_material(const Position& pos, std::vector<Piece>& extras) {
    if (pos.by_type[type_index(PieceType::Pawn)]) return false;  // pawns unsupported

    const int kt = type_index(PieceType::King);
    if (popcount(pos.by_type[kt] & pos.by_color[0]) != 1) return false;
    if (popcount(pos.by_type[kt] & pos.by_color[1]) != 1) return false;

    for (int t = type_index(PieceType::Knight); t <= type_index(PieceType::Queen); ++t)
        for (int c = 0; c < 2; ++c) {
            const int cnt = popcount(pos.by_type[t] & pos.by_color[c]);
            if (cnt == 0) continue;
            if (cnt > 1) return false;  // identical pieces (e.g. KRRK) unsupported
            extras.push_back({static_cast<Color>(c), static_cast<PieceType>(t)});
        }

    return extras.size() <= 2;  // 2 kings + <=2 extras == <=4 men
}

// A collision-free key for a canonical extras list (<=2 entries, each a distinct
// color/type in 0..11). Base-16 digits, +1 so leading entries never vanish.
uint64_t material_key(const std::vector<Piece>& extras) {
    uint64_t k = 0;
    for (const Piece& p : extras)
        k = k * 16 + static_cast<uint64_t>(color_index(p.color) * 6 + type_index(p.type) + 1);
    return k;
}

Entry& get_or_build(uint64_t key, const std::vector<Piece>& extras) {
    const auto it = g_cache.find(key);
    if (it != g_cache.end()) return *it->second;

    Index idx(extras);
    Table tbl = solve_sweep(idx);  // same solver the DAG uses; matches solve_bfs
    ++g_tables_built;

    auto entry = std::make_unique<Entry>(Entry{std::move(idx), std::move(tbl)});
    Entry& ref = *entry;
    g_cache.emplace(key, std::move(entry));
    return ref;
}

}  // namespace

ProbeResult probe(const Position& pos) {
    std::vector<Piece> extras;
    if (!extract_material(pos, extras)) return {};      // unsupported -> found=false
    if (extras.empty()) return {true, 0, 0};            // bare KK: draw

    Entry& e = get_or_build(material_key(extras), extras);
    const int16_t v = e.table.value[e.index.encode(pos)];
    if (v == 0) return {true, 0, 0};
    if (v > 0)  return {true, +1, win_dtm(v)};
    return {true, -1, loss_dtm(v)};
}

uint64_t tables_built() { return g_tables_built; }

void clear() { g_cache.clear(); }

}  // namespace tb
