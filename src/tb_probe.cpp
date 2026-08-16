#include "tb_probe.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "bitboard.hpp"
#include "tb_disk.hpp"
#include "tb_index.hpp"
#include "tb_material.hpp"  // material_name_canonical
#include "tb_solve.hpp"

namespace tb {

namespace {

// A built + solved dense table for one material, kept alive in the cache. Index
// has no default constructor, so the cache stores these by unique_ptr.
struct Entry {
    Index index;
    Table table;
};

std::unordered_map<uint64_t, std::unique_ptr<Entry>> g_cache;

// On-disk table state: where to look, the loaded tables (by canonical name), and
// materials with no file (so a missing table isn't re-opened every probe).
std::string                                                g_table_dir;
std::unordered_map<std::string, std::unique_ptr<DiskTable>> g_disk;
std::unordered_set<std::string>                            g_disk_absent;

// Cap on extras extracted: high enough for any persisted table we could plausibly
// have (5-man = 3 extras); the dense in-RAM path is separately capped at <=2.
constexpr std::size_t kMaxProbeExtras = 5;

// Decompose a position into its extra-piece list. Returns false — "unsupported,
// use the heuristic" — for pawns or duplicate (color,type) pieces (identical
// pieces need count-based capture handling no solver has yet). `extras` empty on
// success means bare KK.
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

    return extras.size() <= kMaxProbeExtras;
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

    auto entry = std::make_unique<Entry>(Entry{std::move(idx), std::move(tbl)});
    Entry& ref = *entry;
    g_cache.emplace(key, std::move(entry));
    return ref;
}

// Return a cached/loaded on-disk table for this material, or nullptr if disk
// probing is off or no file exists. The file must be named canonically
// (<dir>/<KQKR>.tb); a quiet existence check keeps a missing table silent (a
// normal case at a search node) while DiskTable::open still reports corruption.
DiskTable* find_disk_table(const std::vector<Piece>& extras) {
    if (g_table_dir.empty()) return nullptr;
    const std::string name = material_name_canonical(extras);

    if (const auto it = g_disk.find(name); it != g_disk.end()) return it->second.get();
    if (g_disk_absent.count(name)) return nullptr;

    const std::string path = g_table_dir + "/" + name + ".tb";
    if (FILE* test = std::fopen(path.c_str(), "rb")) { std::fclose(test); }
    else { g_disk_absent.insert(name); return nullptr; }

    auto loaded = DiskTable::open(path);
    if (!loaded) { g_disk_absent.insert(name); return nullptr; }  // corrupt -> decline
    const auto ins = g_disk.emplace(name, std::make_unique<DiskTable>(std::move(*loaded)));
    return ins.first->second.get();
}

ProbeResult from_value(int16_t v) {
    if (v == 0) return {true, 0, 0};
    if (v > 0)  return {true, +1, win_dtm(v)};
    return {true, -1, loss_dtm(v)};
}

}  // namespace

void set_table_dir(const std::string& dir) { g_table_dir = dir; }

ProbeResult probe(const Position& pos) {
    std::vector<Piece> extras;
    if (!extract_material(pos, extras)) return {};      // unsupported -> found=false
    if (extras.empty()) return {true, 0, 0};            // bare KK: draw

    // Prefer a persisted table (covers 5-man and anything on disk).
    if (DiskTable* dt = find_disk_table(extras)) return from_value(dt->probe(pos));

    // Otherwise build the dense table in RAM — <=4-man only.
    if (extras.size() <= 2) {
        Entry& e = get_or_build(material_key(extras), extras);
        return from_value(e.table.value[e.index.encode(pos)]);
    }

    return {};  // 5-man material with no persisted table -> heuristic
}

}  // namespace tb
