// Tablebase material-name parsing, shared by the CLI (src/main.cpp) and the CUDA
// solve/dump harness (cuda/sweep_check.cu). Header-only: it is pure string/enum
// glue with no hot-path role, and both a plain-C++ TU and an nvcc TU include it.
//
// A material name is Nalimov-style — "KQKR", "KRKN", "KQRKR", "KQK" — where the
// piece letters before the second 'K' are White's extras and those after are
// Black's. The single-letter legacy form "Q|R|B|N" means KXK (one White piece).
//
// `max_extras` caps how many non-king pieces the caller's indexer can handle:
//   * the dense tb::Index tops out at 4 men  -> pass 2,
//   * the combinatorial CombIndex handles 5+ -> pass up to 6.
// Distinct (color,type) is still required here because both current SOLVERS
// detect a capture by which (color,type) bitboard emptied (ambiguous for
// duplicates); the CombIndex itself can index duplicates, but nothing solves
// them yet, so the parser rejects them rather than produce an unsolvable table.

#pragma once

#include <cctype>
#include <string>
#include <vector>

#include "tb_index.hpp"  // tb::Piece
#include "types.hpp"

namespace tb {

inline bool letter_to_type(char c, PieceType& pt) {
    switch (std::toupper(static_cast<unsigned char>(c))) {
        case 'Q': pt = PieceType::Queen;  return true;
        case 'R': pt = PieceType::Rook;   return true;
        case 'B': pt = PieceType::Bishop; return true;
        case 'N': pt = PieceType::Knight; return true;
        default: return false;
    }
}

inline const char* type_letter(PieceType pt) {
    switch (pt) {
        case PieceType::Queen:  return "Q";
        case PieceType::Rook:   return "R";
        case PieceType::Bishop: return "B";
        case PieceType::Knight: return "N";
        default:                return "?";
    }
}

// Parse `s` into the extra-piece list + canonical name. Returns false (and may
// print a reason to stderr) if the string is malformed, exceeds `max_extras`, or
// names identical pieces. Kings are implicit and never in `extras`.
inline bool parse_material(const char* s, std::vector<Piece>& extras,
                           std::string& name, int max_extras = 2) {
    if (!s || s[0] == '\0') return false;

    // Legacy single-letter KXK (a single White piece).
    if (s[1] == '\0') {
        PieceType pt;
        if (!letter_to_type(s[0], pt)) return false;
        extras.push_back({Color::White, pt});
        name = std::string("K") + type_letter(pt) + "K";
        return true;
    }

    std::string str(s);
    for (char& c : str) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    if (str.size() < 3 || str[0] != 'K') return false;
    const std::size_t k2 = str.find('K', 1);
    if (k2 == std::string::npos) return false;
    if (str.find('K', k2 + 1) != std::string::npos) return false;  // exactly two kings

    for (std::size_t i = 1; i < k2; ++i) {
        PieceType pt;
        if (!letter_to_type(str[i], pt)) return false;
        extras.push_back({Color::White, pt});
    }
    for (std::size_t i = k2 + 1; i < str.size(); ++i) {
        PieceType pt;
        if (!letter_to_type(str[i], pt)) return false;
        extras.push_back({Color::Black, pt});
    }
    if (extras.empty() || static_cast<int>(extras.size()) > max_extras) return false;

    // Distinct (color,type) required: identical pieces need a count-based capture
    // detection no solver has yet.
    for (std::size_t a = 0; a < extras.size(); ++a)
        for (std::size_t b = a + 1; b < extras.size(); ++b)
            if (extras[a].color == extras[b].color && extras[a].type == extras[b].type)
                return false;

    name = "K";
    for (const Piece& p : extras) if (p.color == Color::White) name += type_letter(p.type);
    name += "K";
    for (const Piece& p : extras) if (p.color == Color::Black) name += type_letter(p.type);
    return true;
}

}  // namespace tb
