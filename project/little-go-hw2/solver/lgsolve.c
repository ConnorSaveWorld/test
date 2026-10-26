/* Exact win/loss solver for 5x5 Little-Go (course host.py rules).
 * Offline analysis only; never part of the submission.
 * Board point p = 5*row + col. PASS = -1. Black = 1, White = 2.
 * State: B, W bitboards, side to move, ko (0 or 1 + P*25 + Q: the last move
 * placed P and captured exactly the single stone Q), count of moves already
 * played, last_pass flag. Game ends when count reaches 24 or on the second
 * consecutive pass. Black wins iff stones(B) - stones(W) >= 3 (komi 2.5).
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define FULL 0x1FFFFFFu
#define COL0 0x0108421u
#define COL4 0x1084210u
#define MAXMOVES 24

typedef uint32_t u32;
typedef uint64_t u64;

static inline u32 dilate(u32 g) {
    return (g | (g >> 5) | ((g << 5) & FULL) | ((g & ~COL0) >> 1) | ((g & ~COL4) << 1)) & FULL;
}

static inline u32 flood(u32 stones, u32 seed) {
    u32 g = seed, n;
    for (;;) {
        n = dilate(g) & stones;
        if (n == g) return g;
        g = n;
    }
}

/* Try to play point p for side with stones own/opp. Returns 0 if illegal.
 * On success writes new own/opp and new ko code. */
static inline int play(u32 own, u32 opp, int p, int ko, u32 *nown, u32 *nopp, int *nko) {
    u32 bit = 1u << p;
    if ((own | opp) & bit) return 0;
    own |= bit;
    u32 empty = ~(own | opp) & FULL;
    u32 cap = 0;
    u32 nb = dilate(bit) & ~bit & opp;
    while (nb) {
        u32 q = nb & (-nb);
        nb &= nb - 1;
        if (cap & q) continue;
        u32 g = flood(opp, q);
        if (!(dilate(g) & empty)) cap |= g;
        nb &= ~g;
    }
    opp &= ~cap;
    if (!cap) {
        u32 g = flood(own, bit);
        if (!(dilate(g) & empty)) return 0; /* suicide */
    }
    if (ko && cap && (cap & (cap - 1)) == 0) {
        int P = (ko - 1) / 25, Q = (ko - 1) % 25;
        if (p == Q && cap == (1u << P)) return 0;
    }
    *nown = own;
    *nopp = opp;
    *nko = (cap && (cap & (cap - 1)) == 0) ? 1 + p * 25 + __builtin_ctz(cap) : 0;
    return 1;
}

/* side: 1 black, 2 white. returns 1 if black wins final board */
static inline int black_wins(u32 B, u32 W) {
    return __builtin_popcount(B) - __builtin_popcount(W) >= 3;
}

/* ---------------- transposition table ---------------- */
typedef struct { u64 k1; u32 k2; int8_t val; int8_t best; int8_t pad[2]; } TTE;
static TTE *tt = 0;
static u64 ttmask = 0;
static u64 nodes = 0;


int lg_init(int log2size) {
    if (tt) free(tt);
    ttmask = (1ull << log2size) - 1;
    tt = (TTE *)calloc(ttmask + 1, sizeof(TTE));
    return tt != 0;
}
void lg_clear(void) { if (tt) memset(tt, 0, (ttmask + 1) * sizeof(TTE)); }
u64 lg_nodes(void) { return nodes; }

static inline u64 mkk1(u32 B, u32 W, int side, int count, int lp) {
    return (u64)B | ((u64)W << 25) | ((u64)count << 50) | ((u64)lp << 55) | ((u64)side << 56) | (1ull << 60);
}
static inline u64 hsh(u64 k1, u32 k2) {
    u64 h = k1 * 0x9E3779B97F4A7C15ull ^ ((u64)k2 * 0xC2B2AE3D27D4EB4Full);
    h ^= h >> 29; h *= 0xBF58476D1CE4E5B9ull; h ^= h >> 32;
    return h;
}

/* move ordering: static per-point preference (center first) + killer */
static const int8_t ORDER[25] = {12, 7, 11, 13, 17, 6, 8, 16, 18, 2, 10, 14, 22, 1, 3, 5, 9, 15, 19, 21, 23, 0, 4, 20, 24};
static int8_t killer[MAXMOVES + 2][2];
static u32 hist[3][26];
int lg_order_mode = 0;

/* returns 1 if side to move can force a win */
static int solve(u32 B, u32 W, int side, int ko, int count, int lp) {
    nodes++;
    u64 k1 = mkk1(B, W, side, count, lp);
    u32 k2 = (u32)ko;
    u64 idx = hsh(k1, k2) & ttmask;
    TTE *e = &tt[idx];
    if (e->k1 == k1 && e->k2 == k2) return e->val;
    int hint = (e->k1 == k1 && e->k2 == k2) ? e->best : -2;
    u32 own = side == 1 ? B : W, opp = side == 1 ? W : B;
    int win = 0, best = -2;
    int last = (count + 1 >= MAXMOVES);
    /* candidate list */
    int cand[27], nc = 0;
    if (hint >= -1) cand[nc++] = hint;
    for (int k = 0; k < 2; k++) {
        int m = killer[count][k];
        if (m >= 0 && m != hint) cand[nc++] = m;
    }
    int base = nc;
    for (int i = 0; i < 25; i++) {
        int m = ORDER[i];
        int dup = 0;
        for (int j = 0; j < base; j++) if (cand[j] == m) { dup = 1; break; }
        if (!dup) cand[nc++] = m;
    }
    if (lg_order_mode == 2) {
        /* tactical scoring: captures, ataris, escapes, avoid self-atari */
        int sc[27];
        u32 empty0 = ~(own | opp) & FULL;
        for (int i = base; i < nc; i++) {
            int m = cand[i]; u32 no, np; int nko;
            if (!play(own, opp, m, ko, &no, &np, &nko)) { sc[i] = -100000; continue; }
            int s = 0;
            int capn = __builtin_popcount(opp) - __builtin_popcount(np);
            s += capn * 1000;
            u32 emp = ~(no | np) & FULL;
            u32 bit = 1u << m;
            u32 g = flood(no, bit);
            int gl = __builtin_popcount(dilate(g) & emp);
            if (gl == 1) s -= 800; else if (gl == 2) s -= 100;
            /* opponent groups adjacent now in atari */
            u32 nb = dilate(bit) & np;
            while (nb) { u32 q = nb & -nb; u32 og = flood(np, q); nb &= ~og;
                int ol = __builtin_popcount(dilate(og) & emp);
                if (ol == 1) s += 300 * __builtin_popcount(og); else if (ol == 2) s += 30; }
            /* rescued own stones (own adjacent groups that had 1 liberty before) */
            u32 ob = dilate(bit) & own;
            while (ob) { u32 q = ob & -ob; u32 og = flood(own, q); ob &= ~og;
                if (__builtin_popcount(dilate(og) & empty0) == 1) s += 500 * __builtin_popcount(og); }
            sc[i] = s - i; /* keep ORDER as tiebreak */
        }
        for (int i = base + 1; i < nc; i++) {
            int m = cand[i], v = sc[i], j = i - 1;
            while (j >= base && sc[j] < v) { cand[j + 1] = cand[j]; sc[j + 1] = sc[j]; j--; }
            cand[j + 1] = m; sc[j + 1] = v;
        }
    } else if (lg_order_mode) {
        /* insertion sort the tail by history score (stable => ORDER tiebreak) */
        for (int i = base + 1; i < nc; i++) {
            int m = cand[i]; u32 h = hist[side][m]; int j = i - 1;
            while (j >= base && hist[side][cand[j]] < h) { cand[j + 1] = cand[j]; j--; }
            cand[j + 1] = m;
        }
    }
    if (hint != -1) cand[nc++] = -1;
    for (int i = 0; i < nc && !win; i++) {
        int m = cand[i];
        u32 no, np; int nko;
        u32 nB, nW;
        if (m == -1) {
            if (lp || last) { win = (black_wins(B, W) == (side == 1)); }
            else win = !solve(B, W, 3 - side, 0, count + 1, 1);
        } else {
            if (!play(own, opp, m, ko, &no, &np, &nko)) continue;
            if (side == 1) { nB = no; nW = np; } else { nB = np; nW = no; }
            if (last) win = (black_wins(nB, nW) == (side == 1));
            else win = !solve(nB, nW, 3 - side, nko, count + 1, 0);
        }
        if (win) {
            best = m;
            if (m >= 0 && killer[count][0] != m) { killer[count][1] = killer[count][0]; killer[count][0] = m; }
            if (m >= 0) hist[side][m] += (u32)(MAXMOVES - count) * (MAXMOVES - count);
        }
    }
    e = &tt[idx];
    e->k1 = k1; e->k2 = k2; e->val = (int8_t)win; e->best = (int8_t)best;
    return win;
}

/* Public: solve; returns 1 win / 0 loss for side to move. *best gets a winning move (or -2). */
int lg_solve(u32 B, u32 W, int side, int ko, int count, int lp, int *best) {
    memset(killer, -1, sizeof killer);
    int r = solve(B, W, side, ko, count, lp);
    if (best) {
        u64 k1 = mkk1(B, W, side, count, lp);
        TTE *e = &tt[hsh(k1, (u32)ko) & ttmask];
        *best = (e->k1 == k1 && e->k2 == (u32)ko) ? e->best : -2;
    }
    return r;
}

/* Public: value of each move. out[0..24] for points, out[25] for pass.
 * -1 illegal, 1 move wins for mover, 0 loses. */
void lg_moves(u32 B, u32 W, int side, int ko, int count, int lp, int *out) {
    u32 own = side == 1 ? B : W, opp = side == 1 ? W : B;
    int last = (count + 1 >= MAXMOVES);
    for (int m = -1; m < 25; m++) {
        int slot = m < 0 ? 25 : m;
        u32 no, np, nB, nW; int nko;
        if (m == -1) {
            if (lp || last) out[slot] = (black_wins(B, W) == (side == 1));
            else { memset(killer, -1, sizeof killer); out[slot] = !solve(B, W, 3 - side, 0, count + 1, 1); }
            continue;
        }
        if (!play(own, opp, m, ko, &no, &np, &nko)) { out[slot] = -1; continue; }
        if (side == 1) { nB = no; nW = np; } else { nB = np; nW = no; }
        if (last) out[slot] = (black_wins(nB, nW) == (side == 1));
        else { memset(killer, -1, sizeof killer); out[slot] = !solve(nB, nW, 3 - side, nko, count + 1, 0); }
    }
}

/* Public: apply a move; returns 0 if illegal. */
int lg_play(u32 B, u32 W, int side, int ko, int m, u32 *oB, u32 *oW, int *oko) {
    if (m == -1) { *oB = B; *oW = W; *oko = 0; return 1; }
    u32 own = side == 1 ? B : W, opp = side == 1 ? W : B, no, np;
    if (!play(own, opp, m, ko, &no, &np, oko)) return 0;
    if (side == 1) { *oB = no; *oW = np; } else { *oB = np; *oW = no; }
    return 1;
}

/* ---------------- depth-limited minimax (champion modelling) ---------------- */
static int g_evtype = 0;
static inline int libs_of(u32 own, u32 opp) {
    return __builtin_popcount(dilate(own) & ~own & ~opp & FULL);
}
/* value from perspective of side to move, in units of 1/2 stone (x10 for extras) */
static int evalpos(u32 B, u32 W, int side) {
    int d = 2 * (__builtin_popcount(B) - __builtin_popcount(W)) - 5; /* black perspective, komi 2.5 */
    int v = d * 10;
    if (g_evtype == 1) v += 5 * (libs_of(B, W) - libs_of(W, B));
    if (g_evtype == 2) v += 1 * (libs_of(B, W) - libs_of(W, B));
    return side == 1 ? v : -v;
}
static int ab(u32 B, u32 W, int side, int ko, int count, int lp, int depth, int alpha, int beta) {
    nodes++;
    if (count >= MAXMOVES) return evalpos(B, W, side);
    if (depth == 0) return evalpos(B, W, side);
    u32 own = side == 1 ? B : W, opp = side == 1 ? W : B;
    int best = -1000000;
    for (int i = 0; i < 25; i++) {
        int m = ORDER[i];
        u32 no, np, nB, nW; int nko;
        if (!play(own, opp, m, ko, &no, &np, &nko)) continue;
        if (side == 1) { nB = no; nW = np; } else { nB = np; nW = no; }
        int v = -ab(nB, nW, 3 - side, nko, count + 1, 0, depth - 1, -beta, -alpha);
        if (v > best) best = v;
        if (v > alpha) alpha = v;
        if (alpha >= beta) return best;
    }
    /* pass */
    {
        int v;
        if (lp) v = evalpos(B, W, side);
        else v = -ab(B, W, 3 - side, 0, count + 1, 1, depth - 1, -beta, -alpha);
        if (v > best) best = v;
    }
    return best;
}
/* root values for every move at given depth; out[25] = pass; -9999999 illegal */
void lg_minimax(u32 B, u32 W, int side, int ko, int count, int lp, int depth, int evtype, int *out) {
    g_evtype = evtype;
    u32 own = side == 1 ? B : W, opp = side == 1 ? W : B;
    for (int m = -1; m < 25; m++) {
        int slot = m < 0 ? 25 : m;
        u32 no, np, nB, nW; int nko;
        if (m == -1) {
            if (lp || count + 1 >= MAXMOVES) out[slot] = evalpos(B, W, side);
            else out[slot] = -ab(B, W, 3 - side, 0, count + 1, 1, depth - 1, -1000000, 1000000);
            continue;
        }
        if (!play(own, opp, m, ko, &no, &np, &nko)) { out[slot] = -9999999; continue; }
        if (side == 1) { nB = no; nW = np; } else { nB = np; nW = no; }
        out[slot] = -ab(nB, nW, 3 - side, nko, count + 1, 0, depth - 1, -1000000, 1000000);
    }
}
void lg_set_order(int m) { lg_order_mode = m; }
void lg_dummy(void){}
