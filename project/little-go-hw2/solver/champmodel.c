/* Candidate models of the Championship player (offline analysis only).
 * mcts_choose: plain UCT with uniformly random playouts; returns the most
 * visited root move. Rules are shared with lgsolve.c (included). */
#include "lgsolve.c"
#include <math.h>

static uint64_t rng_s = 88172645463325252ull;
static inline uint64_t xr(void) { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return rng_s; }
void cm_seed(uint64_t s) { rng_s = s ? s : 88172645463325252ull; }
static int g_policy = 0;   /* 0 uniform random playouts, 1 avoid filling own eyes */
static int g_select = 0;   /* 0 most visited root child, 1 best mean */
void cm_config(int policy, int select) { g_policy = policy; g_select = select; }
static inline int own_eye(u32 own, int m) {
    u32 nb = dilate(1u << m) & ~(1u << m);
    return (nb & own) == nb;
}

/* random playout; returns 1 if black wins */
static int playout(u32 B, u32 W, int side, int ko, int count, int lp) {
    while (count < MAXMOVES) {
        u32 own = side == 1 ? B : W, opp = side == 1 ? W : B;
        int moves[25], n = 0;
        u32 empty = ~(B | W) & FULL;
        while (empty) { int p = __builtin_ctz(empty); empty &= empty - 1; moves[n++] = p; }
        int played = 0;
        while (n) {
            int i = xr() % n; int m = moves[i]; moves[i] = moves[--n];
            if (g_policy == 1 && own_eye(own, m)) continue;
            u32 no, np; int nko;
            if (play(own, opp, m, ko, &no, &np, &nko)) {
                if (side == 1) { B = no; W = np; } else { B = np; W = no; }
                ko = nko; lp = 0; played = 1; break;
            }
        }
        if (!played) { if (lp) break; lp = 1; ko = 0; }
        side = 3 - side; count++;
    }
    return black_wins(B, W);
}

#define MAXN 400000
typedef struct { u32 B, W; int8_t side, count, lp, move; int16_t ko; int parent, first, nchild, untried; double wins; int visits; int8_t cand[26]; int8_t ncand; } Node;
static Node *nodesarr = 0;
static int nn = 0;

static int mknode(u32 B, u32 W, int side, int ko, int count, int lp, int parent, int move) {
    Node *x = &nodesarr[nn];
    x->B = B; x->W = W; x->side = side; x->ko = ko; x->count = count; x->lp = lp; x->parent = parent; x->move = move;
    x->first = -1; x->nchild = 0; x->wins = 0; x->visits = 0; x->ncand = 0;
    if (count < MAXMOVES) {
        u32 own = side == 1 ? B : W, opp = side == 1 ? W : B;
        for (int m = 0; m < 25; m++) { u32 a, b; int k; if (play(own, opp, m, ko, &a, &b, &k)) x->cand[x->ncand++] = m; }
        x->cand[x->ncand++] = -1; /* pass */
    }
    x->untried = x->ncand;
    return nn++;
}

/* children stored contiguously? no: linked via first + nchild with indices in order of creation */
static int child_idx[MAXN][27];

int mcts_choose(u32 B, u32 W, int side, int ko, int count, int lp, int iters, double c, int *visits_out) {
    if (!nodesarr) nodesarr = (Node *)malloc(sizeof(Node) * MAXN);
    nn = 0;
    int root = mknode(B, W, side, ko, count, lp, -1, -2);
    for (int it = 0; it < iters && nn < MAXN - 30; it++) {
        int cur = root;
        /* selection */
        while (nodesarr[cur].untried == 0 && nodesarr[cur].nchild > 0) {
            Node *x = &nodesarr[cur];
            double best = -1e9; int bi = -1;
            double lnN = log((double)x->visits + 1);
            for (int i = 0; i < x->nchild; i++) {
                Node *y = &nodesarr[child_idx[cur][i]];
                double q = y->wins / (y->visits + 1e-9);
                double u = q + c * sqrt(lnN / (y->visits + 1e-9));
                if (u > best) { best = u; bi = child_idx[cur][i]; }
            }
            cur = bi;
        }
        Node *x = &nodesarr[cur];
        /* expansion */
        if (x->untried > 0 && x->count < MAXMOVES) {
            int i = xr() % x->untried; int m = x->cand[i];
            x->cand[i] = x->cand[x->untried - 1]; x->cand[x->untried - 1] = m; x->untried--;
            u32 nB = x->B, nW = x->W; int nko = 0, nlp = 0;
            if (m == -1) { nlp = 1; }
            else { lg_play(x->B, x->W, x->side, x->ko, m, &nB, &nW, &nko); }
            int ch;
            if (m == -1 && x->lp) { ch = mknode(nB, nW, 3 - x->side, 0, MAXMOVES, 1, cur, m); }
            else ch = mknode(nB, nW, 3 - x->side, nko, x->count + 1, nlp, cur, m);
            child_idx[cur][nodesarr[cur].nchild++] = ch;
            cur = ch; x = &nodesarr[cur];
        }
        int bw = playout(x->B, x->W, x->side, x->ko, x->count, x->lp);
        /* backprop: wins counted from perspective of player who moved into node */
        while (cur >= 0) {
            Node *y = &nodesarr[cur];
            y->visits++;
            int mover = 3 - y->side; /* player who made the move leading here */
            if ((bw && mover == 1) || (!bw && mover == 2)) y->wins += 1;
            cur = y->parent;
        }
    }
    int bm = -2, bv = -1;
    for (int i = 0; i < 26; i++) visits_out[i] = 0;
    for (int i = 0; i < nodesarr[root].nchild; i++) {
        Node *y = &nodesarr[child_idx[root][i]];
        visits_out[y->move < 0 ? 25 : y->move] = y->visits;
        if (g_select == 0) { if (y->visits > bv) { bv = y->visits; bm = y->move; } }
        else { int sc = (int)(1e6 * y->wins / (y->visits + 1e-9)); if (y->visits >= 5 && sc > bv) { bv = sc; bm = y->move; } }
    }
    return bm;
}
