/*
 * adas_eval_native.c — ADAS 皮层闭环评估器的原生 C 实现（任务层）。
 *
 * 逐语句复刻 tools/train_adas_cortex.py 中的
 *   SdscCell.forward_fast / AdasCortexOrgan.forward / run_scenario / evaluate
 * 包括 Python 的 MT19937 + random.gauss() 噪声流，目标是与 Python 评估
 * **位级一致**（double 运算、同一 libm、同一求和顺序），而不是"差不多"。
 *
 * 用途：把 CMA-ES / 演化的内循环从 0.9 s/评估 压到毫秒级。
 * 这是任务层适配代码，不属于 include/kun/cellular/ 通用底座。
 *
 * 构建：gcc -O2 -fopenmp -shared -fPIC -o build/libadas_eval_native.so tools/adas_eval_native.c -lm
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ── Python random.Random 复刻 ─────────────────────────────── */
typedef struct {
    uint32_t mt[624];
    int mti;
    int has_gauss;
    double gauss_next;
} PyRandom;

static void mt_init_genrand(PyRandom *r, uint32_t s) {
    r->mt[0] = s;
    for (int i = 1; i < 624; i++)
        r->mt[i] = 1812433253U * (r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) + (uint32_t)i;
    r->mti = 624;
}

static void mt_init_by_array(PyRandom *r, const uint32_t *key, int klen) {
    mt_init_genrand(r, 19650218U);
    int i = 1, j = 0, k = 624 > klen ? 624 : klen;
    for (; k; k--) {
        r->mt[i] = (r->mt[i] ^ ((r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) * 1664525U)) + key[j] + (uint32_t)j;
        i++; j++;
        if (i >= 624) { r->mt[0] = r->mt[623]; i = 1; }
        if (j >= klen) j = 0;
    }
    for (k = 623; k; k--) {
        r->mt[i] = (r->mt[i] ^ ((r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) * 1566083941U)) - (uint32_t)i;
        i++;
        if (i >= 624) { r->mt[0] = r->mt[623]; i = 1; }
    }
    r->mt[0] = 0x80000000U;
}

static uint32_t mt_genrand(PyRandom *r) {
    static const uint32_t mag01[2] = {0U, 0x9908b0dfU};
    uint32_t y;
    if (r->mti >= 624) {
        int kk;
        for (kk = 0; kk < 624 - 397; kk++) {
            y = (r->mt[kk] & 0x80000000U) | (r->mt[kk + 1] & 0x7fffffffU);
            r->mt[kk] = r->mt[kk + 397] ^ (y >> 1) ^ mag01[y & 1U];
        }
        for (; kk < 623; kk++) {
            y = (r->mt[kk] & 0x80000000U) | (r->mt[kk + 1] & 0x7fffffffU);
            r->mt[kk] = r->mt[kk + (397 - 624)] ^ (y >> 1) ^ mag01[y & 1U];
        }
        y = (r->mt[623] & 0x80000000U) | (r->mt[0] & 0x7fffffffU);
        r->mt[623] = r->mt[396] ^ (y >> 1) ^ mag01[y & 1U];
        r->mti = 0;
    }
    y = r->mt[r->mti++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= (y >> 18);
    return y;
}

static void py_seed(PyRandom *r, uint64_t seed) {
    uint32_t key[2];
    int klen = 0;
    key[klen++] = (uint32_t)(seed & 0xffffffffU);
    if (seed >> 32) key[klen++] = (uint32_t)(seed >> 32);
    mt_init_by_array(r, key, klen);
    r->has_gauss = 0;
    r->gauss_next = 0.0;
}

static double py_random(PyRandom *r) {
    uint32_t a = mt_genrand(r) >> 5, b = mt_genrand(r) >> 6;
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

static double py_gauss(PyRandom *r, double mu, double sigma) {
    double z;
    if (r->has_gauss) {
        z = r->gauss_next;
        r->has_gauss = 0;
    } else {
        double x2pi = py_random(r) * (2.0 * M_PI);
        double g2rad = sqrt(-2.0 * log(1.0 - py_random(r)));
        z = cos(x2pi) * g2rad;
        r->gauss_next = sin(x2pi) * g2rad;
        r->has_gauss = 1;
    }
    return mu + z * sigma;
}

/* Python 浮点 % ：结果取除数符号 */
static double py_fmod(double a, double b) {
    double m = fmod(a, b);
    if (m != 0.0 && ((m < 0.0) != (b < 0.0))) m += b;
    return m;
}

/* ── 配置：全部由 Python 侧从 train_adas_cortex 读出传入，防止常量漂移 ── */
typedef struct {
    double wheelbase, dt, max_lateral_accel, stg_a_lat_max, stg_curve_safety;
    double cte_fail, max_speed, accel_max, brake_max;
    double steer_rate_max, steer_lag_tau, accel_lag_tau;
    double meas_noise_cte, meas_noise_psi, gust_period_s, gust_accel;
    double lat_env_cruise, lat_env_maneuver;
} EvalConfig;

/* 细胞类型编码：与 Python 侧 PRIM_CODE 字典一致 */
enum {
    P_SUM = 0, P_INTEGRATE, P_AMPLIFY, P_INVERT, P_THRESHOLD, P_DAMPER, P_CLIP, P_ABS,
    P_MULTIPLY, P_DIFF, P_HYSTERESIS, P_DEADZONE, P_INHIBIT, P_SUB, P_RATIO,
    P_OSCILLATOR, P_CORRELATION, P_FATIGUE, P_PASSTHROUGH
};

typedef struct {
    int n_cells, n_rec, n_syn;
    int steer_id, accel_id;
    const int32_t *types;        /* n_cells */
    const int32_t *syn_from;     /* n_syn，Python synapses 列表顺序 */
    const int32_t *syn_to;
} Topology;

enum { PATH_STRAIGHT = 0, PATH_SINE = 1, PATH_ARC = 2 };
enum { SPD_CRUISE = 0, SPD_CRUISE_FAST, SPD_STOP_GO, SPD_FOLLOW, SPD_RAMP };

typedef struct {
    int path_kind; double amp, wavelen, kappa;
    int spd_kind; double v0, duration; int lead_on;
} Scenario;

typedef struct {
    double cost, avg_cte, max_cte, avg_verr, avg_dsteer;
    int ok, steps, total;
} Metrics;

/* ── 细胞 ─────────────────────────────────────────────────── */
typedef struct { double state, aux, out, gain; } Cell;

static inline double dmax(double a, double b) { return a > b ? a : b; }
static inline double dmin(double a, double b) { return a < b ? a : b; }

static inline double cell_fire(Cell *c, int pt, double x) {
    double g = c->gain;
    switch (pt) {
    case P_SUM:       c->out = tanh(x * g); break;
    case P_INTEGRATE: c->state = c->state * 0.85 + x * 0.15; c->out = tanh(c->state * g); break;
    case P_AMPLIFY:   c->out = tanh(x * g * 2.5); break;
    case P_INVERT:    c->out = -tanh(x * g); break;
    case P_THRESHOLD: c->out = x > 0.25 ? 1.0 : (x < -0.25 ? -1.0 : 0.0); break;
    case P_DAMPER:    c->state = c->state * 0.70 + x * 0.30; c->out = c->state; break;
    case P_CLIP:      c->out = dmax(-1.0, dmin(1.0, x * g)); break;
    case P_ABS:       c->out = fabs(tanh(x * g)); break;
    case P_MULTIPLY:  c->out = tanh(x * g * 1.5); break;
    case P_DIFF:      c->out = x - c->state; c->state = x; break;
    case P_HYSTERESIS:
        if (x > 0.15) c->state = 1.0; else if (x < -0.15) c->state = -1.0;
        c->out = c->state; break;
    case P_DEADZONE:  c->out = fabs(x) > 0.08 ? x * g : 0.0; break;
    case P_INHIBIT:
        c->state = c->state * 0.80 + fabs(x) * 0.20;
        c->out = tanh(x * g) * dmax(0.0, 1.0 - c->state); break;
    case P_SUB:
        c->state = c->state * 0.60 + x * 0.40;
        c->out = tanh((x - c->state) * g); break;
    case P_RATIO:
        c->state = c->state * 0.85 + fabs(x) * 0.15;
        c->out = dmax(-2.0, dmin(2.0, x / (c->state + 0.1))); break;
    case P_OSCILLATOR: {
        double s1 = c->state, s2 = c->aux;
        double ds1 = s2;
        double ds2 = 1.0 * (1.0 - s1 * s1) * s2 - s1 + x;
        double dt = 0.05;
        s1 = dmax(-3.0, dmin(3.0, s1 + ds1 * dt));
        s2 = dmax(-3.0, dmin(3.0, s2 + ds2 * dt));
        c->state = s1; c->aux = s2; c->out = tanh(s1); break;
    }
    case P_CORRELATION:
        c->state = c->state * 0.90 + (x * c->aux) * 0.10;
        c->aux = x; c->out = tanh(c->state * g); break;
    case P_FATIGUE:
        c->state = dmin(2.0, c->state + fabs(x) * 0.15) * 0.96;
        c->out = tanh(x * g) / (1.0 + c->state); break;
    default:          c->out = x; break;
    }
    return c->out;
}

/* 器官：入边按 Python compile_incoming 的顺序（synapse 列表序） */
typedef struct {
    const Topology *topo;
    Cell *cells;
    int *inc_off;       /* n_cells+1 */
    int *inc_from;      /* n_syn */
    double *inc_w;      /* n_syn */
} Organ;

static void organ_build(Organ *o, const Topology *t, const double *gains, const double *weights,
                        Cell *cells, int *inc_off, int *inc_from, double *inc_w) {
    o->topo = t; o->cells = cells; o->inc_off = inc_off; o->inc_from = inc_from; o->inc_w = inc_w;
    for (int i = 0; i < t->n_cells; i++) { cells[i].gain = gains[i]; }
    int *cnt = inc_off; /* reuse as counts */
    memset(cnt, 0, sizeof(int) * (t->n_cells + 1));
    for (int s = 0; s < t->n_syn; s++) {
        int f = t->syn_from[s], to = t->syn_to[s];
        if (f >= 0 && f < t->n_cells && to >= t->n_rec && to < t->n_cells) cnt[to + 1]++;
    }
    for (int i = 0; i < t->n_cells; i++) cnt[i + 1] += cnt[i];
    int *fill = (int *)alloca(sizeof(int) * t->n_cells);
    for (int i = 0; i < t->n_cells; i++) fill[i] = inc_off[i];
    for (int s = 0; s < t->n_syn; s++) {
        int f = t->syn_from[s], to = t->syn_to[s];
        if (f >= 0 && f < t->n_cells && to >= t->n_rec && to < t->n_cells) {
            int k = fill[to]++;
            inc_from[k] = f; inc_w[k] = weights[s];
        }
    }
}

static void organ_reset(Organ *o) {
    for (int i = 0; i < o->topo->n_cells; i++) { o->cells[i].state = 0.0; o->cells[i].aux = 0.0; o->cells[i].out = 0.0; }
}

static void organ_forward(Organ *o, double cte_n, double dpsi_n, double kappa_n, double v_n,
                          double verr_n, double danger_n, double *steer, double *accel) {
    Cell *c = o->cells;
    const Topology *t = o->topo;
    c[0].out = dmax(0.0, -cte_n);
    c[1].out = dmax(0.0, cte_n);
    c[2].out = dmax(0.0, -cte_n * 2.0 - 0.5);
    c[3].out = dmax(0.0, cte_n * 2.0 - 0.5);
    c[4].out = dmax(-1.0, dmin(1.0, dpsi_n));
    c[5].out = dmax(-1.0, dmin(1.0, dpsi_n * 1.5));
    c[6].out = dmax(-1.0, dmin(1.0, kappa_n));
    c[7].out = dmax(-1.0, dmin(1.0, kappa_n * v_n));
    c[8].out = dmax(0.0, dmin(1.0, v_n));
    c[9].out = dmax(-1.0, dmin(1.0, verr_n));
    c[10].out = dmax(0.0, dmin(1.0, -verr_n));
    c[11].out = dmax(0.0, dmin(1.0, danger_n));
    for (int i = t->n_rec; i < t->n_cells; i++) {
        int a = o->inc_off[i], b = o->inc_off[i + 1];
        if (b > a) {
            /* Python 3.12 sum() 对 float 使用 Neumaier 补偿求和，这里逐字复刻 */
            double x = 0.0, comp = 0.0;
            for (int k = a; k < b; k++) {
                double v = c[o->inc_from[k]].out * o->inc_w[k];
                double tt = x + v;
                if (fabs(x) >= fabs(v)) comp += (x - tt) + v; else comp += (v - tt) + x;
                x = tt;
            }
            if (comp != 0.0 && isfinite(comp)) x += comp;
            cell_fire(&c[i], t->types[i], x);
        } else {
            c[i].out = c[i].state * 0.90;
        }
    }
    *steer = dmax(-1.0, dmin(1.0, c[t->steer_id].out));
    *accel = dmax(-1.0, dmin(1.0, c[t->accel_id].out));
}

/* ── 环境 ─────────────────────────────────────────────────── */
static void path_at(const Scenario *sc, double s, double *px, double *py, double *ph, double *pk) {
    if (sc->path_kind == PATH_STRAIGHT) { *px = s; *py = 0.0; *ph = 0.0; *pk = 0.0; return; }
    if (sc->path_kind == PATH_SINE) {
        double k = 2.0 * M_PI / sc->wavelen;
        double y = sc->amp * sin(k * s);
        double dy = sc->amp * k * cos(k * s);
        double ddy = -sc->amp * k * k * sin(k * s);
        *px = s; *py = y; *ph = atan(dy); *pk = ddy / pow(1.0 + dy * dy, 1.5);
        return;
    }
    double r = 1.0 / sc->kappa, th = s * sc->kappa;
    *px = r * sin(th); *py = r * (1.0 - cos(th)); *ph = th; *pk = sc->kappa;
}

static double speed_profile(int kind, double t, double duration) {
    switch (kind) {
    case SPD_CRUISE: return 14.0;
    case SPD_CRUISE_FAST: return 19.0;
    case SPD_STOP_GO:
        if (t < duration * 0.35) return 14.0;
        if (t < duration * 0.55) return 0.0;
        return 14.0;
    case SPD_FOLLOW: return 14.0 - 6.0 * dmax(0.0, sin(2.0 * M_PI * t / duration));
    case SPD_RAMP: return 6.0 + 13.0 * dmin(1.0, t / (duration * 0.6));
    default: return 14.0;
    }
}

static double steer_limit_for_speed(const EvalConfig *C, double v, double a_lat) {
    double s = dmax(v, 2.0);
    return dmin(dmax(atan(a_lat * C->wheelbase / (s * s)), 0.016), 0.16);
}

static double adaptive_steer_limit(const EvalConfig *C, double v, double lat_err) {
    double env = fabs(lat_err) > 0.5 ? C->lat_env_maneuver : C->lat_env_cruise;
    return steer_limit_for_speed(C, v, env);
}

static void run_scenario(const EvalConfig *C, Organ *o, const Scenario *sc, uint64_t seed, Metrics *M) {
    organ_reset(o);
    PyRandom rng; py_seed(&rng, seed);
    double x = 0.0, y = 0.6, heading = 0.0, v = sc->v0;
    double steer = 0.0, prev_steer = 0.0, accel_act = 0.0, v_ref = sc->v0;
    double lead_s = 45.0, lead_v = 11.0;
    double s_ref = 0.0;
    int n = (int)(sc->duration / C->dt);
    double cum_cte = 0.0, cum_verr = 0.0, cum_dsteer = 0.0, max_cte = 0.0;
    int steps = 0;

    for (int i = 0; i < n; i++) {
        double t = i * C->dt;
        double best_s = s_ref, best_d2 = 1e18, probe = s_ref;
        while (probe < s_ref + 30.0) {
            double px, py, ph, pk;
            path_at(sc, probe, &px, &py, &ph, &pk);
            double d2 = (px - x) * (px - x) + (py - y) * (py - y);
            if (d2 < best_d2) { best_d2 = d2; best_s = probe; }
            probe += 0.5;
        }
        s_ref = best_s;
        double px, py, ph, pk_unused, kap, dx, dy, dh;
        path_at(sc, s_ref, &px, &py, &ph, &pk_unused);
        path_at(sc, s_ref + dmax(v * 0.8, 2.0), &dx, &dy, &dh, &kap);

        double cte = cos(ph) * (py - y) - sin(ph) * (px - x);
        double dpsi = py_fmod(ph - heading + M_PI, 2.0 * M_PI) - M_PI;
        double v_target = speed_profile(sc->spd_kind, t, sc->duration);
        if (v_target > 0.5 && fabs(kap) > 1e-4) {
            v_target = dmin(v_target, C->stg_curve_safety * sqrt(C->stg_a_lat_max / fabs(kap)));
            v_target = dmin(v_target, 0.75 * sqrt(C->lat_env_maneuver / fabs(kap)));
        }
        {   /* achievable_ref */
            double dv = v_target - v_ref;
            double lim = (dv > 0 ? C->accel_max : C->brake_max) * C->dt;
            v_ref = v_ref + dmax(-lim, dmin(lim, dv));
        }
        double ttc;
        if (!sc->lead_on) ttc = 99.0;
        else {
            lead_v = 11.0 + 3.0 * sin(0.35 * t);
            lead_s += (lead_v - v) * C->dt;
            if (lead_s < 3.0) lead_s = 3.0;
            double rel_v = lead_v - v;
            ttc = rel_v < -0.1 ? (lead_s / -rel_v) : 99.0;
        }
        double danger = 1.0 - dmin(dmax(ttc, 0.0), 10.0) / 10.0;
        if (v_target < 0.5) danger = dmax(danger, 1.0);

        double cte_m = cte + py_gauss(&rng, 0.0, C->meas_noise_cte);
        double dpsi_m = dpsi + py_gauss(&rng, 0.0, C->meas_noise_psi);

        double steer_n, accel_n;
        organ_forward(o,
                      dmax(-1.0, dmin(1.0, cte_m / 2.0)),
                      dmax(-1.0, dmin(1.0, dpsi_m / 0.5)),
                      dmax(-1.0, dmin(1.0, kap * 20.0)),
                      dmax(0.0, dmin(1.0, v / C->max_speed)),
                      dmax(-1.0, dmin(1.0, (v_target - v) / 5.0)),
                      danger, &steer_n, &accel_n);

        double lim = adaptive_steer_limit(C, v, cte);
        double steer_req = dmax(-lim, dmin(lim, steer_n * lim));
        double d_max = C->steer_rate_max * C->dt;
        steer_req = steer + dmax(-d_max, dmin(d_max, steer_req - steer));
        steer += (steer_req - steer) * dmin(1.0, C->dt / C->steer_lag_tau);
        double steer_cmd = dmax(-lim, dmin(lim, steer));

        double accel_req = accel_n > 0 ? accel_n * C->accel_max : accel_n * 6.0;
        accel_act += (accel_req - accel_act) * dmin(1.0, C->dt / C->accel_lag_tau);
        double accel = accel_act;

        if (accel >= 0) { v += dmin(accel, C->accel_max) * C->dt; v = dmin(v, C->max_speed); }
        else            { v += dmax(accel, -C->brake_max) * C->dt; v = dmax(v, 0.0); }
        double yaw_rate = v / C->wheelbase * tan(steer_cmd);
        double half_wb = C->wheelbase * 0.5;
        x += (v * cos(heading) - half_wb * sin(heading) * yaw_rate) * C->dt;
        y += (v * sin(heading) + half_wb * cos(heading) * yaw_rate) * C->dt;
        heading += yaw_rate * C->dt;
        double gust = C->gust_accel * sin(2.0 * M_PI * t / C->gust_period_s);
        heading += (gust / dmax(v, 3.0)) * C->dt;

        double acte = fabs(cte);
        cum_cte += acte;
        max_cte = dmax(max_cte, acte);
        cum_verr += fabs(v_ref - v);
        cum_dsteer += fabs(steer_cmd - prev_steer);
        prev_steer = steer_cmd;
        steps += 1;
        if (acte > C->cte_fail) break;
    }
    int ok = (steps == n);
    int den = steps > 1 ? steps : 1;
    double avg_cte = cum_cte / den, avg_verr = cum_verr / den, avg_dsteer = cum_dsteer / den;
    double cost = avg_cte * 10.0 + avg_verr * 3.0 + avg_dsteer * 40.0;
    if (!ok) cost += 50.0 * (1.0 - (double)steps / n) + 20.0;
    M->cost = cost; M->ok = ok; M->avg_cte = avg_cte; M->max_cte = max_cte;
    M->avg_verr = avg_verr; M->avg_dsteer = avg_dsteer; M->steps = steps; M->total = n;
}

/* ── 导出 API ─────────────────────────────────────────────── */

/* 单场景：返回 cost，填 Metrics */
double adas_eval_scenario(const EvalConfig *C, const Topology *T, const double *gains,
                          const double *weights, const Scenario *sc, uint64_t seed, Metrics *M) {
    Cell *cells = (Cell *)calloc(T->n_cells, sizeof(Cell));
    int *inc_off = (int *)calloc(T->n_cells + 1, sizeof(int));
    int *inc_from = (int *)calloc(T->n_syn > 0 ? T->n_syn : 1, sizeof(int));
    double *inc_w = (double *)calloc(T->n_syn > 0 ? T->n_syn : 1, sizeof(double));
    Organ o; organ_build(&o, T, gains, weights, cells, inc_off, inc_from, inc_w);
    run_scenario(C, &o, sc, seed, M);
    free(cells); free(inc_off); free(inc_from); free(inc_w);
    return M->cost;
}

/* evaluate()：全部场景代价和 + 突触数惩罚 */
static double eval_one(const EvalConfig *C, const Topology *T, const double *gains, const double *weights,
                       const Scenario *scs, int n_scn, uint64_t seed, Metrics *Ms) {
    Cell *cells = (Cell *)calloc(T->n_cells, sizeof(Cell));
    int *inc_off = (int *)calloc(T->n_cells + 1, sizeof(int));
    int *inc_from = (int *)calloc(T->n_syn > 0 ? T->n_syn : 1, sizeof(int));
    double *inc_w = (double *)calloc(T->n_syn > 0 ? T->n_syn : 1, sizeof(double));
    Organ o; organ_build(&o, T, gains, weights, cells, inc_off, inc_from, inc_w);
    double total = 0.0;
    Metrics tmp;
    for (int k = 0; k < n_scn; k++) {
        Metrics *M = Ms ? &Ms[k] : &tmp;
        run_scenario(C, &o, &scs[k], seed, M);
        total += M->cost;
    }
    total += T->n_syn * 0.005;
    free(cells); free(inc_off); free(inc_from); free(inc_w);
    return total;
}

double adas_evaluate(const EvalConfig *C, const Topology *T, const double *gains, const double *weights,
                     const Scenario *scs, int n_scn, uint64_t seed, Metrics *Ms) {
    return eval_one(C, T, gains, weights, scs, n_scn, seed, Ms);
}

/* 批量：n_cand 个候选 × n_seeds 个噪声种子，OpenMP 并行，输出各候选的种子均值 */
void adas_evaluate_batch(const EvalConfig *C, const Topology *T, int n_cand,
                         const double *gains_all,     /* n_cand * n_cells */
                         const double *weights_all,   /* n_cand * n_syn   */
                         const Scenario *scs, int n_scn,
                         const uint64_t *seeds, int n_seeds, double *out_cost) {
    int n_jobs = n_cand * n_seeds;
    double *tmp = (double *)calloc(n_jobs, sizeof(double));
#pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < n_jobs; j++) {
        int c = j / n_seeds, s = j % n_seeds;
        tmp[j] = eval_one(C, T, gains_all + (size_t)c * T->n_cells,
                          weights_all + (size_t)c * T->n_syn, scs, n_scn, seeds[s], NULL);
    }
    for (int c = 0; c < n_cand; c++) {
        double acc = 0.0;
        for (int s = 0; s < n_seeds; s++) acc += tmp[c * n_seeds + s];
        out_cost[c] = acc / n_seeds;
    }
    free(tmp);
}

int adas_eval_num_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
