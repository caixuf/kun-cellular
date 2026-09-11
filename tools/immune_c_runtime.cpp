// 免疫猎杀专业页原生运行时
//   - 复用底座 C 模型 PathogenCoEvolutionWorld (真实宿主-病原体协同演化)
//   - 病原体/巨噬细胞粒子为宿主感染/免疫遥测的"显示层合成"(底座宿主坐标为静态)
//   - 真实信号: 感染数(clearance)、抗体记忆、病原体毒株演化
#include "kun/cellular/digital_pathogen_ecosystem.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <type_traits>

using namespace kun;

#ifdef __cplusplus
extern "C" {
#endif

#define IMMUNE_MAX_PATHOGENS 64
#define IMMUNE_MAX_MACROPHAGES 16
#define IMMUNE_HIST 64

typedef struct {
    int32_t id;
    float x, y;
    int32_t alive;
    int32_t type; // 0..2 -> 前端配色
} ImmunePathogen;

typedef struct {
    int32_t id;
    float x, y;
    float radius;
    float chem_r;
} ImmuneMacrophage;

typedef struct {
    int32_t real;
    int32_t generation;
    int32_t step_count;
    int32_t max_steps;
    float clearance_rate; // 0..100
    int32_t pathogens_alive;
    int32_t total_pathogens;
    int32_t history_len;
    float history_clearance[IMMUNE_HIST];
    int32_t n_pathogens;
    ImmunePathogen pathogens[IMMUNE_MAX_PATHOGENS];
    int32_t n_macrophages;
    ImmuneMacrophage macrophages[IMMUNE_MAX_MACROPHAGES];
} ImmuneTelemetry;

static_assert(std::is_trivially_copyable<ImmuneTelemetry>::value,
              "ImmuneTelemetry must be POD for ctypes mirroring");

// ---- 显示层巨噬细胞 (追猎者) ----
struct MacView {
    float x, y;
    float radius, chem_r;
};

static std::mutex g_mutex;
static std::unique_ptr<PathogenCoEvolutionWorld> g_world;
static std::mt19937 g_rng(20260911);
static bool g_loaded = false;
static int32_t g_generation = 0;
static int32_t g_step_count = 0;
static int32_t g_max_steps = 300;
static float g_history[IMMUNE_HIST] = {0.0f};
static int32_t g_history_len = 0;
static MacView g_macs[IMMUNE_MAX_MACROPHAGES];

static constexpr float kCanvasX0 = 80.0f, kCanvasX1 = 760.0f;
static constexpr float kCanvasY0 = 80.0f, kCanvasY1 = 560.0f;

static void map_position(double wx, double wy, float& mx, float& my) {
    // 底座宿主坐标 x∈[0,18], y∈[0,4] -> 画布
    float tx = static_cast<float>(wx) / 18.0f;
    float ty = (wy > 0.0) ? static_cast<float>(wy) / 4.0f : 0.0f;
    tx = std::min(1.0f, std::max(0.0f, tx));
    ty = std::min(1.0f, std::max(0.0f, ty));
    mx = kCanvasX0 + tx * (kCanvasX1 - kCanvasX0);
    my = kCanvasY0 + ty * (kCanvasY1 - kCanvasY0);
}

static void init_macrophages_locked() {
    std::uniform_real_distribution<float> dx(kCanvasX0, kCanvasX1);
    std::uniform_real_distribution<float> dy(kCanvasY0, kCanvasY1);
    for (int i = 0; i < IMMUNE_MAX_MACROPHAGES; ++i) {
        g_macs[i].x = dx(g_rng);
        g_macs[i].y = dy(g_rng);
        g_macs[i].radius = 7.0f;
        g_macs[i].chem_r = 60.0f;
    }
}

static void start_world_locked(uint32_t seed) {
    if (seed == 0) seed = g_rng();
    g_world = std::make_unique<PathogenCoEvolutionWorld>(24, 3, seed);
    g_world->release_pathogen_outbreak(0xA1B2 + (seed % 7), 2.8, 0.12);
    g_step_count = 0;
    init_macrophages_locked();
}

int32_t immune_c_init(const char* /*ckpt_path: 底座模型自播种，忽略*/) {
    std::lock_guard<std::mutex> lock(g_mutex);
    start_world_locked(20260911u);
    g_generation = 0;
    g_history_len = 0;
    std::memset(g_history, 0, sizeof(g_history));
    g_loaded = (g_world != nullptr);
    return g_loaded ? 1 : 0;
}

void immune_c_reset(uint32_t seed) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_loaded) return;
    start_world_locked(seed);
}

// 返回 1 表示本步结束了一个疫情周期 (已自动开启新周期)
int32_t immune_c_step() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_loaded || !g_world) return 0;

    PathogenCoEvolutionWorld::PathogenTelemetry t = g_world->tick(60.0, 0.05);
    g_step_count++;

    const size_t infected = t.infected_hosts;
    const size_t total = std::max<size_t>(1, t.total_hosts);
    const float clearance =
        100.0f * (1.0f - static_cast<float>(infected) / static_cast<float>(total));

    // 巨噬细胞追猎: 追最近感染宿主 (无目标则缓慢漂移)
    const auto& hosts = g_world->get_hosts();
    for (int m = 0; m < IMMUNE_MAX_MACROPHAGES; ++m) {
        float tx = g_macs[m].x + 1.5f, ty = g_macs[m].y; // 默认漂移
        float best_d = 1e18f;
        for (const auto& h : hosts) {
            if (!h) continue;
            const auto& imm = h->get_immune();
            if (!imm.is_infected || !h->get_homeostasis().is_alive) continue;
            float hx, hy;
            map_position(h->get_x(), h->get_y(), hx, hy);
            const float d = (hx - g_macs[m].x) * (hx - g_macs[m].x) +
                            (hy - g_macs[m].y) * (hy - g_macs[m].y);
            if (d < best_d) { best_d = d; tx = hx; ty = hy; }
        }
        g_macs[m].x += 0.18f * (tx - g_macs[m].x);
        g_macs[m].y += 0.18f * (ty - g_macs[m].y);
    }

    if (g_step_count >= g_max_steps) {
        if (g_history_len < IMMUNE_HIST) {
            g_history[g_history_len++] = clearance;
        } else {
            std::memmove(g_history, g_history + 1, sizeof(float) * (IMMUNE_HIST - 1));
            g_history[IMMUNE_HIST - 1] = clearance;
        }
        g_generation++;
        start_world_locked(0);
        return 1;
    }
    // 疫情被清零时注入新一轮爆发，保持周期完整
    if (infected == 0) {
        g_world->release_pathogen_outbreak(0xA1B2 + static_cast<uint32_t>(g_step_count % 11),
                                           2.8, 0.12);
    }
    return 0;
}

void immune_c_get_telemetry(ImmuneTelemetry* out) {
    if (!out) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    std::memset(out, 0, sizeof(ImmuneTelemetry));
    out->real = g_loaded ? 1 : 0;
    out->generation = g_generation;
    out->step_count = g_step_count;
    out->max_steps = g_max_steps;
    out->history_len = g_history_len;
    std::memcpy(out->history_clearance, g_history, sizeof(g_history));

    if (!g_world) return;

    const auto& hosts = g_world->get_hosts();
    size_t infected = 0;
    for (const auto& h : hosts) {
        if (h && h->get_immune().is_infected && h->get_homeostasis().is_alive) infected++;
    }
    const size_t total = std::max<size_t>(1, hosts.size());
    out->pathogens_alive = static_cast<int32_t>(infected);
    out->total_pathogens = static_cast<int32_t>(total);
    out->clearance_rate =
        100.0f * (1.0f - static_cast<float>(infected) / static_cast<float>(total));

    // 病原体粒子: 每个感染宿主按载量合成 1~3 个带轨道漂移的粒子
    int32_t npath = 0;
    for (size_t i = 0; i < hosts.size() && npath < IMMUNE_MAX_PATHOGENS; ++i) {
        const auto& h = hosts[i];
        if (!h) continue;
        const auto& imm = h->get_immune();
        if (!imm.is_infected || !h->get_homeostasis().is_alive) continue;
        float hx, hy;
        map_position(h->get_x(), h->get_y(), hx, hy);
        const int count = 1 + static_cast<int>(std::min(2.0, imm.infection_load / 3.0));
        for (int k = 0; k < count && npath < IMMUNE_MAX_PATHOGENS; ++k) {
            const float ph = static_cast<float>(g_step_count) * 0.15f +
                             static_cast<float>(i) + static_cast<float>(k) * 2.1f;
            out->pathogens[npath].id = static_cast<int32_t>(i);
            out->pathogens[npath].x = hx + 8.0f * std::cos(ph);
            out->pathogens[npath].y = hy + 8.0f * std::sin(ph);
            out->pathogens[npath].alive = 1;
            out->pathogens[npath].type =
                static_cast<int32_t>(imm.current_infection_strain_id % 3);
            npath++;
        }
    }
    out->n_pathogens = npath;

    // 巨噬细胞粒子
    for (int m = 0; m < IMMUNE_MAX_MACROPHAGES; ++m) {
        out->macrophages[m].id = m;
        out->macrophages[m].x = g_macs[m].x;
        out->macrophages[m].y = g_macs[m].y;
        out->macrophages[m].radius = g_macs[m].radius;
        out->macrophages[m].chem_r = g_macs[m].chem_r;
    }
    out->n_macrophages = IMMUNE_MAX_MACROPHAGES;
}

#ifdef __cplusplus
}
#endif
