// 生物圈生态专业页原生运行时
//   - 复用底座 C 模型 EcoBiosphere (真实多生境生态圈: 代谢/捕食/气候/多样性)
//   - prey/predator 的位移为显示层合成 (底座 EcoAgent 坐标静态)，theta 由显示层速度导出
//   - 真实信号: 生态位种群、捕食事件(PREDATION)、香农多样性
#include "kun/cellular/ecosystem_biosphere.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <type_traits>
#include <unordered_map>
#include <vector>

using namespace kun;

#ifdef __cplusplus
extern "C" {
#endif

#define ECO_MAX_FOOD 80
#define ECO_MAX_PREY 80
#define ECO_MAX_PRED 40
#define ECO_HIST 64

typedef struct { float x, y; } EcoFood;
typedef struct { float x, y, theta; int32_t alive; } EcoPrey;
typedef struct { float x, y, theta; } EcoPred;

typedef struct {
    int32_t real;
    int32_t generation;
    int32_t step_count;
    int32_t max_steps;
    int32_t prey_alive;
    int32_t total_prey;
    int32_t total_hunts;
    int32_t history_len;
    int32_t history_prey[ECO_HIST];
    int32_t history_pred[ECO_HIST];
    int32_t n_food;
    EcoFood food[ECO_MAX_FOOD];
    int32_t n_prey;
    EcoPrey prey[ECO_MAX_PREY];
    int32_t n_pred;
    EcoPred predators[ECO_MAX_PRED];
} EcoTelemetry;

static_assert(std::is_trivially_copyable<EcoTelemetry>::value,
              "EcoTelemetry must be POD for ctypes mirroring");

struct EcoView {
    uint64_t id = 0;
    float x = 0.0f, y = 0.0f;
    float dx = 0.0f, dy = 0.0f;
    float theta = 0.0f;
    bool seen = false;
};

static std::mutex g_mutex;
static std::unique_ptr<EcoBiosphere> g_bio;
static std::mt19937 g_rng(20260911);
static bool g_loaded = false;
static int32_t g_generation = 0;
static int32_t g_step_count = 0;
static int32_t g_max_steps = 360;
static int32_t g_total_hunts = 0;
static int32_t g_total_prey0 = 36;
static int32_t g_history_prey[ECO_HIST] = {0};
static int32_t g_history_pred[ECO_HIST] = {0};
static int32_t g_history_len = 0;
static std::unordered_map<uint64_t, EcoView> g_views;

static constexpr float kCanvasX0 = 70.0f, kCanvasX1 = 730.0f;
static constexpr size_t kInitPerNiche = 12; // 底座生态平衡点偏低(草食≈3)，控制捕食者规模
static constexpr float kCanvasY0 = 60.0f, kCanvasY1 = 540.0f;

static void map_base(double bx, double by, float& mx, float& my) {
    float tx = (static_cast<float>(bx) + 50.0f) / 100.0f;  // 底座 x∈[-50,50]
    float ty = (static_cast<float>(by) + 50.0f) / 100.0f;
    tx = std::min(1.0f, std::max(0.0f, tx));
    ty = std::min(1.0f, std::max(0.0f, ty));
    mx = kCanvasX0 + tx * (kCanvasX1 - kCanvasX0);
    my = kCanvasY0 + ty * (kCanvasY1 - kCanvasY0);
}

static void reset_world_locked(uint32_t seed) {
    g_bio = std::make_unique<EcoBiosphere>(kInitPerNiche, seed ? seed : static_cast<uint32_t>(g_rng()));
    g_views.clear();
    g_step_count = 0;
    // 采样初始草食种群作为 total_prey
    auto counts = g_bio->get_niche_population();
    auto it = counts.find(SpeciesNiche::HERBIVORE);
    g_total_prey0 = (it != counts.end()) ? static_cast<int32_t>(it->second) : 36;
    if (g_total_prey0 <= 0) g_total_prey0 = 36;
}

int32_t eco_c_init(const char* /*ckpt: 底座模型自播种，忽略*/) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_generation = 0;
    g_total_hunts = 0;
    g_history_len = 0;
    std::memset(g_history_prey, 0, sizeof(g_history_prey));
    std::memset(g_history_pred, 0, sizeof(g_history_pred));
    reset_world_locked(20260911u);
    g_loaded = (g_bio != nullptr);
    return g_loaded ? 1 : 0;
}

void eco_c_reset(uint32_t seed) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_loaded) return;
    reset_world_locked(seed);
}

int32_t eco_c_step() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_loaded || !g_bio) return 0;

    // 中性调用: 不注入 market 参数
    g_bio->step_ecosystem(1.0, 0.15, 0.0, 0.0);
    g_step_count++;

    // 真实捕食事件计数
    for (const auto& tr : g_bio->recent_transfers()) {
        if (tr.transfer_type == "PREDATION") g_total_hunts++;
    }
    return 0; // 周期切换在取遥测时判定，保证一定步数
}

void eco_c_get_telemetry(EcoTelemetry* out) {
    if (!out) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    std::memset(out, 0, sizeof(EcoTelemetry));
    out->real = g_loaded ? 1 : 0;
    out->max_steps = g_max_steps;

    if (!g_bio) return;

    const auto& agents = g_bio->agents();

    // 收集存活 prey/predator 的显示坐标 (用于显示层追逐)
    std::vector<std::pair<float, float>> prey_pos;
    std::vector<std::pair<float, float>> pred_pos;
    for (const auto& a : agents) {
        if (!a.is_alive) continue;
        auto& v = g_views[a.id];
        if (!v.seen) {
            v.seen = true;
            map_base(a.x, a.y, v.x, v.y);
            std::uniform_real_distribution<float> dv(-1.5f, 1.5f);
            v.dx = dv(g_rng);
            v.dy = dv(g_rng);
        }
        if (a.niche == SpeciesNiche::HERBIVORE) prey_pos.push_back({v.x, v.y});
        else if (a.niche == SpeciesNiche::PREDATOR) pred_pos.push_back({v.x, v.y});
    }

    // 显示层运动: 猎物躲避最近捕食者, 捕食者追最近猎物
    const float speed = 3.2f;
    for (auto& a : agents) {
        if (!a.is_alive) continue;
        auto& v = g_views[a.id];
        float tx = v.x, ty = v.y;
        bool flee = false;
        if (a.niche == SpeciesNiche::HERBIVORE && !pred_pos.empty()) {
            float best = 1e18f;
            for (auto& p : pred_pos) {
                float d = (p.first - v.x) * (p.first - v.x) + (p.second - v.y) * (p.second - v.y);
                if (d < best) { best = d; tx = p.first; ty = p.second; }
            }
            flee = true;
        } else if (a.niche == SpeciesNiche::PREDATOR && !prey_pos.empty()) {
            float best = 1e18f;
            for (auto& p : prey_pos) {
                float d = (p.first - v.x) * (p.first - v.x) + (p.second - v.y) * (p.second - v.y);
                if (d < best) { best = d; tx = p.first; ty = p.second; }
            }
        } else if (a.niche == SpeciesNiche::PRODUCER) {
            continue; // 生产者静止
        } else {
            continue;
        }
        float ddx = tx - v.x, ddy = ty - v.y;
        float len = std::sqrt(ddx * ddx + ddy * ddy) + 1e-5f;
        float dir = flee ? -1.0f : 1.0f;
        v.dx = 0.85f * v.dx + 0.15f * (dir * ddx / len) * speed;
        v.dy = 0.85f * v.dy + 0.15f * (dir * ddy / len) * speed;
        v.x = std::min(kCanvasX1, std::max(kCanvasX0, v.x + v.dx));
        v.y = std::min(kCanvasY1, std::max(kCanvasY0, v.y + v.dy));
        v.theta = std::atan2(v.dy, v.dx);
    }

    int32_t n_food = 0, n_prey = 0, n_pred = 0, prey_alive = 0;
    for (const auto& a : agents) {
        if (!a.is_alive) continue;
        auto it = g_views.find(a.id);
        if (it == g_views.end()) continue;
        const EcoView& v = it->second;
        if (a.niche == SpeciesNiche::PRODUCER && n_food < ECO_MAX_FOOD) {
            out->food[n_food].x = v.x;
            out->food[n_food].y = v.y;
            n_food++;
        } else if (a.niche == SpeciesNiche::HERBIVORE) {
            prey_alive++;
            if (n_prey < ECO_MAX_PREY) {
                out->prey[n_prey].x = v.x;
                out->prey[n_prey].y = v.y;
                out->prey[n_prey].theta = v.theta;
                out->prey[n_prey].alive = 1;
                n_prey++;
            }
        } else if (a.niche == SpeciesNiche::PREDATOR && n_pred < ECO_MAX_PRED) {
            out->predators[n_pred].x = v.x;
            out->predators[n_pred].y = v.y;
            out->predators[n_pred].theta = v.theta;
            n_pred++;
        }
    }

    out->n_food = n_food;
    out->n_prey = n_prey;
    out->n_pred = n_pred;
    out->prey_alive = prey_alive;
    out->total_prey = g_total_prey0;
    out->total_hunts = g_total_hunts;
    out->step_count = g_step_count;
    out->generation = g_generation;

    // 周期切换 (在遥测阶段判定, 保证步数可见)
    if (g_step_count >= g_max_steps) {
        if (g_history_len < ECO_HIST) {
            g_history_prey[g_history_len] = prey_alive;
            g_history_pred[g_history_len] = n_pred;
            g_history_len++;
        } else {
            std::memmove(g_history_prey, g_history_prey + 1, sizeof(int32_t) * (ECO_HIST - 1));
            std::memmove(g_history_pred, g_history_pred + 1, sizeof(int32_t) * (ECO_HIST - 1));
            g_history_prey[ECO_HIST - 1] = prey_alive;
            g_history_pred[ECO_HIST - 1] = n_pred;
        }
        g_generation++;
        g_step_count = 0;
        g_bio = std::make_unique<EcoBiosphere>(kInitPerNiche, static_cast<uint32_t>(g_rng()));
        g_views.clear();
    }

    out->history_len = g_history_len;
    std::memcpy(out->history_prey, g_history_prey, sizeof(g_history_prey));
    std::memcpy(out->history_pred, g_history_pred, sizeof(g_history_pred));
}

#ifdef __cplusplus
}
#endif
