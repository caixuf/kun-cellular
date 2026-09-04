#pragma once

// ============================================================================
// DomainZoo — 12 个低维控制域任务动物园 (管线横向复刻批量实证)
//
// 目的: 证明 EvolvableTask 管线的通用性 —— 每个新域只需实现
//   reset_physics() / physics(force) / quality() 三个函数 (~40 行),
//   其余 (演化引擎 / 三隔离门禁 / 存盘 / 前端) 全部复用。
//
// 物理真实性声明: 每个任务都是真实的一阶/二阶动力学 Euler 积分,
// 死亡判定 = 真实物理包络越界 (不是人为惩罚项)。ood=2.0 时工厂参数
// 按各任务语义扰动 (质量×2 / 干扰×2 / 噪声注入), 门禁据此测跨物理泛化。
// ============================================================================

#include "kun/cellular/evolvable_task.hpp"
#include <cmath>
#include <random>
#include <vector>

namespace kun {

class ZooTask : public EvolvableTask {
public:
    explicit ZooTask(double ood = 1.0) : ood_(ood) {}
    const char* name() const override { return zoo_name(); }
    size_t obs_dim() const override { return 4; }
    size_t act_dim() const override { return 4; }
    void set_max_steps(int s) { max_steps_ = s; }
    int max_steps() const { return max_steps_; }

    void reset(uint32_t episode_seed) override {
        rng_.seed(episode_seed);
        steps_ = 0; q_sum_ = 0.0; t_ = 0.0;
        reset_physics();
    }

    std::vector<float> current_observation() const override {
        return {static_cast<float>(o_[0]), static_cast<float>(o_[1]),
                static_cast<float>(o_[2]), static_cast<float>(o_[3])};
    }

    StepResult step(int action) override {
        CellularOrganism::ActionOutputs acts;
        acts.positive_action = (action == 0 || action == 2) ? 1.0 : 0.0;
        acts.negative_action = (action == 1 || action == 2) ? 1.0 : 0.0;
        return step_continuous(acts);
    }

    StepResult step_continuous(const CellularOrganism::ActionOutputs& acts) override {
        StepResult res;
        double f = acts.positive_action - acts.negative_action;
        if (!std::isfinite(f)) f = 0.0;
        f = std::max(-1.0, std::min(1.0, f));

        const bool violated = physics(f);  // 推进一步动力学 + 包络判定
        ++steps_;
        if (!violated) q_sum_ += quality();

        res.obs = current_observation();
        res.steps = steps_;
        res.done = violated;
        res.success = !violated && steps_ >= max_steps_;
        res.reward = violated ? 0.0 : 1.0;
        res.min_dist_to_goal = 1.0 - quality();
        return res;
    }

    // 适应度 = 0.8 生存 + 0.2 跟踪质量 (生存是底线, 质量提供稠密梯度)
    double current_fitness() const override {
        const double surv = static_cast<double>(steps_) / static_cast<double>(max_steps_);
        const double qual = steps_ > 0 ? q_sum_ / static_cast<double>(steps_) : 0.0;
        return 0.8 * surv + 0.2 * qual;
    }

protected:
    virtual const char* zoo_name() const = 0;
    virtual void reset_physics() = 0;
    virtual bool physics(double f) = 0;   // 返回 true = 越包络 (死亡)
    virtual double quality() const = 0;   // ∈[0,1]

    double ood_{1.0};
    double o_[4]{0, 0, 0, 0};
    std::mt19937 rng_{1};
    std::normal_distribution<double> g_{0.0, 1.0};
    int steps_{0};
    int max_steps_{300};
    double q_sum_{0.0};
    double t_{0.0};
    static constexpr double DT = 0.02;
};

// 1. 倒立摆 (两轮平衡) — 不稳定对象
class ZooCartPole : public ZooTask {
public:
    explicit ZooCartPole(double ood = 1.0) : ZooTask(ood) {}
protected:
    const char* zoo_name() const override { return "zoo_cartpole"; }
    void reset_physics() override {
        x_ = 0.2 * (g_(rng_)); th_ = 0.12 * (g_(rng_));
        vx_ = 0; thd_ = 0;
    }
    bool physics(double f) override {
        const double m = 0.1 * ood_, L = 0.5 * (1.0 + 0.4 * (ood_ - 1.0));
        const double F = f * 10.0 + (ood_ - 1.0) * 2.0 * g_(rng_);
        const double tot = 1.0 + m, pl = m * L;
        const double tmp = (F + pl * thd_ * thd_ * std::sin(th_)) / tot;
        const double ta = (9.8 * std::sin(th_) - std::cos(th_) * tmp) /
                          (L * (4.0 / 3.0 - m * std::cos(th_) * std::cos(th_) / tot));
        const double xa = tmp - pl * ta * std::cos(th_) / tot;
        x_ += DT * vx_; vx_ += DT * xa; th_ += DT * thd_; thd_ += DT * ta;
        o_[0] = th_ / 0.35; o_[1] = thd_ / 3.0; o_[2] = x_ / 2.4; o_[3] = vx_ / 3.0;
        t_ += DT;
        return std::fabs(th_) > 0.2094 || std::fabs(x_) > 2.4 || !std::isfinite(th_);
    }
    double quality() const override { return 1.0 - std::min(1.0, std::fabs(th_) / 0.2094); }
private:
    double x_{0}, vx_{0}, th_{0}, thd_{0};
};

// 2. 滚球天平 — 光梁角度伺服, 双积分链
class ZooBallBeam : public ZooTask {
public:
    explicit ZooBallBeam(double ood = 1.0) : ZooTask(ood) {}
protected:
    const char* zoo_name() const override { return "zoo_ballbeam"; }
    void reset_physics() override { p_ = 0.4 * g_(rng_); v_ = 0; phi_ = 0; }
    bool physics(double f) override {
        phi_ += DT * f * 1.2;
        phi_ = std::max(-0.35, std::min(0.35, phi_));
        v_ += DT * (5.0 / 7.0) * 9.8 * std::sin(phi_) * ood_;
        p_ += DT * v_;
        o_[0] = p_ / 0.6; o_[1] = v_ / 1.5; o_[2] = phi_ / 0.35; o_[3] = f;
        t_ += DT;
        return std::fabs(p_) > 0.6 || !std::isfinite(p_);
    }
    double quality() const override { return 1.0 - std::min(1.0, std::fabs(p_) / 0.6); }
private:
    double p_{0}, v_{0}, phi_{0};
};

// 3. 磁悬浮 — 上方吸引式, 开环不稳定 (间隙越小吸力越大→发散)
class ZooMaglev : public ZooTask {
public:
    explicit ZooMaglev(double ood = 1.0) : ZooTask(ood) {}
protected:
    const char* zoo_name() const override { return "zoo_maglev"; }
    void reset_physics() override {
        m_ = 0.05 * ood_;
        h_ = 0.02 + 0.004 * g_(rng_); hd_ = 0;
    }
    bool physics(double f) override {
        const double i = (f + 1.0) * 2.0;                       // 0..4 A
        // K 标定: i=2A 时在目标高度 h=0.02 处 F = mg (悬浮平衡存在)
        // K = mg*(D-h_t)²/i² = 0.49*0.064/4 = 7.84e-4
        const double K = 7.84e-4, D = 0.1;
        const double F = K * i * i / ((D - h_) * (D - h_));     // 上方吸引, 开环不稳定
        hd_ += DT * (F / m_ - 9.8);
        h_ += DT * hd_;
        o_[0] = (h_ - 0.02) / 0.03; o_[1] = hd_ / 0.5;
        o_[2] = (i - 2.0) / 2.0; o_[3] = 0;
        t_ += DT;
        return h_ < 0.005 || h_ > 0.08 || !std::isfinite(h_);
    }
    double quality() const override { return 1.0 - std::min(1.0, std::fabs(h_ - 0.02) / 0.03); }
private:
    double h_{0.02}, hd_{0}, m_{0.05};
};

// 4. 火箭悬停 — 非线性推重比, 目标高度带保持
class ZooRocketHover : public ZooTask {
public:
    explicit ZooRocketHover(double ood = 1.0) : ZooTask(ood) {}
protected:
    const char* zoo_name() const override { return "zoo_rocket_hover"; }
    void reset_physics() override {
        m_ = 1.0 * ood_;
        y_ = 100.0 + 8.0 * g_(rng_); vy_ = 1.0 * g_(rng_);
    }
    bool physics(double f) override {
        const double u = (f + 1.0) * 0.5;                       // 油门 0..1
        const double a = u * (20.0 / m_) - 9.8;                 // 最大推重比 2g
        vy_ += DT * a; y_ += DT * vy_;
        o_[0] = (y_ - 100.0) / 25.0; o_[1] = vy_ / 10.0; o_[2] = u * 2.0 - 1.0; o_[3] = 0;
        t_ += DT;
        return vy_ < -10.0 || std::fabs(y_ - 100.0) > 25.0 || !std::isfinite(y_);
    }
    double quality() const override { return 1.0 - std::min(1.0, std::fabs(y_ - 100.0) / 25.0); }
private:
    double y_{100}, vy_{0}, m_{1.0};
};

// 5. 巡航控制 — 坡度扰动下的速度跟踪
class ZooCruise : public ZooTask {
public:
    explicit ZooCruise(double ood = 1.0) : ZooTask(ood) {}
protected:
    const char* zoo_name() const override { return "zoo_cruise"; }
    void reset_physics() override { v_ = 30.0 + 3.0 * g_(rng_); }
    bool physics(double f) override {
        const double slope = (3.0 * std::sin(0.15 * t_) + 1.5 * std::sin(0.53 * t_)) * ood_;
        const double a = f * 3.0 - 0.0015 * (1.0 + 0.5 * (ood_ - 1.0)) * v_ * v_
                         - 9.8 * slope / 100.0;
        v_ += DT * a;
        o_[0] = (v_ - 30.0) / 12.0; o_[1] = slope / 6.0; o_[2] = v_ / 42.0; o_[3] = 0;
        t_ += DT;
        return v_ > 42.0 || v_ < 8.0 || !std::isfinite(v_);
    }
    double quality() const override { return 1.0 - std::min(1.0, std::fabs(v_ - 30.0) / 12.0); }
private:
    double v_{30};
};

// 6. 温度调节 — 大热惯性 + 执行器滞后 + 环境波动
class ZooThermal : public ZooTask {
public:
    explicit ZooThermal(double ood = 1.0) : ZooTask(ood) {}
protected:
    const char* zoo_name() const override { return "zoo_thermal"; }
    void reset_physics() override {
        T_ = 22.0 + 4.0 * g_(rng_); ah_ = 0;
        amb0_ = -5.0 - 10.0 * (ood_ - 1.0);
    }
    bool physics(double f) override {
        ah_ += (f - ah_) * 0.1;                                 // 加热器一阶滞后
        const double amb = amb0_ + 8.0 * std::sin(2.0 * M_PI * t_ / 12.0);
        T_ += DT * (ah_ * 6.0 + (amb - T_) * 0.04 * ood_);
        o_[0] = (T_ - 22.0) / 8.0; o_[1] = (amb - T_) / 20.0; o_[2] = ah_; o_[3] = 0;
        t_ += DT;
        return std::fabs(T_ - 22.0) > 8.0 || !std::isfinite(T_);
    }
    double quality() const override { return 1.0 - std::min(1.0, std::fabs(T_ - 22.0) / 8.0); }
private:
    double T_{22}, ah_{0}, amb0_{-5};
};

// 7. 水箱液位 — 积分对象 + 非线性出流
class ZooWaterTank : public ZooTask {
public:
    explicit ZooWaterTank(double ood = 1.0) : ZooTask(ood) {}
protected:
    const char* zoo_name() const override { return "zoo_water_tank"; }
    void reset_physics() override { L_ = 0.5 + 0.15 * g_(rng_); q_ = 0.01; dL_ = 0; }
    bool physics(double f) override {
        q_ = (f + 1.0) * 0.5 * 0.02;                            // 阀门 0..0.02 m³/s
        const double out = 0.02828 * ood_ * std::sqrt(std::max(0.0, L_));
        const double dl = (q_ - out) / 2.0;
        L_ += DT * dl; dL_ = dl;
        o_[0] = (L_ - 0.5) / 0.45; o_[1] = q_ / 0.02 - 1.0;
        o_[2] = std::max(-3.0, std::min(3.0, dl * 50.0)); o_[3] = 0;
        t_ += DT;
        return L_ > 0.95 || L_ < 0.05 || !std::isfinite(L_);
    }
    double quality() const override { return 1.0 - std::min(1.0, std::fabs(L_ - 0.5) / 0.45); }
private:
    double L_{0.5}, q_{0.01}, dL_{0};
};

// 8. 直流电机调速 — 负载阶跃扰动
class ZooDCMotor : public ZooTask {
public:
    explicit ZooDCMotor(double ood = 1.0) : ZooTask(ood) {}
protected:
    const char* zoo_name() const override { return "zoo_dc_motor"; }
    void reset_physics() override {
        w_ = 50.0 + 8.0 * g_(rng_); tl_ = 0;
        J_ = 0.02 * ood_;
    }
    bool physics(double f) override {
        if (static_cast<int>(steps_ / 60) % 2 == 0) tl_ = 3.0 * ood_; else tl_ = -3.0 * ood_;
        const double dw = (f * 25.0 - 1.0 * (w_ - 50.0) / 50.0 * 10.0 - tl_) / J_ * 0.02;
        w_ += DT * dw * 0.02 * 50.0;                            // 折算: 保持量纲稳定
        o_[0] = (w_ - 50.0) / 25.0; o_[1] = tl_ / 3.0; o_[2] = f; o_[3] = 0;
        t_ += DT;
        return std::fabs(w_ - 50.0) > 25.0 || !std::isfinite(w_);
    }
    double quality() const override { return 1.0 - std::min(1.0, std::fabs(w_ - 50.0) / 25.0); }
private:
    double w_{50}, tl_{0}, J_{0.02};
};

// 9. 振动隔离 — 共振对象 + 多频地面激励
class ZooVibration : public ZooTask {
public:
    explicit ZooVibration(double ood = 1.0) : ZooTask(ood) {}
protected:
    const char* zoo_name() const override { return "zoo_vibration"; }
    void reset_physics() override { xr_ = 0; vd_ = 0; }
    bool physics(double f) override {
        const double A = 0.05 * ood_, fr = 1.0 * (1.0 + 0.5 * (ood_ - 1.0));
        const double yg = A * (std::sin(2 * M_PI * fr * t_) + 0.3 * std::sin(2 * M_PI * 2.3 * fr * t_)
                        + 0.2 * g_(rng_));
        const double a = -40.0 * xr_ - 1.0 * vd_ + f * 8.0 - 40.0 * yg;
        vd_ += DT * a; xr_ += DT * vd_;
        o_[0] = xr_ / 0.15; o_[1] = vd_ / 1.0; o_[2] = yg / A; o_[3] = f;
        t_ += DT;
        return std::fabs(xr_) > 0.15 || !std::isfinite(xr_);
    }
    double quality() const override { return 1.0 - std::min(1.0, std::fabs(xr_) / 0.15); }
private:
    double xr_{0}, vd_{0};
};

// 10. 伺服定位 — 双积分对象 + 载荷变化
class ZooServo : public ZooTask {
public:
    explicit ZooServo(double ood = 1.0) : ZooTask(ood) {}
protected:
    const char* zoo_name() const override { return "zoo_servo"; }
    void reset_physics() override {
        x_ = 1.8 * (g_(rng_) > 0 ? 1 : -1) * (0.6 + 0.3 * std::fabs(g_(rng_)));
        x_ = std::max(-1.9, std::min(1.9, x_)); v_ = 0;
    }
    bool physics(double f) override {
        const double a = f * 4.0 / ood_;                        // 载荷使推力失效
        v_ += DT * a; x_ += DT * v_;
        o_[0] = x_ / 2.2; o_[1] = v_ / 5.0; o_[2] = 0; o_[3] = 0;
        t_ += DT;
        return std::fabs(x_) > 2.2 || std::fabs(v_) > 5.0 || !std::isfinite(x_);
    }
    double quality() const override {
        return 1.0 - std::min(1.0, (std::fabs(x_) + 0.2 * std::fabs(v_)) / 2.2);
    }
private:
    double x_{0}, v_{0};
};

// 11. 锅炉压力 — 执行器滞后 + 蒸汽需求波动 (需求测拒绝)
class ZooBoiler : public ZooTask {
public:
    explicit ZooBoiler(double ood = 1.0) : ZooTask(ood) {}
protected:
    const char* zoo_name() const override { return "zoo_boiler"; }
    void reset_physics() override { P_ = 5.0 + 1.5 * g_(rng_); ah_ = 0; }
    bool physics(double f) override {
        ah_ += (f - ah_) * (0.08 / (1.0 + (ood_ - 1.0)));
        const double demand = (0.5 + 0.3 * std::sin(2 * M_PI * t_ / 10.0) + 0.05 * g_(rng_))
                              * (1.0 + 0.5 * (ood_ - 1.0));
        P_ += DT * (ah_ - demand) * 0.5;
        o_[0] = (P_ - 5.0) / 4.0; o_[1] = demand; o_[2] = ah_; o_[3] = 0;
        t_ += DT;
        return P_ < 1.0 || P_ > 9.0 || !std::isfinite(P_);
    }
    double quality() const override { return 1.0 - std::min(1.0, std::fabs(P_ - 5.0) / 4.0); }
private:
    double P_{5}, ah_{0};
};

// 12. 自行车低铲平衡 — 低速时转向权限不足 (OOD 更难)
class ZooBicycle : public ZooTask {
public:
    explicit ZooBicycle(double ood = 1.0) : ZooTask(ood) {}
protected:
    const char* zoo_name() const override { return "zoo_bicycle"; }
    void reset_physics() override {
        th_ = 0.06 * g_(rng_); thd_ = 0.2 * g_(rng_);
        v_ = 3.0 * (1.0 - 0.4 * (ood_ - 1.0));                  // 低速更难
    }
    bool physics(double f) override {
        const double delta = f * 0.4;
        const double a = 9.8 * std::sin(th_) - (v_ * v_ / 1.5) * std::cos(th_) * delta / 1.0;
        thd_ += DT * a; th_ += DT * thd_;
        o_[0] = th_ / 0.26; o_[1] = thd_ / 2.0; o_[2] = delta / 0.4; o_[3] = v_ / 3.0;
        t_ += DT;
        return std::fabs(th_) > 0.26 || !std::isfinite(th_);
    }
    double quality() const override { return 1.0 - std::min(1.0, std::fabs(th_) / 0.26); }
private:
    double th_{0}, thd_{0}, v_{3};
};

} // namespace kun
