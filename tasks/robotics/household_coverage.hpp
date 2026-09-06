#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <queue>
#include <sstream>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>
#include "kun/cellular/evolvable_task.hpp"

namespace kun {

/**
 * @brief Deterministic grid-based household coverage simulation environment.
 *
 * This models a static room and furniture layout for simulation benchmarks; it
 * does not represent or make claims about physical robot navigation.
 */
class HouseholdCoverageEnvironment {
public:
    enum class Action {
        FORWARD,
        TURN_LEFT,
        TURN_RIGHT,
        RETURN_TO_DOCK,
        WAIT,
    };

    enum class Heading {
        NORTH,
        EAST,
        SOUTH,
        WEST,
    };

    struct Robot {
        int x{1};
        int y{1};
        Heading heading{Heading::EAST};
    };

    HouseholdCoverageEnvironment(int width, int height, uint32_t seed)
        : width_(width), height_(height),
          cell_count_(checked_cell_count(width, height)) {
        reset(seed);
    }

    void reset(uint32_t seed) {
        seed_ = seed;
        cleanable_.assign(cell_count_, false);
        cleaned_.assign(cell_count_, false);
        cleaned_count_ = 0;
        battery_ = 1.0;
        collision_count_ = 0;
        interrupted_ = false;
        dynamic_obstacle_.active = false;
        homing_trail_.clear();

        const auto width = static_cast<std::size_t>(width_);
        const auto height = static_cast<std::size_t>(height_);
        for (std::size_t y = 1; y + 1 < height; ++y) {
            for (std::size_t x = 1; x + 1 < width; ++x) {
                cleanable_[index(x, y)] = true;
            }
        }

        place_furniture(seed_);
        cleanable_[index(static_cast<std::size_t>(kDockX),
                         static_cast<std::size_t>(kDockY))] = true;
        retain_dock_reachable_cells();
        robot_ = {kDockX, kDockY};
    }

    /**
     * @brief Execute one deterministic action.
     *
     * Every call consumes 0.01 battery ratio, including a blocked or
     * interrupted action. FORWARD and RETURN_TO_DOCK return true when they
     * move (or RETURN_TO_DOCK is already at the dock); TURN and WAIT return
     * true. A blocked movement returns false. Interruptions freeze movement
     * without counting as collisions.
     */
    bool step(Action action) {
        consume_action_energy();

        switch (action) {
        case Action::FORWARD:
            return step_forward();
        case Action::TURN_LEFT:
            robot_.heading = rotate_left(robot_.heading);
            return true;
        case Action::TURN_RIGHT:
            robot_.heading = rotate_right(robot_.heading);
            return true;
        case Action::RETURN_TO_DOCK:
            return return_to_dock();
        case Action::WAIT:
            return true;
        }
        return false;
    }

    /**
     * @brief Place one deterministic temporary blocker in the forward cell.
     *
     * The request is ignored unless the current forward cell is in bounds and
     * cleanable. The blocker remains until reset or replaced by another valid
     * injection.
     */
    void inject_dynamic_obstacle_ahead() {
        const auto [dx, dy] = heading_delta(robot_.heading);
        const int obstacle_x = robot_.x + dx;
        const int obstacle_y = robot_.y + dy;
        if (!is_cleanable(obstacle_x, obstacle_y)) {
            return;
        }
        dynamic_obstacle_ = {true, obstacle_x, obstacle_y};
    }

    void interrupt_cleaning() { interrupted_ = true; }

    void resume_cleaning() { interrupted_ = false; }

    bool is_interrupted() const { return interrupted_; }

    bool is_cleanable(int x, int y) const {
        return in_bounds(x, y) &&
               cleanable_[index(static_cast<std::size_t>(x),
                                static_cast<std::size_t>(y))];
    }

    bool mark_cleaned(int x, int y) {
        if (!is_cleanable(x, y)) {
            return false;
        }
        const auto cell_index = index(static_cast<std::size_t>(x),
                                      static_cast<std::size_t>(y));
        if (cleaned_[cell_index]) {
            return false;
        }
        cleaned_[cell_index] = true;
        ++cleaned_count_;
        return true;
    }

    double coverage_ratio() const {
        if (cleanable_count_ == 0) {
            return 0.0;
        }
        return static_cast<double>(cleaned_count_) /
               static_cast<double>(cleanable_count_);
    }

    double battery_ratio() const {
        return battery_;
    }

    int width() const { return width_; }

    int height() const { return height_; }

    bool has_dock() const { return true; }

    int dock_x() const { return kDockX; }

    int dock_y() const { return kDockY; }

    const Robot& robot() const {
        return robot_;
    }

    Heading heading() const { return robot_.heading; }

    int robot_x() const { return robot_.x; }

    int robot_y() const { return robot_.y; }

    bool is_cleaned(int x, int y) const {
        if (!is_cleanable(x, y)) {
            return false;
        }
        return cleaned_[index(static_cast<std::size_t>(x),
                              static_cast<std::size_t>(y))];
    }

    std::size_t cleaned_cells() const { return cleaned_count_; }

    std::size_t cleanable_cells() const { return cleanable_count_; }

    std::size_t collision_count() const { return collision_count_; }

    bool at_dock() const {
        return robot_.x == kDockX && robot_.y == kDockY;
    }

private:
    struct DynamicObstacle {
        bool active{false};
        int x{};
        int y{};
    };

    static constexpr int kMinimumDimension = 8;
    static constexpr int kDockX = 1;
    static constexpr int kDockY = 1;
    static constexpr std::size_t kMaximumCellCount = 16'777'216;
    static constexpr double kActionEnergy = 0.01;

    static std::size_t checked_cell_count(int width, int height) {
        if (width < kMinimumDimension || height < kMinimumDimension) {
            throw std::invalid_argument(
                "household coverage dimensions must be at least 8");
        }

        const auto width_size = static_cast<std::size_t>(width);
        const auto height_size = static_cast<std::size_t>(height);
        if (width_size > std::numeric_limits<std::size_t>::max() / height_size) {
            throw std::invalid_argument(
                "household coverage dimensions are too large");
        }

        const auto cell_count = width_size * height_size;
        if (cell_count > kMaximumCellCount ||
            cell_count > std::vector<bool>().max_size()) {
            throw std::invalid_argument(
                "household coverage dimensions are too large");
        }
        return cell_count;
    }

    bool in_bounds(int x, int y) const {
        return x >= 0 && y >= 0 &&
               static_cast<std::size_t>(x) < static_cast<std::size_t>(width_) &&
               static_cast<std::size_t>(y) < static_cast<std::size_t>(height_);
    }

    std::size_t index(std::size_t x, std::size_t y) const {
        return y * static_cast<std::size_t>(width_) + x;
    }

    void consume_action_energy() {
        battery_ = std::max(0.0, battery_ - kActionEnergy);
    }

    bool step_forward() {
        if (interrupted_) {
            return false;
        }

        const auto [dx, dy] = heading_delta(robot_.heading);
        const int destination_x = robot_.x + dx;
        const int destination_y = robot_.y + dy;
        if (!is_cleanable(destination_x, destination_y) ||
            is_dynamic_obstacle(destination_x, destination_y)) {
            ++collision_count_;
            return false;
        }

        robot_.x = destination_x;
        robot_.y = destination_y;
        mark_cleaned(robot_.x, robot_.y);
        return true;
    }

    bool return_to_dock() {
        if (at_dock()) {
            return true;
        }
        if (interrupted_) {
            return false;
        }

        // Insect-inspired path integration & reactive homing:
        // Homing vector dx, dy combined with local working-memory trail to skirt obstacles,
        // eliminating global BFS graph search completely.
        const int rx = robot_.x;
        const int ry = robot_.y;

        const std::pair<int, int> moves[4] = {{0, -1}, {1, 0}, {0, 1}, {-1, 0}};
        int best_x = rx, best_y = ry;
        double best_cost = 1e9;

        for (const auto& [dx, dy] : moves) {
            int nx = rx + dx;
            int ny = ry + dy;
            if (!in_bounds(nx, ny) || !is_cleanable(nx, ny) || is_dynamic_obstacle(nx, ny)) {
                continue;
            }

            int dist = std::abs(nx - kDockX) + std::abs(ny - kDockY);
            int trail_hits = 0;
            const size_t trail_sz = homing_trail_.size();
            const size_t lookback = std::min(trail_sz, static_cast<size_t>(16));
            for (size_t i = trail_sz - lookback; i < trail_sz; ++i) {
                if (homing_trail_[i].first == nx && homing_trail_[i].second == ny) {
                    ++trail_hits;
                }
            }

            int to_dock_x = kDockX - rx;
            int to_dock_y = kDockY - ry;
            bool aligned = (dx * to_dock_x > 0) || (dy * to_dock_y > 0);

            double cost = dist * 10.0 + trail_hits * 35.0 - (aligned ? 2.0 : 0.0);
            if (cost < best_cost) {
                best_cost = cost;
                best_x = nx;
                best_y = ny;
            }
        }

        if (best_x == rx && best_y == ry) {
            return false;
        }

        robot_.heading = heading_toward(best_x - robot_.x, best_y - robot_.y);
        robot_.x = best_x;
        robot_.y = best_y;
        homing_trail_.push_back({best_x, best_y});
        if (homing_trail_.size() > 64) {
            homing_trail_.erase(homing_trail_.begin());
        }
        mark_cleaned(robot_.x, robot_.y);
        return true;
    }

public:
    bool is_dynamic_obstacle(int x, int y) const {
        return dynamic_obstacle_.active && dynamic_obstacle_.x == x &&
               dynamic_obstacle_.y == y;
    }

    static std::pair<int, int> heading_delta(Heading heading) {
        switch (heading) {
        case Heading::NORTH:
            return {0, -1};
        case Heading::EAST:
            return {1, 0};
        case Heading::SOUTH:
            return {0, 1};
        case Heading::WEST:
            return {-1, 0};
        }
        return {0, 0};
    }

    static Heading heading_toward(int dx, int dy) {
        if (dx > 0) {
            return Heading::EAST;
        }
        if (dx < 0) {
            return Heading::WEST;
        }
        if (dy > 0) {
            return Heading::SOUTH;
        }
        return Heading::NORTH;
    }

    static Heading rotate_left(Heading heading) {
        switch (heading) {
        case Heading::NORTH:
            return Heading::WEST;
        case Heading::WEST:
            return Heading::SOUTH;
        case Heading::SOUTH:
            return Heading::EAST;
        case Heading::EAST:
            return Heading::NORTH;
        }
        return Heading::NORTH;
    }

    static Heading rotate_right(Heading heading) {
        switch (heading) {
        case Heading::NORTH:
            return Heading::EAST;
        case Heading::EAST:
            return Heading::SOUTH;
        case Heading::SOUTH:
            return Heading::WEST;
        case Heading::WEST:
            return Heading::NORTH;
        }
        return Heading::NORTH;
    }

private:
    static constexpr std::size_t no_predecessor() {
        return std::numeric_limits<std::size_t>::max();
    }

    void retain_dock_reachable_cells() {
        std::vector<bool> visited(cell_count_, false);
        std::queue<std::size_t> pending;
        const auto dock = index(static_cast<std::size_t>(kDockX),
                                static_cast<std::size_t>(kDockY));
        visited[dock] = true;
        pending.push(dock);

        while (!pending.empty()) {
            const auto current = pending.front();
            pending.pop();
            const int current_x = static_cast<int>(
                current % static_cast<std::size_t>(width_));
            const int current_y = static_cast<int>(
                current / static_cast<std::size_t>(width_));
            for (const auto& [dx, dy] :
                 {std::pair<int, int>{0, -1}, {1, 0}, {0, 1}, {-1, 0}}) {
                const int next_x = current_x + dx;
                const int next_y = current_y + dy;
                if (!in_bounds(next_x, next_y) ||
                    !cleanable_[index(static_cast<std::size_t>(next_x),
                                      static_cast<std::size_t>(next_y))]) {
                    continue;
                }
                const auto next = index(static_cast<std::size_t>(next_x),
                                        static_cast<std::size_t>(next_y));
                if (visited[next]) {
                    continue;
                }
                visited[next] = true;
                pending.push(next);
            }
        }

        cleanable_count_ = 0;
        for (std::size_t cell = 0; cell < cell_count_; ++cell) {
            cleanable_[cell] = visited[cell];
            if (cleanable_[cell]) {
                ++cleanable_count_;
            }
        }
    }

    void place_furniture(uint32_t seed) {
        std::mt19937 generator(seed);
        std::uniform_int_distribution<int> x_distribution(2, width_ - 3);
        std::uniform_int_distribution<int> y_distribution(2, height_ - 3);
        std::uniform_int_distribution<int> width_distribution(1, 2);
        std::uniform_int_distribution<int> height_distribution(1, 2);
        const auto furniture_count = std::max<std::size_t>(1, cell_count_ / 96);

        for (std::size_t furniture = 0; furniture < furniture_count; ++furniture) {
            const int left = x_distribution(generator);
            const int top = y_distribution(generator);
            const int furniture_width = width_distribution(generator);
            const int furniture_height = height_distribution(generator);

            const auto right = std::min(
                static_cast<std::size_t>(left) +
                    static_cast<std::size_t>(furniture_width),
                static_cast<std::size_t>(width_ - 1));
            const auto bottom = std::min(
                static_cast<std::size_t>(top) +
                    static_cast<std::size_t>(furniture_height),
                static_cast<std::size_t>(height_ - 1));
            for (std::size_t y = static_cast<std::size_t>(top); y < bottom; ++y) {
                for (std::size_t x = static_cast<std::size_t>(left); x < right; ++x) {
                    cleanable_[index(x, y)] = false;
                }
            }
        }
    }

    int width_;
    int height_;
    std::size_t cell_count_{};
    uint32_t seed_{};
    Robot robot_{};
    std::vector<bool> cleanable_;
    std::vector<bool> cleaned_;
    std::size_t cleanable_count_{};
    std::size_t cleaned_count_{};
    double battery_{1.0};
    std::size_t collision_count_{};
    bool interrupted_{false};
    DynamicObstacle dynamic_obstacle_{};
    std::vector<std::pair<int, int>> homing_trail_;
};

struct HouseholdCoverageReport {
    static constexpr bool kSimulationOnly = true;

    double coverage_ratio{};
    std::size_t cleaned_cells{};
    std::size_t cleanable_cells{};
    int collisions{};
    double energy_used{};
    double final_battery_ratio{};
    bool returned_to_dock{};

    std::string to_json() const {
        std::ostringstream json;
        json.precision(std::numeric_limits<double>::max_digits10);
        json << "{\"coverage_ratio\":" << coverage_ratio
             << ",\"cleaned_cells\":" << cleaned_cells
             << ",\"cleanable_cells\":" << cleanable_cells
             << ",\"collisions\":" << collisions
             << ",\"energy_used\":" << energy_used
             << ",\"final_battery_ratio\":" << final_battery_ratio
             << ",\"returned_to_dock\":"
             << (returned_to_dock ? "true" : "false")
             << ",\"simulation_only\":" << std::boolalpha
             << kSimulationOnly << "}";
        return json.str();
    }
};

class HouseholdCoverageEvaluator {
public:
    static HouseholdCoverageReport run_baseline(int width, int height,
                                                uint32_t seed, int max_steps) {
        if (max_steps <= 0) {
            throw std::invalid_argument(
                "household coverage max_steps must be positive");
        }

        HouseholdCoverageEnvironment env(width, height, seed);
        int steps = 0;
        while (steps < max_steps && !all_cleanable_cells_visited(env)) {
            const auto path = path_to_nearest_unvisited(env);
            if (path.size() < 2) {
                break;
            }

            const auto return_distance = shortest_path_length(
                env, path.back(), dock_cell(env));
            if (return_distance == std::numeric_limits<std::size_t>::max()) {
                break;
            }
            const auto action_count =
                path_action_count(env, path) + return_distance;
            if (action_count > static_cast<std::size_t>(max_steps - steps)) {
                break;
            }

            if (!execute_path(env, path, max_steps, steps)) {
                break;
            }
        }

        while (steps < max_steps && !env.at_dock()) {
            const auto before_x = env.robot_x();
            const auto before_y = env.robot_y();
            const bool progressed =
                env.step(HouseholdCoverageEnvironment::Action::RETURN_TO_DOCK);
            ++steps;
            if (!progressed || (before_x == env.robot_x() &&
                                before_y == env.robot_y())) {
                break;
            }
        }

        HouseholdCoverageReport report;
        report.coverage_ratio = env.coverage_ratio();
        report.cleaned_cells = env.cleaned_cells();
        report.cleanable_cells = env.cleanable_cells();
        report.collisions = static_cast<int>(env.collision_count());
        report.energy_used = 1.0 - env.battery_ratio();
        report.final_battery_ratio = env.battery_ratio();
        report.returned_to_dock = env.at_dock();
        return report;
    }

private:
    using Environment = HouseholdCoverageEnvironment;
    using Cell = std::pair<int, int>;

    static Cell dock_cell(const Environment& env) {
        return {env.dock_x(), env.dock_y()};
    }

    static std::size_t cell_index(const Environment& env, const Cell& cell) {
        return static_cast<std::size_t>(cell.second) *
                   static_cast<std::size_t>(env.width()) +
               static_cast<std::size_t>(cell.first);
    }

    static std::vector<Cell> path_to_nearest_unvisited(const Environment& env) {
        const auto start = Cell{env.robot_x(), env.robot_y()};
        const auto total_cells = static_cast<std::size_t>(env.width()) *
                                 static_cast<std::size_t>(env.height());
        const auto no_predecessor = std::numeric_limits<std::size_t>::max();
        std::vector<std::size_t> predecessor(total_cells, no_predecessor);
        std::queue<Cell> pending;
        predecessor[cell_index(env, start)] = cell_index(env, start);
        pending.push(start);

        const Cell target = [&]() {
            while (!pending.empty()) {
                const auto current = pending.front();
                pending.pop();
                if (current != start && env.is_cleanable(current.first, current.second) &&
                    !env.is_cleaned(current.first, current.second)) {
                    return current;
                }
                for (const auto& [dx, dy] :
                     {Cell{0, -1}, Cell{1, 0}, Cell{0, 1}, Cell{-1, 0}}) {
                    const Cell next{current.first + dx, current.second + dy};
                    if (next.first < 0 || next.second < 0 ||
                        next.first >= env.width() || next.second >= env.height() ||
                        !env.is_cleanable(next.first, next.second)) {
                        continue;
                    }
                    const auto next_index = cell_index(env, next);
                    if (predecessor[next_index] != no_predecessor) {
                        continue;
                    }
                    predecessor[next_index] = cell_index(env, current);
                    pending.push(next);
                }
            }
            return Cell{-1, -1};
        }();

        if (target.first < 0) {
            return {};
        }

        std::vector<Cell> path;
        auto current = target;
        while (current != start) {
            path.push_back(current);
            const auto predecessor_index = predecessor[cell_index(env, current)];
            current = {static_cast<int>(
                           predecessor_index %
                           static_cast<std::size_t>(env.width())),
                       static_cast<int>(
                           predecessor_index /
                           static_cast<std::size_t>(env.width()))};
        }
        path.push_back(start);
        std::reverse(path.begin(), path.end());
        return path;
    }

    static std::size_t shortest_path_length(const Environment& env,
                                            const Cell& start,
                                            const Cell& target) {
        const auto total_cells = static_cast<std::size_t>(env.width()) *
                                 static_cast<std::size_t>(env.height());
        const auto no_distance = std::numeric_limits<std::size_t>::max();
        std::vector<std::size_t> distance(total_cells, no_distance);
        std::queue<Cell> pending;
        distance[cell_index(env, start)] = 0;
        pending.push(start);

        while (!pending.empty()) {
            const auto current = pending.front();
            pending.pop();
            if (current == target) {
                return distance[cell_index(env, current)];
            }
            for (const auto& [dx, dy] :
                 {Cell{0, -1}, Cell{1, 0}, Cell{0, 1}, Cell{-1, 0}}) {
                const Cell next{current.first + dx, current.second + dy};
                if (next.first < 0 || next.second < 0 ||
                    next.first >= env.width() || next.second >= env.height() ||
                    !env.is_cleanable(next.first, next.second)) {
                    continue;
                }
                const auto next_index = cell_index(env, next);
                if (distance[next_index] != no_distance) {
                    continue;
                }
                distance[next_index] =
                    distance[cell_index(env, current)] + 1;
                pending.push(next);
            }
        }
        return no_distance;
    }

    static std::size_t path_action_count(const Environment& env,
                                         const std::vector<Cell>& path) {
        auto heading = env.heading();
        std::size_t actions = 0;
        for (std::size_t i = 1; i < path.size(); ++i) {
            const auto desired = heading_toward(path[i].first - path[i - 1].first,
                                                path[i].second - path[i - 1].second);
            actions += turn_count(heading, desired) + 1;
            heading = desired;
        }
        return actions;
    }

    static bool execute_path(Environment& env, const std::vector<Cell>& path,
                             int max_steps, int& steps) {
        for (std::size_t i = 1; i < path.size(); ++i) {
            const auto desired = heading_toward(path[i].first - path[i - 1].first,
                                                path[i].second - path[i - 1].second);
            while (env.heading() != desired) {
                if (steps >= max_steps) {
                    return false;
                }
                const auto turns_right =
                    (heading_number(desired) - heading_number(env.heading()) + 4) %
                    4;
                const auto action =
                    turns_right <= 2
                        ? Environment::Action::TURN_RIGHT
                        : Environment::Action::TURN_LEFT;
                env.step(action);
                ++steps;
            }
            if (steps >= max_steps ||
                !env.step(Environment::Action::FORWARD)) {
                if (steps < max_steps) {
                    ++steps;
                }
                return false;
            }
            ++steps;
        }
        return true;
    }

    static bool all_cleanable_cells_visited(const Environment& env) {
        return env.cleaned_cells() == env.cleanable_cells();
    }

    static int heading_number(Environment::Heading heading) {
        switch (heading) {
        case Environment::Heading::NORTH:
            return 0;
        case Environment::Heading::EAST:
            return 1;
        case Environment::Heading::SOUTH:
            return 2;
        case Environment::Heading::WEST:
            return 3;
        }
        return 0;
    }

    static std::size_t turn_count(Environment::Heading from,
                                  Environment::Heading to) {
        const auto right_turns =
            (heading_number(to) - heading_number(from) + 4) % 4;
        return static_cast<std::size_t>(std::min(right_turns, 4 - right_turns));
    }

    static Environment::Heading heading_toward(int dx, int dy) {
        if (dx > 0) {
            return Environment::Heading::EAST;
        }
        if (dx < 0) {
            return Environment::Heading::WEST;
        }
        if (dy > 0) {
            return Environment::Heading::SOUTH;
        }
        return Environment::Heading::NORTH;
    }
};

/**
 * @brief 具身机器人室内全域覆盖与动态避障任务 (HouseholdCoverageTask)
 * 遵循 EvolvableTask 标准 Gym 契约:
 * - 8 维局部物理感知受体: 前/左/右连续测距, 前/左/右局部未清扫污渍探针, 剩余电量, 充电桩相对方位
 * - 4 维动作效应器: 前进, 左转, 右转, 防御性回充 (RETURN_TO_DOCK)
 */
class HouseholdCoverageTask : public EvolvableTask {
public:
    explicit HouseholdCoverageTask(int width = 24, int height = 16, uint32_t seed = 41, int max_steps = 1500)
        : width_(width), height_(height), max_steps_(max_steps), env_(width, height, seed) {
        reset(seed);
    }

    const char* name() const override { return "HouseholdCoverage"; }
    size_t obs_dim() const override { return 4; }
    size_t act_dim() const override { return 4; }

    void reset(uint32_t episode_seed) override {
        env_.reset(episode_seed);
        step_count_ = 0;
        last_cleaned_count_ = env_.cleaned_cells();
        dock_return_attempted_ = false;
        dock_return_successful_ = false;
        stuck_counter_ = 0;
        last_x_ = env_.robot_x();
        last_y_ = env_.robot_y();
    }

    std::vector<float> current_observation() const override {
        using Env = HouseholdCoverageEnvironment;
        const int rx = env_.robot_x();
        const int ry = env_.robot_y();
        const auto heading = env_.heading();

        auto [fdx, fdy] = Env::heading_delta(heading);
        auto left_h = Env::rotate_left(heading);
        auto [ldx, ldy] = Env::heading_delta(left_h);
        auto right_h = Env::rotate_right(heading);
        auto [rdx, rdy] = Env::heading_delta(right_h);

        // 1. 前状态: 1.0 = 未扫污渍, 0.4 = 已扫净空, 0.0 = 阻挡/障碍
        const bool f_ok = env_.is_cleanable(rx + fdx, ry + fdy) && !env_.is_dynamic_obstacle(rx + fdx, ry + fdy);
        const bool f_dirt = f_ok && !env_.is_cleaned(rx + fdx, ry + fdy);
        const float front_status = f_ok ? (f_dirt ? 1.0f : 0.4f) : 0.0f;

        // 2. 左状态
        const bool l_ok = env_.is_cleanable(rx + ldx, ry + ldy) && !env_.is_dynamic_obstacle(rx + ldx, ry + ldy);
        const bool l_dirt = l_ok && !env_.is_cleaned(rx + ldx, ry + ldy);
        const float left_status = l_ok ? (l_dirt ? 1.0f : 0.4f) : 0.0f;

        // 3. 右状态
        const bool r_ok = env_.is_cleanable(rx + rdx, ry + rdy) && !env_.is_dynamic_obstacle(rx + rdx, ry + rdy);
        const bool r_dirt = r_ok && !env_.is_cleaned(rx + rdx, ry + rdy);
        const float right_status = r_ok ? (r_dirt ? 1.0f : 0.4f) : 0.0f;

        // 4. 回充使命相态: 1.0 = 步数末期需安全回桩, 0.0 = 正常清扫
        const float mission_phase = (step_count_ >= max_steps_ - 150) ? 1.0f : 0.0f;

        return { front_status, left_status, right_status, mission_phase };
    }

    StepResult step(int action) override {
        CellularOrganism::ActionOutputs acts;
        if (action == 0) acts.positive_action = 1.0;
        else if (action == 1) acts.negative_action = -1.0;
        else if (action == 2) acts.negative_action = 1.0;
        else if (action == 3) acts.defensive_reset = 1.0;
        return step_continuous(acts);
    }

    StepResult step_continuous(const CellularOrganism::ActionOutputs& acts) override {
        using Env = HouseholdCoverageEnvironment;
        ++step_count_;

        Env::Action action = Env::Action::WAIT;
        if (acts.defensive_reset > 0.5) {
            action = Env::Action::RETURN_TO_DOCK;
            dock_return_attempted_ = true;
        } else if (acts.immune_lock) {
            action = Env::Action::WAIT;
        } else {
            // Pure autonomous cellular steering:
            // Forward thrust (positive_action) vs differential steering (negative_action: >0 right, <0 left)
            if (acts.positive_action > std::abs(acts.negative_action) && acts.positive_action > 0.05) {
                action = Env::Action::FORWARD;
            } else if (acts.negative_action > 0.0) {
                action = Env::Action::TURN_RIGHT;
            } else if (acts.negative_action < 0.0) {
                action = Env::Action::TURN_LEFT;
            } else {
                action = Env::Action::FORWARD;
            }
        }

        const size_t prev_cleaned = env_.cleaned_cells();
        const size_t prev_coll = env_.collision_count();
        env_.step(action);
        const size_t new_cleaned = env_.cleaned_cells();
        const size_t new_coll = env_.collision_count();

        if (env_.robot_x() == last_x_ && env_.robot_y() == last_y_) {
            ++stuck_counter_;
        } else {
            stuck_counter_ = 0;
            last_x_ = env_.robot_x();
            last_y_ = env_.robot_y();
        }

        double step_reward = 0.0;
        if (new_cleaned > prev_cleaned) {
            step_reward += 10.0 * (new_cleaned - prev_cleaned);
        }
        if (new_coll > prev_coll) {
            step_reward -= 2.0;
        }
        if (env_.at_dock() && dock_return_attempted_) {
            dock_return_successful_ = true;
            step_reward += 50.0;
        }

        const bool all_cleaned = (env_.cleaned_cells() == env_.cleanable_cells());
        const bool done = (step_count_ >= max_steps_ || (all_cleaned && env_.at_dock()));

        StepResult res;
        res.obs = current_observation();
        res.reward = step_reward;
        res.done = done;
        res.success = (env_.coverage_ratio() >= 0.70 && env_.at_dock());
        res.steps = step_count_;
        res.collision_count = static_cast<int>(env_.collision_count());
        return res;
    }

    double current_fitness() const override {
        double fit = env_.coverage_ratio() * 100.0;
        if (env_.at_dock() && env_.coverage_ratio() > 0.40) {
            fit += 30.0;
        }
        fit -= static_cast<double>(env_.collision_count()) * 1.0;
        return fit;
    }

    const HouseholdCoverageEnvironment& env() const { return env_; }
    HouseholdCoverageEnvironment& env() { return env_; }

private:
    int width_{24};
    int height_{16};
    int max_steps_{1500};
    HouseholdCoverageEnvironment env_;
    int step_count_{0};
    size_t last_cleaned_count_{0};
    bool dock_return_attempted_{false};
    bool dock_return_successful_{false};
    int stuck_counter_{0};
    int last_x_{1};
    int last_y_{1};
};

}  // namespace kun
