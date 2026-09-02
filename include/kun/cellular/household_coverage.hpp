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

        const auto predecessor = shortest_path_predecessors();
        const auto current = index(static_cast<std::size_t>(robot_.x),
                                   static_cast<std::size_t>(robot_.y));
        const auto dock = index(static_cast<std::size_t>(kDockX),
                                 static_cast<std::size_t>(kDockY));
        if (predecessor[dock] == no_predecessor() && current != dock) {
            return false;
        }

        auto next = dock;
        while (predecessor[next] != current) {
            next = predecessor[next];
        }
        const int next_x = static_cast<int>(next %
                                            static_cast<std::size_t>(width_));
        const int next_y = static_cast<int>(next /
                                            static_cast<std::size_t>(width_));
        robot_.heading = heading_toward(next_x - robot_.x, next_y - robot_.y);
        robot_.x = next_x;
        robot_.y = next_y;
        mark_cleaned(robot_.x, robot_.y);
        return true;
    }

    std::vector<std::size_t> shortest_path_predecessors() const {
        const auto no_predecessor_value = no_predecessor();
        std::vector<std::size_t> predecessor(cell_count_, no_predecessor_value);
        std::queue<std::size_t> pending;
        const auto start = index(static_cast<std::size_t>(robot_.x),
                                 static_cast<std::size_t>(robot_.y));
        const auto dock = index(static_cast<std::size_t>(kDockX),
                                static_cast<std::size_t>(kDockY));
        predecessor[start] = start;
        pending.push(start);

        while (!pending.empty()) {
            const auto current = pending.front();
            pending.pop();
            if (current == dock) {
                break;
            }
            const int current_x = static_cast<int>(
                current % static_cast<std::size_t>(width_));
            const int current_y = static_cast<int>(
                current / static_cast<std::size_t>(width_));
            for (const auto& [dx, dy] :
                 {std::pair<int, int>{0, -1}, {1, 0}, {0, 1}, {-1, 0}}) {
                const int next_x = current_x + dx;
                const int next_y = current_y + dy;
                if (!is_cleanable(next_x, next_y) ||
                    is_dynamic_obstacle(next_x, next_y)) {
                    continue;
                }
                const auto next = index(static_cast<std::size_t>(next_x),
                                        static_cast<std::size_t>(next_y));
                if (predecessor[next] != no_predecessor_value) {
                    continue;
                }
                predecessor[next] = current;
                pending.push(next);
            }
        }
        return predecessor;
    }

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

}  // namespace kun
