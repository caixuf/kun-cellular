#pragma once
// ============================================================================
// population/avalanche_metric.hpp — Prefer-A minimal avalanche / SOC metric
//
// Mechanism #5: criticality via avalanche size histogram + branching ratio.
// Lyapunov loop-gain (CellularOrganism::check_lyapunov_stability) is a COMPANION
// stability signal — it does NOT satisfy the avalanche requirement.
//
// Minimal Prefer-A definition:
//   Cascade over a 1-D activity buffer: sites above threshold "topple", distribute
//   excess to neighbors (Abelian sandpile-like). Record cascade size (sites toppled
//   per drive event). Branching ratio ≈ mean children-per-topple.
// ============================================================================
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace kun {
namespace population {

inline constexpr std::size_t kAvalancheMaxSites = 128;
inline constexpr std::size_t kAvalancheHistBins = 32;

struct AvalancheReport {
    std::size_t events{0};
    std::size_t total_topples{0};
    double mean_size{0.0};
    double max_size{0.0};
    double branching_ratio{0.0};   // mean offspring activations / topple
    double size_entropy{0.0};      // histogram Shannon entropy (nats)
    std::array<std::uint32_t, kAvalancheHistBins> size_histogram{};
    bool valid{false};
};

// Prefer-A avalanche analyzer over a fixed-capacity site buffer.
class AvalancheMetric {
public:
    explicit AvalancheMetric(std::size_t n_sites = 82, float threshold = 1.0f)
        : n_sites_(std::clamp(n_sites, std::size_t{1}, kAvalancheMaxSites)),
          threshold_(threshold) {
        heights_.assign(n_sites_, 0.0f);
        hist_.fill(0);
    }

    std::size_t n_sites() const { return n_sites_; }
    float threshold() const { return threshold_; }

    void reset() {
        std::fill(heights_.begin(), heights_.end(), 0.0f);
        hist_.fill(0);
        events_ = 0;
        total_topples_ = 0;
        max_size_ = 0;
        branching_sum_ = 0.0;
        branching_count_ = 0;
    }

    // Drive one site by +grain, then relax. Returns cascade size (topples).
    std::size_t drive(std::size_t site, float grain = 1.0f) {
        if (site >= n_sites_) site = site % n_sites_;
        heights_[site] += grain;
        return relax();
    }

    // Batch: seed heights from activity vector (Prefer-A organism activations), then relax once.
    std::size_t ingest_activity(const float* activity, std::size_t n, float scale = 1.0f) {
        const std::size_t m = std::min(n, n_sites_);
        for (std::size_t i = 0; i < m; ++i) {
            heights_[i] += std::abs(activity[i]) * scale;
        }
        return relax();
    }

    AvalancheReport report() const {
        AvalancheReport r;
        r.events = events_;
        r.total_topples = total_topples_;
        r.mean_size = (events_ > 0)
            ? (static_cast<double>(total_topples_) / static_cast<double>(events_))
            : 0.0;
        r.max_size = static_cast<double>(max_size_);
        r.branching_ratio = (branching_count_ > 0)
            ? (branching_sum_ / static_cast<double>(branching_count_))
            : 0.0;
        r.size_histogram = hist_;
        // Shannon entropy of size histogram
        double total = 0.0;
        for (auto h : hist_) total += static_cast<double>(h);
        double H = 0.0;
        if (total > 0.0) {
            for (auto h : hist_) {
                if (h == 0) continue;
                const double p = static_cast<double>(h) / total;
                H -= p * std::log(p);
            }
        }
        r.size_entropy = H;
        r.valid = (events_ > 0);
        return r;
    }

    // Lyapunov is companion — documented helper, not a substitute.
    static constexpr const char* lyapunov_companion_note() {
        return "Lyapunov loop-gain (check_lyapunov_stability) is stability companion; "
               "NOT an avalanche / SOC substitute.";
    }

private:
    std::size_t relax() {
        std::size_t topples = 0;
        std::size_t offspring = 0;
        // Bounded iterations: Prefer-A det + no infinite cascade.
        constexpr std::size_t kMaxIters = kAvalancheMaxSites * 8;
        for (std::size_t iter = 0; iter < kMaxIters; ++iter) {
            bool any = false;
            for (std::size_t i = 0; i < n_sites_; ++i) {
                if (heights_[i] < threshold_) continue;
                any = true;
                const float excess = heights_[i];
                heights_[i] = 0.0f;
                ++topples;
                // Distribute to neighbors (periodic).
                const std::size_t left = (i + n_sites_ - 1) % n_sites_;
                const std::size_t right = (i + 1) % n_sites_;
                heights_[left] += excess * 0.5f;
                heights_[right] += excess * 0.5f;
                offspring += 2;
            }
            if (!any) break;
        }
        ++events_;
        total_topples_ += topples;
        if (topples > max_size_) max_size_ = topples;
        if (topples > 0) {
            branching_sum_ += static_cast<double>(offspring) / static_cast<double>(topples);
            ++branching_count_;
        }
        // Histogram: bin by size (log-ish linear bins clipped).
        std::size_t bin = topples;
        if (bin >= kAvalancheHistBins) bin = kAvalancheHistBins - 1;
        hist_[bin] += 1;
        return topples;
    }

    std::size_t n_sites_{82};
    float threshold_{1.0f};
    std::vector<float> heights_;
    std::array<std::uint32_t, kAvalancheHistBins> hist_{};
    std::size_t events_{0};
    std::size_t total_topples_{0};
    std::size_t max_size_{0};
    double branching_sum_{0.0};
    std::size_t branching_count_{0};
};

}  // namespace population
}  // namespace kun
