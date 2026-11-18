// Monotonic-deque sliding-window traffic shockwave detector.
// Complexity: O(n) amortized (each index pushed/popped once); O(window) space.
#pragma once
#include <vector>
#include <deque>
#include <cstddef>

namespace physrag {

inline std::vector<double> sliding_extrema(const std::vector<double>& v, int window,
                                           bool want_max) {
    std::vector<double> out;
    std::deque<int> dq;
    for (int i = 0; i < static_cast<int>(v.size()); ++i) {
        while (!dq.empty() && dq.front() <= i - window) dq.pop_front();
        if (want_max) { while (!dq.empty() && v[dq.back()] <= v[i]) dq.pop_back(); }
        else          { while (!dq.empty() && v[dq.back()] >= v[i]) dq.pop_back(); }
        dq.push_back(i);
        if (i >= window - 1) out.push_back(v[dq.front()]);
    }
    return out;
}

inline std::vector<int> shockwave_detector(const std::vector<double>& speeds,
                                           int window, double drop_fps) {
    std::vector<int> flagged;
    if (static_cast<int>(speeds.size()) < window) return flagged;
    auto mx = sliding_extrema(speeds, window, true);
    auto mn = sliding_extrema(speeds, window, false);
    for (std::size_t j = 0; j < mx.size(); ++j)
        if (mx[j] - mn[j] >= drop_fps) flagged.push_back(static_cast<int>(j) + window - 1);
    return flagged;
}

}  // namespace physrag
