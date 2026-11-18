// Multi-source bounded Dijkstra causal subgraph extraction.
// Complexity: O((V + E) log V) via a binary heap; each node settled once.
#pragma once
#include <vector>
#include <queue>
#include <tuple>
#include <unordered_map>
#include <limits>
#include <cstdint>

namespace physrag {

using Graph = std::unordered_map<int64_t, std::vector<std::pair<int64_t, double>>>;

inline std::unordered_map<int64_t, std::pair<double, int>>
multi_source_dijkstra_subgraph(const Graph& g,
                               const std::vector<int64_t>& sources,
                               int max_hops,
                               double max_dist = std::numeric_limits<double>::infinity()) {
    std::unordered_map<int64_t, std::pair<double, int>> best;
    using QE = std::tuple<double, int, int64_t>;   // (dist, hops, node)
    std::priority_queue<QE, std::vector<QE>, std::greater<QE>> pq;
    for (auto s : sources) pq.emplace(0.0, 0, s);
    while (!pq.empty()) {
        auto [dist, hops, u] = pq.top();
        pq.pop();
        auto it = best.find(u);
        if (it != best.end() && it->second.first <= dist) continue;
        best[u] = {dist, hops};
        if (hops >= max_hops) continue;
        auto git = g.find(u);
        if (git == g.end()) continue;
        for (const auto& [v, w] : git->second) {
            double nd = dist + w;
            if (nd > max_dist) continue;
            auto vit = best.find(v);
            if (vit == best.end() || nd < vit->second.first) pq.emplace(nd, hops + 1, v);
        }
    }
    return best;
}

}  // namespace physrag
