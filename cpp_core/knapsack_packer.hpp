// 0-1 Knapsack physics-penalized context token packer.
// Complexity: O(N * B) time/space (N items, B = token budget).
#pragma once
#include <vector>
#include <algorithm>
#include <utility>

namespace physrag {

inline std::pair<std::vector<int>, double>
knapsack_pack(const std::vector<int>& tokens, const std::vector<double>& relevance,
              const std::vector<double>& residual, int budget, double physics_lambda = 4.0) {
    std::vector<int> chosen;
    int n = static_cast<int>(tokens.size());
    if (budget <= 0 || n == 0) return {chosen, 0.0};
    std::vector<double> values(n);
    for (int i = 0; i < n; ++i)
        values[i] = std::max(0.0, relevance[i] - physics_lambda * residual[i]);
    std::vector<std::vector<double>> dp(n + 1, std::vector<double>(budget + 1, 0.0));
    for (int i = 1; i <= n; ++i) {
        int w = tokens[i - 1];
        double val = values[i - 1];
        for (int b = 0; b <= budget; ++b) {
            if (w <= b && val > 0.0)
                dp[i][b] = std::max(dp[i - 1][b - w] + val, dp[i - 1][b]);
            else
                dp[i][b] = dp[i - 1][b];
        }
    }
    int b = budget;
    for (int i = n; i >= 1; --i)
        if (dp[i][b] != dp[i - 1][b]) { chosen.push_back(i - 1); b -= tokens[i - 1]; }
    std::reverse(chosen.begin(), chosen.end());
    return {chosen, dp[n][budget]};
}

}  // namespace physrag
