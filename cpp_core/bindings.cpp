// pybind11 bindings exposing the C++17 algorithmic core as `physrag_cpp`.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <limits>
#include "subgraph_extractor.hpp"
#include "shockwave_window.hpp"
#include "knapsack_packer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(physrag_cpp, m) {
    m.doc() = "physrag-v2x C++17 algorithmic core (Dijkstra subgraph, monotonic-deque "
              "shockwave detector, physics-penalized knapsack packer)";

    m.def("multi_source_dijkstra_subgraph",
          [](py::dict graph, std::vector<int64_t> sources, int max_hops, double max_dist) {
              physrag::Graph g;
              for (auto item : graph) {
                  int64_t u = item.first.cast<int64_t>();
                  std::vector<std::pair<int64_t, double>> nbrs;
                  for (auto e : item.second.cast<py::list>()) {
                      auto t = e.cast<py::tuple>();
                      nbrs.emplace_back(t[0].cast<int64_t>(), t[1].cast<double>());
                  }
                  g[u] = std::move(nbrs);
              }
              auto best = physrag::multi_source_dijkstra_subgraph(g, sources, max_hops, max_dist);
              py::dict out;
              for (const auto& kv : best)
                  out[py::int_(kv.first)] = py::make_tuple(kv.second.first, kv.second.second);
              return out;
          },
          py::arg("graph"), py::arg("sources"), py::arg("max_hops"),
          py::arg("max_dist") = std::numeric_limits<double>::infinity());

    m.def("shockwave_detector", &physrag::shockwave_detector,
          py::arg("speeds"), py::arg("window"), py::arg("drop_fps"));

    m.def("knapsack_pack", &physrag::knapsack_pack,
          py::arg("tokens"), py::arg("relevance"), py::arg("residual"),
          py::arg("budget"), py::arg("physics_lambda") = 4.0);
}
