#include "polyamg/amg_policy.hpp"

#include <cmath>
#include <limits>

namespace polyamg {

double GridSearchPolicy::select_theta(const FeatureTensor& tensor, double h, const ModelManifest& manifest) {
  (void)tensor;
  (void)h;
  // Placeholder policy: choose center of manifest theta grid.
  // Extension: replace with ONNX inference over theta candidates minimizing predicted rho.
  if (!manifest.theta_grid.empty()) {
    const std::size_t mid = manifest.theta_grid.size() / 2;
    return manifest.theta_grid[mid];
  }
  return 0.25;
}

}  // namespace polyamg
