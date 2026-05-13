#pragma once

#include "polyamg/feature_extractor.hpp"
#include "polyamg/types.hpp"

#include <memory>

namespace polyamg {

class AMGPolicy {
 public:
  virtual ~AMGPolicy() = default;
  virtual double select_theta(const FeatureTensor& tensor, double h, const ModelManifest& manifest) = 0;
};

class GridSearchPolicy final : public AMGPolicy {
 public:
  double select_theta(const FeatureTensor& tensor, double h, const ModelManifest& manifest) override;
};

std::unique_ptr<AMGPolicy> create_ann_policy();

}  // namespace polyamg
