#pragma once

#include "polyamg/types.hpp"

#include <petscmat.h>

namespace polyamg {

struct FeatureTensor {
  int m = 0;
  int c = 1;
  std::vector<double> values;  // CHW flattened
};

class FeatureExtractor {
 public:
  virtual ~FeatureExtractor() = default;
  virtual FeatureTensor extract(Mat A, const FeatureConfig& cfg) = 0;
};

class MatrixPoolingFeatureExtractor final : public FeatureExtractor {
 public:
  FeatureTensor extract(Mat A, const FeatureConfig& cfg) override;
};

}  // namespace polyamg
