#pragma once

#include "polyamg/amg_policy.hpp"
#include "polyamg/amg_solver.hpp"
#include "polyamg/config.hpp"
#include "polyamg/discretization.hpp"
#include "polyamg/experiment/result_sink.hpp"
#include "polyamg/feature_extractor.hpp"
#include "polyamg/mesh_adapter.hpp"

#include <memory>

namespace polyamg {

class ExperimentRunner {
 public:
  ExperimentRunner(std::unique_ptr<MeshAdapter> mesh, std::unique_ptr<DiscretizationAdapter> discretization,
                   std::unique_ptr<FeatureExtractor> extractor, std::unique_ptr<AMGPolicy> policy);
  ExperimentRunner(std::unique_ptr<MeshAdapter> mesh, std::unique_ptr<DiscretizationAdapter> discretization,
                   std::unique_ptr<FeatureExtractor> extractor, std::unique_ptr<AMGPolicy> policy,
                   std::unique_ptr<ResultSink> sink);

  int run(const ExperimentConfig& cfg);

 private:
  std::unique_ptr<MeshAdapter> mesh_;
  std::unique_ptr<DiscretizationAdapter> discretization_;
  std::unique_ptr<FeatureExtractor> extractor_;
  std::unique_ptr<AMGPolicy> policy_;
  std::unique_ptr<ResultSink> sink_;
  AMGSolver solver_;
};

}  // namespace polyamg
