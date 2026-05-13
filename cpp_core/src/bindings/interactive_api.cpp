#include "polyamg/bindings/interactive_api.hpp"

#include "polyamg/amg_policy.hpp"
#include "polyamg/discretization.hpp"
#include "polyamg/experiment_runner.hpp"
#include "polyamg/feature_extractor.hpp"
#include "polyamg/mesh_adapter.hpp"

#include <memory>

namespace polyamg::bindings {

std::vector<SampleRecord> run_experiment_in_memory(const ExperimentConfig& cfg) {
  auto sink = std::make_unique<InMemoryResultSink>();
  auto* sink_ptr = sink.get();
  ExperimentRunner runner(std::make_unique<DMPlexMeshAdapter>(), std::make_unique<DMPlexEllipticDiscretization>(),
                          std::make_unique<MatrixPoolingFeatureExtractor>(), std::make_unique<GridSearchPolicy>(),
                          std::move(sink));
  runner.run(cfg);
  return sink_ptr->records();
}

}  // namespace polyamg::bindings
