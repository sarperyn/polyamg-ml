#include "polyamg/config.hpp"
#include "polyamg/discretization.hpp"
#include "polyamg/experiment_runner.hpp"
#include "polyamg/feature_extractor.hpp"
#include "polyamg/mesh_adapter.hpp"

#include <petscsys.h>

#include <iostream>
#include <memory>

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, nullptr, "PolyAMG baseline sweep");

  if (argc < 2) {
    std::cerr << "Usage: polyamg_baseline <config.kv>\n";
    PetscFinalize();
    return 1;
  }

  auto cfg = polyamg::load_experiment_config(argv[1]);

  polyamg::ExperimentRunner runner(std::make_unique<polyamg::DMPlexMeshAdapter>(),
                                   std::make_unique<polyamg::DMPlexEllipticDiscretization>(),
                                   std::make_unique<polyamg::MatrixPoolingFeatureExtractor>(),
                                   std::make_unique<polyamg::GridSearchPolicy>());

  const int rc = runner.run(cfg);
  PetscFinalize();
  return rc;
}
