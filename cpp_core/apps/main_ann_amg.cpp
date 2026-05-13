#include "polyamg/config.hpp"
#include "polyamg/amg_policy.hpp"
#include "polyamg/discretization.hpp"
#include "polyamg/experiment_runner.hpp"
#include "polyamg/feature_extractor.hpp"
#include "polyamg/mesh_adapter.hpp"

#include <petscsys.h>

#include <iostream>
#include <memory>

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, nullptr, "PolyAMG ANN-AMG run");

  if (argc < 2) {
    std::cerr << "Usage: polyamg_ann_amg <config.kv>\n";
    PetscFinalize();
    return 1;
  }

  auto cfg = polyamg::load_experiment_config(argv[1]);
  if (!cfg.model_manifest.has_value()) {
    std::cerr << "Config must include model.* fields for ANN-AMG mode.\n";
    PetscFinalize();
    return 2;
  }

#ifndef POLYAMG_WITH_ORT
  std::cerr << "[WARN] Built without ONNX Runtime. ANN policy will fallback to grid placeholder.\n";
#endif

  polyamg::ExperimentRunner runner(std::make_unique<polyamg::DMPlexMeshAdapter>(),
                                   std::make_unique<polyamg::DMPlexEllipticDiscretization>(),
                                   std::make_unique<polyamg::MatrixPoolingFeatureExtractor>(),
                                   polyamg::create_ann_policy());

  const int rc = runner.run(cfg);
  PetscFinalize();
  return rc;
}
