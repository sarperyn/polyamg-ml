#pragma once

#include "polyamg/types.hpp"

#include <string>
#include <vector>

namespace polyamg {

struct ExperimentConfig {
  std::string experiment_id;
  std::string output_dir;
  std::string mesh_path;
  std::string pde = "elliptic";
  std::vector<double> theta_values;
  std::vector<double> epsilon_values;
  std::vector<double> epsilon1_values;
  std::vector<double> epsilon2_values;
  std::vector<double> h_values;
  DiffusionPattern diffusion_pattern = DiffusionPattern::Checker2x2;
  SolverConfig solver;
  FeatureConfig features;
  std::optional<ModelManifest> model_manifest;
  std::uint64_t seed = 1;
};

ExperimentConfig load_experiment_config(const std::string& path);

}  // namespace polyamg
