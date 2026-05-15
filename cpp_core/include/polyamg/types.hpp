#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace polyamg {

enum class DiffusionPattern {
  VerticalSplit = 0,
  Checker2x2 = 1,
  VerticalStripes4 = 2,
  Checker4x4 = 3,
};

struct SampleMeta {
  std::string sample_id;
  std::string pde_type;  // elliptic|stokes
  std::string mesh_id;
  double h = 0.0;
  double theta = 0.25;
  std::uint64_t seed = 0;
  std::optional<double> epsilon;
  std::optional<double> epsilon1;
  std::optional<double> epsilon2;
  std::optional<double> viscosity;
  std::optional<double> inflow_u;
  std::optional<int> sequence_step;
  std::unordered_map<std::string, double> polygonal_descriptors;
};

struct SolverConfig {
  int max_it = 500;
  double rtol = 1e-8;
  double atol = 1e-50;
  bool use_boomeramg = true;
  std::string ksp_type = "cg";
  std::string pc_type = "hypre";
  std::string hypre_type = "boomeramg";
};

struct FeatureConfig {
  int m = 50;
  std::string op = "sum";          // sum|max|pp+np|pp+np+sum
  std::string normalize = "std+id"; // std+id|std+avg|scale+id|scale+avg|log+id|log+avg
};

struct AMGMetrics {
  int iterations = 0;
  double rho = 0.0;
  double elapsed_sec = 0.0;
  int n_levels = -1;
  int coarse_size = -1;
  bool converged = false;
  double residual_norm = 0.0;
};

struct ModelManifest {
  std::string model_id;
  std::string onnx_path;
  FeatureConfig features;
  std::vector<double> theta_grid;
  std::string preprocessing_sha256;
};

struct MeshSnapshot {
  std::string mesh_id;
  std::string mesh_path;
  int dim = 2;
  int n_cells = 0;
  int n_vertices = 0;
};

struct SequenceStep {
  int step = 0;
  MeshSnapshot snapshot;
  std::optional<std::string> parent_step_map_path;
};

}  // namespace polyamg
