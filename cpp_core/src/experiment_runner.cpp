#include "polyamg/experiment_runner.hpp"
#include "polyamg/core/petsc_handles.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

namespace polyamg {

namespace {
struct EpsilonPair {
  double epsilon1;
  double epsilon2;
};

std::vector<EpsilonPair> epsilon_pairs(const ExperimentConfig& cfg) {
  std::vector<EpsilonPair> pairs;
  if (!cfg.epsilon1_values.empty() && !cfg.epsilon2_values.empty()) {
    for (double eps1 : cfg.epsilon1_values) {
      for (double eps2 : cfg.epsilon2_values) {
        pairs.push_back({eps1, eps2});
      }
    }
    return pairs;
  }

  for (double eps : cfg.epsilon_values) {
    pairs.push_back({0.0, eps});
  }
  return pairs;
}

void populate_meta(SampleMeta& meta,
                   const ExperimentConfig& cfg,
                   const MeshSnapshot& snap,
                   std::size_t sample_id,
                   double h,
                   double theta,
                   const EpsilonPair& eps) {
  meta.sample_id = "sample_" + std::to_string(sample_id);
  meta.pde_type = cfg.pde;
  meta.mesh_id = snap.mesh_id;
  meta.h = h;
  meta.theta = theta;
  meta.seed = cfg.seed;
  meta.epsilon = eps.epsilon2;
  meta.epsilon1 = eps.epsilon1;
  meta.epsilon2 = eps.epsilon2;
  meta.polygonal_descriptors = {
      {"cell_arity_mean", 0.0},
      {"cell_arity_std", 0.0},
      {"non_orthogonality_proxy", 0.0},
      {"diffusion_pattern", static_cast<double>(cfg.diffusion_pattern)},
  };
}
}  // namespace

ExperimentRunner::ExperimentRunner(std::unique_ptr<MeshAdapter> mesh,
                                   std::unique_ptr<DiscretizationAdapter> discretization,
                                   std::unique_ptr<FeatureExtractor> extractor,
                                   std::unique_ptr<AMGPolicy> policy)
    : mesh_(std::move(mesh)),
      discretization_(std::move(discretization)),
      extractor_(std::move(extractor)),
      policy_(std::move(policy)) {}

ExperimentRunner::ExperimentRunner(std::unique_ptr<MeshAdapter> mesh,
                                   std::unique_ptr<DiscretizationAdapter> discretization,
                                   std::unique_ptr<FeatureExtractor> extractor,
                                   std::unique_ptr<AMGPolicy> policy,
                                   std::unique_ptr<ResultSink> sink)
    : mesh_(std::move(mesh)),
      discretization_(std::move(discretization)),
      extractor_(std::move(extractor)),
      policy_(std::move(policy)),
      sink_(std::move(sink)) {}

int ExperimentRunner::run(const ExperimentConfig& cfg) {
  std::filesystem::create_directories(cfg.output_dir);
  if (!sink_) {
    sink_ = std::make_unique<JsonResultSink>(cfg.output_dir, cfg.experiment_id);
  }

  PetscDmHandle dm;
  const MeshSnapshot snap = mesh_->load(cfg.mesh_path, dm.out());

  std::size_t rec_id = 0;
  for (double h : cfg.h_values) {
    for (const EpsilonPair& eps : epsilon_pairs(cfg)) {
      PetscMatHandle A;
      PetscVecHandle b;
      PetscVecHandle x;

      discretization_->assemble_elliptic_system(
          dm.get(), h, eps.epsilon1, eps.epsilon2, cfg.diffusion_pattern, A.out(), b.out(), x.out());
      const FeatureTensor tensor = extractor_->extract(A.get(), cfg.features);

      if (cfg.model_manifest.has_value()) {
        const double theta_star = policy_->select_theta(tensor, h, *cfg.model_manifest);
        const AMGMetrics m = solver_.solve(A.get(), b.get(), x.get(), theta_star, cfg.solver);

        SampleMeta meta;
        populate_meta(meta, cfg, snap, rec_id++, h, theta_star, eps);

        SampleRecord record{meta, m, cfg.features, tensor, true};
        sink_->write(record);
      } else {
        for (double theta : cfg.theta_values) {
          VecZeroEntries(x.get());
          const AMGMetrics m = solver_.solve(A.get(), b.get(), x.get(), theta, cfg.solver);

          SampleMeta meta;
          populate_meta(meta, cfg, snap, rec_id++, h, theta, eps);

          SampleRecord record{meta, m, cfg.features, tensor, true};
          sink_->write(record);
        }
      }
    }
  }

  std::cout << "Completed experiment: " << cfg.experiment_id << " on mesh " << snap.mesh_id << "\n";
  return 0;
}

}  // namespace polyamg
