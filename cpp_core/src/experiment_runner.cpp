#include "polyamg/experiment_runner.hpp"
#include "polyamg/core/petsc_handles.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

namespace polyamg {

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
    for (double eps : cfg.epsilon_values) {
      PetscMatHandle A;
      PetscVecHandle b;
      PetscVecHandle x;

      discretization_->assemble_elliptic_system(dm.get(), h, eps, cfg.diffusion_pattern, A.out(), b.out(), x.out());
      const FeatureTensor tensor = extractor_->extract(A.get(), cfg.features);

      if (cfg.model_manifest.has_value()) {
        const double theta_star = policy_->select_theta(tensor, h, *cfg.model_manifest);
        const AMGMetrics m = solver_.solve(A.get(), b.get(), x.get(), theta_star, cfg.solver);

        SampleMeta meta;
        meta.sample_id = "sample_" + std::to_string(rec_id++);
        meta.pde_type = cfg.pde;
        meta.mesh_id = snap.mesh_id;
        meta.h = h;
        meta.theta = theta_star;
        meta.seed = cfg.seed;
        meta.epsilon = eps;
        meta.polygonal_descriptors = {
          {"cell_arity_mean", 0.0},
          {"cell_arity_std", 0.0},
          {"non_orthogonality_proxy", 0.0},
          {"diffusion_pattern", static_cast<double>(cfg.diffusion_pattern)},
        };

        SampleRecord record{meta, m, cfg.features, tensor, true};
        sink_->write(record);
      } else {
        for (double theta : cfg.theta_values) {
          VecZeroEntries(x.get());
          const AMGMetrics m = solver_.solve(A.get(), b.get(), x.get(), theta, cfg.solver);

          SampleMeta meta;
          meta.sample_id = "sample_" + std::to_string(rec_id++);
          meta.pde_type = cfg.pde;
          meta.mesh_id = snap.mesh_id;
          meta.h = h;
          meta.theta = theta;
          meta.seed = cfg.seed;
          meta.epsilon = eps;
            meta.polygonal_descriptors = {
              {"cell_arity_mean", 0.0},
              {"cell_arity_std", 0.0},
              {"non_orthogonality_proxy", 0.0},
              {"diffusion_pattern", static_cast<double>(cfg.diffusion_pattern)},
            };

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
