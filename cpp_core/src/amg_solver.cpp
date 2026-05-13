#include "polyamg/amg_solver.hpp"

#include <petscpc.h>
#include <petsctime.h>
#include <regex>
#include <string>

namespace polyamg {
namespace {
void parse_hypre_stats(const std::string& s, AMGMetrics* out) {
  if (!out) return;
  std::smatch m;
  std::regex levels_re("Number of levels[^0-9]*([0-9]+)");
  if (std::regex_search(s, m, levels_re) && m.size() > 1) {
    out->n_levels = std::stoi(m[1].str());
  }

  // Accept several common print variants.
  std::regex coarse_re1("Coarse grid size[^0-9]*([0-9]+)");
  std::regex coarse_re2("coarsest[^0-9]*size[^0-9]*([0-9]+)", std::regex_constants::icase);
  if (std::regex_search(s, m, coarse_re1) && m.size() > 1) {
    out->coarse_size = std::stoi(m[1].str());
    return;
  }
  if (std::regex_search(s, m, coarse_re2) && m.size() > 1) {
    out->coarse_size = std::stoi(m[1].str());
  }
}
}  // namespace

AMGMetrics AMGSolver::solve(Mat A, Vec b, Vec x, double theta, const SolverConfig& cfg) const {
  AMGMetrics out;

  PetscReal rnorm0 = 0.0;
  {
    Vec r0;
    VecDuplicate(b, &r0);
    MatMult(A, x, r0);
    VecWAXPY(r0, -1.0, r0, b);  // r0 = b - A * x
    VecNorm(r0, NORM_2, &rnorm0);
    VecDestroy(&r0);
  }

  KSP ksp;
  KSPCreate(PETSC_COMM_WORLD, &ksp);
  KSPSetOperators(ksp, A, A);
  KSPSetType(ksp, cfg.ksp_type.c_str());
  KSPSetTolerances(ksp, cfg.rtol, cfg.atol, PETSC_DEFAULT, cfg.max_it);

  PC pc;
  KSPGetPC(ksp, &pc);
  PCSetType(pc, cfg.pc_type.c_str());

  if (cfg.pc_type == "hypre") {
    PCHYPRESetType(pc, cfg.hypre_type.c_str());
    if (cfg.hypre_type == "boomeramg") {
      // Mirrors paper's theta role via boomeramg strong-threshold.
      PCSetFromOptions(pc);
      char opt_val[64];
      PetscSNPrintf(opt_val, sizeof(opt_val), "%g", theta);
      PetscOptionsSetValue(nullptr, "-pc_hypre_boomeramg_strong_threshold", opt_val);
      PetscOptionsSetValue(nullptr, "-pc_hypre_boomeramg_print_statistics", "1");
    }
  }

  KSPSetFromOptions(ksp);

  PetscLogDouble t0 = 0.0, t1 = 0.0;
  PetscTime(&t0);
  KSPSolve(ksp, b, x);
  PetscTime(&t1);

  KSPConvergedReason reason;
  KSPGetConvergedReason(ksp, &reason);
  out.converged = (reason > 0);

  PetscInt its = 0;
  KSPGetIterationNumber(ksp, &its);
  out.iterations = static_cast<int>(its);

  PetscReal rnorm = 0.0;
  KSPGetResidualNorm(ksp, &rnorm);
  out.residual_norm = static_cast<double>(rnorm);

  // Paper proxy: rho = (||r_k|| / ||r_0||)^(1/k) with r_0 from initial guess.
  const double rel = std::max(1e-16, static_cast<double>(rnorm) / std::max(1e-16, static_cast<double>(rnorm0)));
  out.rho = (its > 0) ? std::pow(rel, 1.0 / static_cast<double>(its)) : rel;

  out.elapsed_sec = static_cast<double>(t1 - t0);
  out.n_levels = -1;
  out.coarse_size = -1;

  // Capture solver/PC textual view and parse BoomerAMG diagnostics.
  {
    char view_buf[131072];
    view_buf[0] = '\0';
    PetscViewer viewer = nullptr;
    if (!PetscViewerStringOpen(PETSC_COMM_SELF, view_buf, sizeof(view_buf), &viewer)) {
      KSPView(ksp, viewer);
      PetscViewerDestroy(&viewer);
      parse_hypre_stats(std::string(view_buf), &out);
    }
  }

  KSPDestroy(&ksp);
  return out;
}

}  // namespace polyamg
