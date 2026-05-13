#include "polyamg/amg_solver.hpp"

#include <petscpc.h>
#include <petsctime.h>

#include <cstdio>
#include <regex>
#include <string>

#include <unistd.h>

namespace polyamg {
namespace {
template <typename Fn>
std::string capture_stdio(Fn fn) {
  std::string output;
  FILE* tmp = std::tmpfile();
  if (!tmp) {
    fn();
    return output;
  }
  const int stdout_fd = dup(fileno(stdout));
  const int stderr_fd = dup(fileno(stderr));
  if (stdout_fd < 0 || stderr_fd < 0) {
    std::fclose(tmp);
    fn();
    return output;
  }
  std::fflush(stdout);
  std::fflush(stderr);
  dup2(fileno(tmp), fileno(stdout));
  dup2(fileno(tmp), fileno(stderr));

  fn();

  std::fflush(stdout);
  std::fflush(stderr);
  dup2(stdout_fd, fileno(stdout));
  dup2(stderr_fd, fileno(stderr));
  close(stdout_fd);
  close(stderr_fd);

  std::fseek(tmp, 0, SEEK_SET);
  char buffer[4096];
  size_t n = 0;
  while ((n = std::fread(buffer, 1, sizeof(buffer), tmp)) > 0) {
    output.append(buffer, n);
  }
  std::fclose(tmp);
  return output;
}

void parse_hypre_stats(const std::string& s, AMGMetrics* out) {
  if (!out) return;
  std::smatch m;
  std::regex levels_re("(^|\\n)\\s*(Number of levels|Num levels)[^0-9]*([0-9]+)",
                       std::regex_constants::icase);
  if (std::regex_search(s, m, levels_re) && m.size() > 3) {
    out->n_levels = std::stoi(m[3].str());
  }

  // Accept several common print variants.
  std::regex coarse_re1("Coarse grid size[^0-9]*([0-9]+)", std::regex_constants::icase);
  std::regex coarse_re2("coarsest[^0-9]*size[^0-9]*([0-9]+)", std::regex_constants::icase);
  std::regex coarse_re3("coarse[^0-9]*size[^0-9]*([0-9]+)", std::regex_constants::icase);
  if (std::regex_search(s, m, coarse_re1) && m.size() > 1) {
    out->coarse_size = std::stoi(m[1].str());
    return;
  }
  if (std::regex_search(s, m, coarse_re2) && m.size() > 1) {
    out->coarse_size = std::stoi(m[1].str());
    return;
  }
  if (std::regex_search(s, m, coarse_re3) && m.size() > 1) {
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
    VecAYPX(r0, -1.0, b);  // r0 = b - A * x
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
  const std::string hypre_log = capture_stdio([&]() { KSPSolve(ksp, b, x); });
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

  if (cfg.pc_type == "hypre" && cfg.hypre_type == "boomeramg") {
    PetscInt nlevels = 0;
    Mat* operators = nullptr;
    const PetscErrorCode ierr = PCGetCoarseOperators(pc, &nlevels, &operators);
    if (!ierr && nlevels > 0) {
      out.n_levels = static_cast<int>(nlevels);
      PetscInt min_size = -1;
      const PetscInt ncoarse_ops = nlevels > 0 ? nlevels - 1 : 0;
      for (PetscInt i = 0; i < ncoarse_ops; ++i) {
        if (!operators || !operators[i]) continue;
        PetscInt m = 0, n = 0;
        MatGetSize(operators[i], &m, &n);
        const PetscInt size = m < n ? m : n;
        if (min_size < 0 || size < min_size) min_size = size;
      }
      if (min_size >= 0) out.coarse_size = static_cast<int>(min_size);
    }
    if (operators) {
      const PetscInt ncoarse_ops = nlevels > 0 ? nlevels - 1 : 0;
      for (PetscInt i = 0; i < ncoarse_ops; ++i) {
        if (operators[i]) MatDestroy(&operators[i]);
      }
      PetscFree(operators);
    }
  }

  if (!hypre_log.empty() && (out.n_levels < 0 || out.coarse_size < 0)) {
    AMGMetrics parsed = out;
    parse_hypre_stats(hypre_log, &parsed);
    if (out.n_levels < 0) out.n_levels = parsed.n_levels;
    if (out.coarse_size < 0) out.coarse_size = parsed.coarse_size;
  }

  // Capture solver/PC textual view and parse BoomerAMG diagnostics.
  {
    char view_buf[131072];
    view_buf[0] = '\0';
    PetscViewer viewer = nullptr;
    if (!PetscViewerStringOpen(PETSC_COMM_SELF, view_buf, sizeof(view_buf), &viewer)) {
      KSPView(ksp, viewer);
      PetscViewerDestroy(&viewer);
      if (out.n_levels < 0 || out.coarse_size < 0) {
        parse_hypre_stats(std::string(view_buf), &out);
      }
    }
  }

  KSPDestroy(&ksp);
  return out;
}

}  // namespace polyamg
