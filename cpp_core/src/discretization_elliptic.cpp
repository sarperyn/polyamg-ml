#include "polyamg/discretization.hpp"

#include <petscdmplex.h>

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace polyamg {

namespace {
constexpr double kPi = 3.14159265358979323846;

int pattern_k(DiffusionPattern pattern) {
  if (pattern == DiffusionPattern::VerticalSplit || pattern == DiffusionPattern::Checker2x2) return 1;
  return 2;
}

int tile_index(double x, int n) {
  const double xi = (x + 1.0) * 0.5;
  int idx = static_cast<int>(std::floor(xi * static_cast<double>(n)));
  if (idx < 0) idx = 0;
  if (idx >= n) idx = n - 1;
  return idx;
}

double mu_value(double x, double y, double epsilon1, double epsilon2, DiffusionPattern pattern) {
  const double kappa_white = std::pow(10.0, epsilon1);
  const double kappa_gray = std::pow(10.0, epsilon2);
  bool gray = false;
  switch (pattern) {
    case DiffusionPattern::VerticalSplit:
      gray = (x >= 0.0);
      break;
    case DiffusionPattern::Checker2x2:
      gray = (x * y > 0.0);
      break;
    case DiffusionPattern::VerticalStripes4: {
      const int ix = tile_index(x, 4);
      gray = (ix % 2 == 1);
      break;
    }
    case DiffusionPattern::Checker4x4: {
      const int ix = tile_index(x, 4);
      const int iy = tile_index(y, 4);
      gray = ((ix + iy) % 2 == 1);
      break;
    }
  }
  return gray ? kappa_gray : kappa_white;
}

double exact_u(double x, double y, int k) {
  return std::cos(kPi * static_cast<double>(k) * x) * std::cos(kPi * static_cast<double>(k) * y);
}

double forcing_f(double x, double y, double mu, int k) {
  // For piecewise-constant mu, use -mu * Laplacian(u) inside each tile; interface flux continuity is not enforced.
  const double kp = kPi * static_cast<double>(k);
  const double u = exact_u(x, y, k);
  return 2.0 * kp * kp * mu * u;
}

}  // namespace

void DMPlexEllipticDiscretization::assemble_elliptic_system(DM dm,
                                                            double h,
                                                            double epsilon1,
                                                            double epsilon2,
                                                            DiffusionPattern pattern,
                                                            Mat* A,
                                                            Vec* b,
                                                            Vec* x0) {
  (void)dm;
  if (!A || !b || !x0) throw std::runtime_error("Null output pointer in assemble_elliptic_system");

  const double h_safe = std::max(1e-6, h);
  const int nx = std::max(2, static_cast<int>(std::round(2.0 / h_safe)));
  const int ny = nx;
  const double hx = 2.0 / static_cast<double>(nx);
  const double hy = 2.0 / static_cast<double>(ny);
  const PetscInt n_dof = static_cast<PetscInt>((nx + 1) * (ny + 1));
  const int k = pattern_k(pattern);

  MatCreate(PETSC_COMM_WORLD, A);
  MatSetSizes(*A, PETSC_DECIDE, PETSC_DECIDE, n_dof, n_dof);
  MatSetType(*A, MATAIJ);
  MatMPIAIJSetPreallocation(*A, 9, nullptr, 9, nullptr);
  MatSeqAIJSetPreallocation(*A, 9, nullptr);
  MatSetUp(*A);

  VecCreate(PETSC_COMM_WORLD, b);
  VecSetSizes(*b, PETSC_DECIDE, n_dof);
  VecSetFromOptions(*b);

  auto node = [nx](int i, int j) { return static_cast<PetscInt>(j * (nx + 1) + i); };

  const double q = 1.0 / std::sqrt(3.0);
  const std::array<double, 2> qpts{-q, q};

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const double x0c = -1.0 + static_cast<double>(i) * hx;
      const double y0c = -1.0 + static_cast<double>(j) * hy;

      const std::array<PetscInt, 4> nodes = {
          node(i, j),
          node(i + 1, j),
          node(i + 1, j + 1),
          node(i, j + 1),
      };

      double ke[16] = {0.0};
      double fe[4] = {0.0};

      for (double xi : qpts) {
        for (double eta : qpts) {
          const double N1 = 0.25 * (1.0 - xi) * (1.0 - eta);
          const double N2 = 0.25 * (1.0 + xi) * (1.0 - eta);
          const double N3 = 0.25 * (1.0 + xi) * (1.0 + eta);
          const double N4 = 0.25 * (1.0 - xi) * (1.0 + eta);

          const double dN1_dxi = -0.25 * (1.0 - eta);
          const double dN2_dxi = 0.25 * (1.0 - eta);
          const double dN3_dxi = 0.25 * (1.0 + eta);
          const double dN4_dxi = -0.25 * (1.0 + eta);

          const double dN1_deta = -0.25 * (1.0 - xi);
          const double dN2_deta = -0.25 * (1.0 + xi);
          const double dN3_deta = 0.25 * (1.0 + xi);
          const double dN4_deta = 0.25 * (1.0 - xi);

          const double x = x0c + (xi + 1.0) * hx * 0.5;
          const double y = y0c + (eta + 1.0) * hy * 0.5;

          const double mu = mu_value(x, y, epsilon1, epsilon2, pattern);
          const double f = forcing_f(x, y, mu, k);

          const double detJ = hx * hy * 0.25;
          const double dxi_dx = 2.0 / hx;
          const double deta_dy = 2.0 / hy;

          const double dNdx[4] = {
              dN1_dxi * dxi_dx,
              dN2_dxi * dxi_dx,
              dN3_dxi * dxi_dx,
              dN4_dxi * dxi_dx,
          };
          const double dNdy[4] = {
              dN1_deta * deta_dy,
              dN2_deta * deta_dy,
              dN3_deta * deta_dy,
              dN4_deta * deta_dy,
          };

          const double phi[4] = {N1, N2, N3, N4};

          for (int a = 0; a < 4; ++a) {
            fe[a] += f * phi[a] * detJ;
            for (int b_idx = 0; b_idx < 4; ++b_idx) {
              ke[a * 4 + b_idx] += mu * (dNdx[a] * dNdx[b_idx] + dNdy[a] * dNdy[b_idx]) * detJ;
            }
          }
        }
      }

      MatSetValues(*A, 4, nodes.data(), 4, nodes.data(), ke, ADD_VALUES);
      VecSetValues(*b, 4, nodes.data(), fe, ADD_VALUES);
    }
  }

  MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY);
  VecAssemblyBegin(*b);
  VecAssemblyEnd(*b);

  std::vector<PetscInt> rows;
  rows.reserve(static_cast<std::size_t>(2 * (nx + ny) + 4));

  Vec x_bc;
  VecDuplicate(*b, &x_bc);
  VecSet(x_bc, 0.0);

  for (int j = 0; j <= ny; ++j) {
    for (int i = 0; i <= nx; ++i) {
      if (i == 0 || j == 0 || i == nx || j == ny) {
        const double x = -1.0 + static_cast<double>(i) * hx;
        const double y = -1.0 + static_cast<double>(j) * hy;
        const double g = exact_u(x, y, k);
        const PetscInt id = node(i, j);
        rows.push_back(id);
        VecSetValue(x_bc, id, g, INSERT_VALUES);
      }
    }
  }

  VecAssemblyBegin(x_bc);
  VecAssemblyEnd(x_bc);

  if (!rows.empty()) {
    MatZeroRowsColumns(*A, static_cast<PetscInt>(rows.size()), rows.data(), 1.0, x_bc, *b);
  }

  VecDuplicate(*b, x0);
  VecCopy(x_bc, *x0);
  VecDestroy(&x_bc);
}

}  // namespace polyamg
