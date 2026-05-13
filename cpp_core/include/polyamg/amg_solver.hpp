#pragma once

#include "polyamg/types.hpp"

#include <petscmat.h>
#include <petscksp.h>
#include <petscvec.h>

namespace polyamg {

class AMGSolver {
 public:
  AMGMetrics solve(Mat A, Vec b, Vec x, double theta, const SolverConfig& cfg) const;
};

}  // namespace polyamg
