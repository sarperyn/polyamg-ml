#pragma once

#include "polyamg/types.hpp"

#include <petscdm.h>
#include <petscmat.h>
#include <petscvec.h>

namespace polyamg {

class DiscretizationAdapter {
 public:
  virtual ~DiscretizationAdapter() = default;
  virtual void assemble_elliptic_system(DM dm,
                                        double h,
                                        double epsilon,
                                        DiffusionPattern pattern,
                                        Mat* A,
                                        Vec* b,
                                        Vec* x0) = 0;
};

class DMPlexEllipticDiscretization final : public DiscretizationAdapter {
 public:
  void assemble_elliptic_system(DM dm,
                                double h,
                                double epsilon,
                                DiffusionPattern pattern,
                                Mat* A,
                                Vec* b,
                                Vec* x0) override;
};

}  // namespace polyamg
