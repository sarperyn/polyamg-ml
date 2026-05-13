#include "polyamg/mesh_adapter.hpp"

#include <petscdmplex.h>
#include <stdexcept>

namespace polyamg {

MeshSnapshot DMPlexMeshAdapter::load(const std::string& mesh_path, DM* dm) {
  if (!dm) throw std::runtime_error("Null DM pointer");

  PetscErrorCode ierr;
  if (mesh_path.empty()) {
    // PETSc API here expects 11 args:
    // (comm, dim, simplex, faces, lower, upper, periodicity, interpolate, localize, sparseLocalize, dm)
    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, 2, PETSC_FALSE, nullptr, nullptr, nullptr, nullptr, PETSC_TRUE,
                               PETSC_TRUE, PETSC_FALSE, dm);
    if (ierr) throw std::runtime_error("DMPlexCreateBoxMesh failed");
  } else {
    ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, mesh_path.c_str(), nullptr, PETSC_TRUE, dm);
    if (ierr) throw std::runtime_error("DMPlexCreateFromFile failed for " + mesh_path);
  }

  PetscInt cStart, cEnd, vStart, vEnd;
  DMPlexGetHeightStratum(*dm, 0, &cStart, &cEnd);
  DMPlexGetDepthStratum(*dm, 0, &vStart, &vEnd);

  MeshSnapshot m;
  m.mesh_id = mesh_path.empty() ? "boxmesh2d" : mesh_path;
  m.mesh_path = mesh_path;
  m.dim = 2;
  m.n_cells = static_cast<int>(cEnd - cStart);
  m.n_vertices = static_cast<int>(vEnd - vStart);
  return m;
}

}  // namespace polyamg
