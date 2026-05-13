#pragma once

#include "polyamg/types.hpp"

#include <petscdm.h>

namespace polyamg {

class MeshAdapter {
 public:
  virtual ~MeshAdapter() = default;
  virtual MeshSnapshot load(const std::string& mesh_path, DM* dm) = 0;
};

class DMPlexMeshAdapter final : public MeshAdapter {
 public:
  MeshSnapshot load(const std::string& mesh_path, DM* dm) override;
};

}  // namespace polyamg
