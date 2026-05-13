#pragma once

#include <petscdm.h>
#include <petscmat.h>
#include <petscvec.h>

#include <utility>

namespace polyamg {

class PetscDmHandle {
 public:
  PetscDmHandle() = default;
  explicit PetscDmHandle(DM dm) : dm_(dm) {}
  ~PetscDmHandle() { reset(); }

  PetscDmHandle(const PetscDmHandle&) = delete;
  PetscDmHandle& operator=(const PetscDmHandle&) = delete;

  PetscDmHandle(PetscDmHandle&& other) noexcept : dm_(std::exchange(other.dm_, nullptr)) {}
  PetscDmHandle& operator=(PetscDmHandle&& other) noexcept {
    if (this != &other) {
      reset();
      dm_ = std::exchange(other.dm_, nullptr);
    }
    return *this;
  }

  DM get() const { return dm_; }
  DM* out() {
    reset();
    return &dm_;
  }
  void reset(DM dm = nullptr) {
    if (dm_) DMDestroy(&dm_);
    dm_ = dm;
  }

 private:
  DM dm_ = nullptr;
};

class PetscMatHandle {
 public:
  PetscMatHandle() = default;
  explicit PetscMatHandle(Mat mat) : mat_(mat) {}
  ~PetscMatHandle() { reset(); }

  PetscMatHandle(const PetscMatHandle&) = delete;
  PetscMatHandle& operator=(const PetscMatHandle&) = delete;

  PetscMatHandle(PetscMatHandle&& other) noexcept : mat_(std::exchange(other.mat_, nullptr)) {}
  PetscMatHandle& operator=(PetscMatHandle&& other) noexcept {
    if (this != &other) {
      reset();
      mat_ = std::exchange(other.mat_, nullptr);
    }
    return *this;
  }

  Mat get() const { return mat_; }
  Mat* out() {
    reset();
    return &mat_;
  }
  void reset(Mat mat = nullptr) {
    if (mat_) MatDestroy(&mat_);
    mat_ = mat;
  }

 private:
  Mat mat_ = nullptr;
};

class PetscVecHandle {
 public:
  PetscVecHandle() = default;
  explicit PetscVecHandle(Vec vec) : vec_(vec) {}
  ~PetscVecHandle() { reset(); }

  PetscVecHandle(const PetscVecHandle&) = delete;
  PetscVecHandle& operator=(const PetscVecHandle&) = delete;

  PetscVecHandle(PetscVecHandle&& other) noexcept : vec_(std::exchange(other.vec_, nullptr)) {}
  PetscVecHandle& operator=(PetscVecHandle&& other) noexcept {
    if (this != &other) {
      reset();
      vec_ = std::exchange(other.vec_, nullptr);
    }
    return *this;
  }

  Vec get() const { return vec_; }
  Vec* out() {
    reset();
    return &vec_;
  }
  void reset(Vec vec = nullptr) {
    if (vec_) VecDestroy(&vec_);
    vec_ = vec;
  }

 private:
  Vec vec_ = nullptr;
};

}  // namespace polyamg
