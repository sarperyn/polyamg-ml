#include "polyamg/feature_extractor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace polyamg {
namespace {

double signed_log_scale(double v) {
  if (v == 0.0) return 0.0;
  const double s = v > 0 ? 1.0 : -1.0;
  return s * std::log(std::abs(v) + 1.0);
}

void normalize_inplace(std::vector<double>& v, const std::string& mode) {
  if (v.empty()) return;
  if (mode.rfind("std", 0) == 0) {
    double mean = 0.0;
    for (double x : v) mean += x;
    mean /= static_cast<double>(v.size());
    double var = 0.0;
    for (double x : v) var += (x - mean) * (x - mean);
    var /= static_cast<double>(v.size());
    const double sigma = std::sqrt(std::max(var, 1e-16));
    for (double& x : v) x = (x - mean) / sigma;
    return;
  }

  if (mode.rfind("scale", 0) == 0 || mode.rfind("log", 0) == 0) {
    if (mode.rfind("log", 0) == 0) {
      for (double& x : v) x = signed_log_scale(x);
    }
    double mx = 0.0;
    for (double x : v) mx = std::max(mx, std::abs(x));
    if (mx < 1e-16) mx = 1.0;
    for (double& x : v) x /= mx;
    return;
  }

  throw std::runtime_error("Unknown normalize mode: " + mode);
}

void normalize_channelwise(std::vector<double>& v, int c, int m, const std::string& mode) {
  if (c <= 0 || m <= 0) return;
  const std::size_t chan_sz = static_cast<std::size_t>(m) * static_cast<std::size_t>(m);
  for (int ch = 0; ch < c; ++ch) {
    const std::size_t off = static_cast<std::size_t>(ch) * chan_sz;
    std::vector<double> tmp(chan_sz);
    std::copy(v.begin() + static_cast<std::ptrdiff_t>(off),
              v.begin() + static_cast<std::ptrdiff_t>(off + chan_sz), tmp.begin());
    normalize_inplace(tmp, mode);
    std::copy(tmp.begin(), tmp.end(),
              v.begin() + static_cast<std::ptrdiff_t>(off));
  }
}

}  // namespace

FeatureTensor MatrixPoolingFeatureExtractor::extract(Mat A, const FeatureConfig& cfg) {
  if (cfg.m <= 0) throw std::runtime_error("FeatureConfig.m must be > 0");

  PetscInt nrows, ncols;
  MatGetSize(A, &nrows, &ncols);
  if (nrows != ncols) throw std::runtime_error("Expected square matrix for pooling");

  const int m = cfg.m;
  int c = 1;
  if (cfg.op == "pp+np") c = 2;
  if (cfg.op == "pp+np+sum") c = 3;
  const std::size_t chan_sz = static_cast<std::size_t>(m) * static_cast<std::size_t>(m);
  std::vector<double> V(static_cast<std::size_t>(c) * chan_sz, 0.0);
  std::vector<double> C(static_cast<std::size_t>(c) * chan_sz, 0.0);

  const PetscInt q = nrows / m;
  const PetscInt p = nrows % m;
  const PetscInt t = (q + 1) * p;

  auto map_idx = [&](PetscInt r) -> int {
    if (r < t) return static_cast<int>(r / (q + 1));
    return static_cast<int>((r - t) / std::max<PetscInt>(q, 1) + p);
  };

  for (PetscInt r = 0; r < nrows; ++r) {
    const PetscInt* cols = nullptr;
    const PetscScalar* vals = nullptr;
    PetscInt nz = 0;
    MatGetRow(A, r, &nz, &cols, &vals);
    const int i = std::min(map_idx(r), m - 1);

    for (PetscInt k = 0; k < nz; ++k) {
      const int j = std::min(map_idx(cols[k]), m - 1);
      const size_t idx0 = static_cast<size_t>(i * m + j);
      const double v = static_cast<double>(vals[k]);

      if (cfg.op == "sum") {
        V[idx0] += v;
        C[idx0] += 1.0;
      } else if (cfg.op == "max") {
        V[idx0] = std::max(V[idx0], std::abs(v));
        C[idx0] += 1.0;
      } else if (cfg.op == "pp+np") {
        // channel 0: positive part max, channel 1: negative part max
        const size_t idx_pp = idx0;
        const size_t idx_np = chan_sz + idx0;
        V[idx_pp] = std::max(V[idx_pp], std::max(0.0, v));
        V[idx_np] = std::max(V[idx_np], std::max(0.0, -v));
        C[idx_pp] += 1.0;
        C[idx_np] += 1.0;
      } else if (cfg.op == "pp+np+sum") {
        // ch0: pp(max), ch1: np(max), ch2: sum
        const size_t idx_pp = idx0;
        const size_t idx_np = chan_sz + idx0;
        const size_t idx_sm = 2 * chan_sz + idx0;
        V[idx_pp] = std::max(V[idx_pp], std::max(0.0, v));
        V[idx_np] = std::max(V[idx_np], std::max(0.0, -v));
        V[idx_sm] += v;
        C[idx_pp] += 1.0;
        C[idx_np] += 1.0;
        C[idx_sm] += 1.0;
      } else {
        throw std::runtime_error("Unknown op mode: " + cfg.op);
      }
    }
    MatRestoreRow(A, r, &nz, &cols, &vals);
  }

  const bool avg = cfg.normalize.find("+avg") != std::string::npos;
  if (avg) {
    for (size_t i = 0; i < V.size(); ++i) {
      if (C[i] > 0.0) V[i] /= C[i];
    }
  }

  std::string base_mode = cfg.normalize;
  const auto plus_pos = base_mode.find('+');
  if (plus_pos != std::string::npos) base_mode = base_mode.substr(0, plus_pos);
  normalize_channelwise(V, c, m, base_mode);

  FeatureTensor out;
  out.m = m;
  out.c = c;
  out.values = std::move(V);
  return out;
}

}  // namespace polyamg
