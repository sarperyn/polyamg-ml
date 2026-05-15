#include "polyamg/bindings/interactive_api.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <utility>

namespace py = pybind11;

namespace {

template <typename T>
T get_or(const py::dict& d, const char* key, T fallback) {
  return d.contains(key) ? py::cast<T>(d[key]) : fallback;
}

std::vector<double> get_float_vector_or(const py::dict& d, const char* key, std::vector<double> fallback) {
  return d.contains(key) ? py::cast<std::vector<double>>(d[key]) : std::move(fallback);
}

polyamg::ExperimentConfig config_from_dict(const py::dict& d) {
  polyamg::ExperimentConfig cfg;
  cfg.experiment_id = get_or<std::string>(d, "experiment_id", "interactive");
  cfg.output_dir = get_or<std::string>(d, "output_dir", "data/runs/interactive/records");
  cfg.mesh_path = get_or<std::string>(d, "mesh_path", "");
  cfg.pde = get_or<std::string>(d, "pde", "elliptic");
  cfg.seed = get_or<std::uint64_t>(d, "seed", 1);
  cfg.theta_values = get_float_vector_or(d, "theta_values", {0.25});
  cfg.epsilon_values = get_float_vector_or(d, "epsilon_values", {0.0});
  cfg.epsilon1_values = get_float_vector_or(d, "epsilon1_values", {});
  cfg.epsilon2_values = get_float_vector_or(d, "epsilon2_values", {});
  cfg.h_values = get_float_vector_or(d, "h_values", {0.125});
  if (d.contains("features")) {
    py::dict features = py::cast<py::dict>(d["features"]);
    cfg.features.m = get_or<int>(features, "m", cfg.features.m);
    cfg.features.op = get_or<std::string>(features, "op", cfg.features.op);
    cfg.features.normalize = get_or<std::string>(features, "normalize", cfg.features.normalize);
  }
  if (d.contains("solver")) {
    py::dict solver = py::cast<py::dict>(d["solver"]);
    cfg.solver.max_it = get_or<int>(solver, "max_it", cfg.solver.max_it);
    cfg.solver.rtol = get_or<double>(solver, "rtol", cfg.solver.rtol);
    cfg.solver.atol = get_or<double>(solver, "atol", cfg.solver.atol);
  }
  return cfg;
}

py::dict metrics_to_dict(const polyamg::AMGMetrics& metrics) {
  py::dict out;
  out["iterations"] = metrics.iterations;
  out["rho"] = metrics.rho;
  out["elapsed_sec"] = metrics.elapsed_sec;
  out["n_levels"] = metrics.n_levels;
  out["coarse_size"] = metrics.coarse_size;
  out["converged"] = metrics.converged;
  out["residual_norm"] = metrics.residual_norm;
  return out;
}

py::dict meta_to_dict(const polyamg::SampleMeta& meta) {
  py::dict out;
  out["sample_id"] = meta.sample_id;
  out["pde_type"] = meta.pde_type;
  out["mesh_id"] = meta.mesh_id;
  out["h"] = meta.h;
  out["theta"] = meta.theta;
  out["seed"] = meta.seed;
  out["epsilon"] = meta.epsilon.has_value() ? py::cast(*meta.epsilon) : py::none();
  out["epsilon1"] = meta.epsilon1.has_value() ? py::cast(*meta.epsilon1) : py::none();
  out["epsilon2"] = meta.epsilon2.has_value() ? py::cast(*meta.epsilon2) : py::none();
  out["polygonal_descriptors"] = meta.polygonal_descriptors;
  return out;
}

py::dict record_to_dict(const polyamg::SampleRecord& record) {
  py::dict out;
  out["sample_meta"] = meta_to_dict(record.meta);
  out["metrics"] = metrics_to_dict(record.metrics);
  out["schema_version"] = "1.0";
  return out;
}

}  // namespace

PYBIND11_MODULE(_polyamg_core, m) {
  m.doc() = "Interactive bindings for PolyAMG C++ core DTO workflows";
  m.def("run_experiment",
        [](const py::dict& config) {
          py::list out;
          for (const auto& record : polyamg::bindings::run_experiment_in_memory(config_from_dict(config))) {
            out.append(record_to_dict(record));
          }
          return out;
        },
        py::arg("config"));
}
