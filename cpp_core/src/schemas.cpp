#include "polyamg/schemas.hpp"

#include <fstream>
#include <sstream>

namespace polyamg {

namespace {
std::string json_num_or_null(const std::optional<double>& v) { return v.has_value() ? std::to_string(*v) : "null"; }
std::string json_int_or_null(const std::optional<int>& v) { return v.has_value() ? std::to_string(*v) : "null"; }
}  // namespace

void write_sample_record_json(const std::string& path, const SampleMeta& meta, const AMGMetrics& metrics,
                              const FeatureConfig& features, const FeatureTensor* tensor) {
  std::ostringstream j;
  j << "{\n";
  j << "  \"schema_version\": \"1.0\",\n";
  j << "  \"sample_meta\": {\n";
  j << "    \"sample_id\": \"" << meta.sample_id << "\",\n";
  j << "    \"pde_type\": \"" << meta.pde_type << "\",\n";
  j << "    \"mesh_id\": \"" << meta.mesh_id << "\",\n";
  j << "    \"h\": " << meta.h << ",\n";
  j << "    \"theta\": " << meta.theta << ",\n";
  j << "    \"seed\": " << meta.seed << ",\n";
  j << "    \"epsilon\": " << json_num_or_null(meta.epsilon) << ",\n";
  j << "    \"epsilon1\": " << json_num_or_null(meta.epsilon1) << ",\n";
  j << "    \"epsilon2\": " << json_num_or_null(meta.epsilon2) << ",\n";
  j << "    \"viscosity\": " << json_num_or_null(meta.viscosity) << ",\n";
  j << "    \"inflow_u\": " << json_num_or_null(meta.inflow_u) << ",\n";
  j << "    \"sequence_step\": " << json_int_or_null(meta.sequence_step) << ",\n";
  j << "    \"polygonal_descriptors\": {";
  bool first = true;
  for (const auto& kv : meta.polygonal_descriptors) {
    if (!first) j << ",";
    j << "\n      \"" << kv.first << "\": " << kv.second;
    first = false;
  }
  if (!meta.polygonal_descriptors.empty()) j << "\n    ";
  j << "}\n";
  j << "  },\n";
  j << "  \"feature_config\": {\n";
  j << "    \"m\": " << features.m << ",\n";
  j << "    \"op\": \"" << features.op << "\",\n";
  j << "    \"normalize\": \"" << features.normalize << "\"\n";
  j << "  },\n";
  if (tensor) {
    j << "  \"feature_tensor\": {\n";
    j << "    \"m\": " << tensor->m << ",\n";
    j << "    \"c\": " << tensor->c << ",\n";
    j << "    \"values\": [";
    for (size_t i = 0; i < tensor->values.size(); ++i) {
      if (i) j << ", ";
      j << tensor->values[i];
    }
    j << "]\n";
    j << "  },\n";
  }
  j << "  \"metrics\": {\n";
  j << "    \"iterations\": " << metrics.iterations << ",\n";
  j << "    \"rho\": " << metrics.rho << ",\n";
  j << "    \"elapsed_sec\": " << metrics.elapsed_sec << ",\n";
  j << "    \"n_levels\": " << metrics.n_levels << ",\n";
  j << "    \"coarse_size\": " << metrics.coarse_size << ",\n";
  j << "    \"converged\": " << (metrics.converged ? "true" : "false") << ",\n";
  j << "    \"residual_norm\": " << metrics.residual_norm << "\n";
  j << "  }\n";
  j << "}\n";
  std::ofstream out(path);
  out << j.str();
}

void write_model_manifest_json(const std::string& path, const ModelManifest& manifest) {
  std::ostringstream j;
  j << "{\n";
  j << "  \"schema_version\": \"1.0\",\n";
  j << "  \"model_id\": \"" << manifest.model_id << "\",\n";
  j << "  \"onnx_path\": \"" << manifest.onnx_path << "\",\n";
  j << "  \"features\": {\"m\": " << manifest.features.m << ", \"op\": \"" << manifest.features.op
    << "\", \"normalize\": \"" << manifest.features.normalize << "\"},\n";
  j << "  \"theta_grid\": [";
  for (size_t i = 0; i < manifest.theta_grid.size(); ++i) {
    if (i) j << ", ";
    j << manifest.theta_grid[i];
  }
  j << "],\n";
  j << "  \"preprocessing_sha256\": \"" << manifest.preprocessing_sha256 << "\"\n";
  j << "}\n";
  std::ofstream out(path);
  out << j.str();
}

}  // namespace polyamg
