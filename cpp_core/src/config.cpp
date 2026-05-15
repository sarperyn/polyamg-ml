#include "polyamg/config.hpp"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace polyamg {

namespace {
std::string trim(const std::string& s) {
  const auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";
  const auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

std::vector<double> parse_vec(const std::string& v) {
  std::vector<double> out;
  std::stringstream ss(v);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    tok = trim(tok);
    if (!tok.empty()) out.push_back(std::stod(tok));
  }
  return out;
}

std::string to_lower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

DiffusionPattern parse_pattern(const std::string& raw) {
  const std::string v = to_lower(trim(raw));
  if (v == "a" || v == "vertical" || v == "vertical_split") return DiffusionPattern::VerticalSplit;
  if (v == "b" || v == "checker" || v == "checker2" || v == "checker2x2") return DiffusionPattern::Checker2x2;
  if (v == "c" || v == "stripes" || v == "vertical_stripes" || v == "vertical_stripes4") {
    return DiffusionPattern::VerticalStripes4;
  }
  if (v == "d" || v == "checker4" || v == "checker4x4") return DiffusionPattern::Checker4x4;
  if (v == "0") return DiffusionPattern::VerticalSplit;
  if (v == "1") return DiffusionPattern::Checker2x2;
  if (v == "2") return DiffusionPattern::VerticalStripes4;
  if (v == "3") return DiffusionPattern::Checker4x4;
  return DiffusionPattern::Checker2x2;
}
}  // namespace

ExperimentConfig load_experiment_config(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Cannot open config file: " + path);

  std::unordered_map<std::string, std::string> kv;
  std::string line;
  while (std::getline(in, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#') continue;
    const auto p = line.find('=');
    if (p == std::string::npos) continue;
    kv[trim(line.substr(0, p))] = trim(line.substr(p + 1));
  }

  ExperimentConfig cfg;
  if (kv.count("experiment_id")) cfg.experiment_id = kv["experiment_id"];
  if (kv.count("output_dir")) cfg.output_dir = kv["output_dir"];
  if (kv.count("mesh_path")) cfg.mesh_path = kv["mesh_path"];
  if (kv.count("pde")) cfg.pde = kv["pde"];
  if (kv.count("seed")) cfg.seed = static_cast<std::uint64_t>(std::stoull(kv["seed"]));
  if (kv.count("diffusion.pattern")) cfg.diffusion_pattern = parse_pattern(kv["diffusion.pattern"]);

  if (kv.count("theta_values")) cfg.theta_values = parse_vec(kv["theta_values"]);
  if (kv.count("epsilon_values")) cfg.epsilon_values = parse_vec(kv["epsilon_values"]);
  if (kv.count("epsilon1_values")) cfg.epsilon1_values = parse_vec(kv["epsilon1_values"]);
  if (kv.count("epsilon2_values")) cfg.epsilon2_values = parse_vec(kv["epsilon2_values"]);
  if (kv.count("h_values")) cfg.h_values = parse_vec(kv["h_values"]);
  if (cfg.theta_values.empty()) cfg.theta_values = {0.25};
  if (cfg.epsilon_values.empty() && (cfg.epsilon1_values.empty() || cfg.epsilon2_values.empty())) {
    cfg.epsilon_values = {0.0};
  }
  if (cfg.h_values.empty()) cfg.h_values = {0.125};

  if (kv.count("solver.max_it")) cfg.solver.max_it = std::stoi(kv["solver.max_it"]);
  if (kv.count("solver.rtol")) cfg.solver.rtol = std::stod(kv["solver.rtol"]);
  if (kv.count("solver.atol")) cfg.solver.atol = std::stod(kv["solver.atol"]);
  if (kv.count("solver.ksp_type")) cfg.solver.ksp_type = kv["solver.ksp_type"];
  if (kv.count("solver.pc_type")) cfg.solver.pc_type = kv["solver.pc_type"];
  if (kv.count("solver.hypre_type")) cfg.solver.hypre_type = kv["solver.hypre_type"];

  if (kv.count("features.m")) cfg.features.m = std::stoi(kv["features.m"]);
  if (kv.count("features.op")) cfg.features.op = kv["features.op"];
  if (kv.count("features.normalize")) cfg.features.normalize = kv["features.normalize"];

  if (kv.count("model.model_id")) {
    ModelManifest mm;
    mm.model_id = kv["model.model_id"];
    mm.onnx_path = kv.count("model.onnx_path") ? kv["model.onnx_path"] : "";
    mm.preprocessing_sha256 = kv.count("model.preprocessing_sha256") ? kv["model.preprocessing_sha256"] : "";
    mm.theta_grid = kv.count("model.theta_grid") ? parse_vec(kv["model.theta_grid"]) : std::vector<double>{};
    mm.features = cfg.features;
    cfg.model_manifest = mm;
  }

  return cfg;
}

}  // namespace polyamg
