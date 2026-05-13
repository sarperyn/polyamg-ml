#include "polyamg/amg_policy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef POLYAMG_WITH_ORT
#include <onnxruntime_cxx_api.h>
#endif

namespace polyamg {

namespace {

#ifdef POLYAMG_WITH_ORT
class OnnxRuntimePolicy final : public AMGPolicy {
 public:
  OnnxRuntimePolicy() : env_(ORT_LOGGING_LEVEL_WARNING, "polyamg") {}

  double select_theta(const FeatureTensor& tensor, double h, const ModelManifest& manifest) override {
    if (manifest.onnx_path.empty()) throw std::runtime_error("Model ONNX path is empty in manifest");
    if (manifest.theta_grid.empty()) throw std::runtime_error("theta_grid is empty in manifest");

    ensure_session(manifest.onnx_path);

    double best_theta = manifest.theta_grid.front();
    float best_rho = std::numeric_limits<float>::infinity();

    const int64_t m = static_cast<int64_t>(tensor.m);
    const std::array<int64_t, 4> img_shape{1, tensor.c, m, m};
    const std::array<int64_t, 1> scalar_shape{1};

    const float h_feat = static_cast<float>(-std::log2(std::max(h, 1e-12)));

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    for (double theta : manifest.theta_grid) {
      std::vector<float> img(tensor.values.size());
      for (std::size_t i = 0; i < tensor.values.size(); ++i) img[i] = static_cast<float>(tensor.values[i]);
      std::vector<float> hvec{h_feat};
      std::vector<float> tvec{static_cast<float>(theta)};

      Ort::Value x_img = Ort::Value::CreateTensor<float>(mem_info, img.data(), img.size(), img_shape.data(),
                                                         img_shape.size());
      Ort::Value x_h =
          Ort::Value::CreateTensor<float>(mem_info, hvec.data(), hvec.size(), scalar_shape.data(), scalar_shape.size());
      Ort::Value x_t =
          Ort::Value::CreateTensor<float>(mem_info, tvec.data(), tvec.size(), scalar_shape.data(), scalar_shape.size());

      std::array<const char*, 3> input_names{"x_img", "x_h", "x_theta"};
      std::array<const char*, 1> output_names{"rho_pred"};
      std::array<Ort::Value, 3> inputs{std::move(x_img), std::move(x_h), std::move(x_t)};

      auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(),
                                   output_names.data(), output_names.size());
      if (outputs.empty()) continue;

      auto& out = outputs.front();
      const float* pred = out.GetTensorData<float>();
      const float rho = pred ? pred[0] : std::numeric_limits<float>::infinity();

      if (rho < best_rho) {
        best_rho = rho;
        best_theta = theta;
      }
    }

    return best_theta;
  }

 private:
  void ensure_session(const std::string& onnx_path) {
    if (session_ && loaded_path_ == onnx_path) return;

    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    session_ = std::make_unique<Ort::Session>(env_, onnx_path.c_str(), opts);
    loaded_path_ = onnx_path;
  }

  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  std::string loaded_path_;
};
#endif

}  // namespace

std::unique_ptr<AMGPolicy> create_ann_policy() {
#ifdef POLYAMG_WITH_ORT
  return std::make_unique<OnnxRuntimePolicy>();
#else
  return std::make_unique<GridSearchPolicy>();
#endif
}

}  // namespace polyamg
