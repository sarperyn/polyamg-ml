#pragma once

#include "polyamg/feature_extractor.hpp"
#include "polyamg/types.hpp"

#include <string>

namespace polyamg {

void write_sample_record_json(const std::string& path, const SampleMeta& meta, const AMGMetrics& metrics,
                              const FeatureConfig& features, const FeatureTensor* tensor = nullptr);

void write_model_manifest_json(const std::string& path, const ModelManifest& manifest);

}  // namespace polyamg
