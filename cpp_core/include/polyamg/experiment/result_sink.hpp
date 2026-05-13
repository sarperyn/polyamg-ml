#pragma once

#include "polyamg/feature_extractor.hpp"
#include "polyamg/types.hpp"

#include <string>
#include <vector>

namespace polyamg {

struct SampleRecord {
  SampleMeta meta;
  AMGMetrics metrics;
  FeatureConfig features;
  FeatureTensor tensor;
  bool has_tensor = false;
};

class ResultSink {
 public:
  virtual ~ResultSink() = default;
  virtual void write(const SampleRecord& record) = 0;
};

class JsonResultSink final : public ResultSink {
 public:
  explicit JsonResultSink(std::string output_dir, std::string experiment_id);
  void write(const SampleRecord& record) override;

 private:
  std::string output_dir_;
  std::string experiment_id_;
};

class InMemoryResultSink final : public ResultSink {
 public:
  void write(const SampleRecord& record) override;
  const std::vector<SampleRecord>& records() const { return records_; }

 private:
  std::vector<SampleRecord> records_;
};

}  // namespace polyamg
