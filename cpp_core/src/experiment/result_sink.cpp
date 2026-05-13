#include "polyamg/experiment/result_sink.hpp"

#include "polyamg/schemas.hpp"

#include <sstream>
#include <utility>

namespace polyamg {

JsonResultSink::JsonResultSink(std::string output_dir, std::string experiment_id)
    : output_dir_(std::move(output_dir)), experiment_id_(std::move(experiment_id)) {}

void JsonResultSink::write(const SampleRecord& record) {
  std::ostringstream path;
  path << output_dir_ << "/" << experiment_id_ << "_" << record.meta.sample_id << ".json";
  write_sample_record_json(path.str(), record.meta, record.metrics, record.features,
                           record.has_tensor ? &record.tensor : nullptr);
}

void InMemoryResultSink::write(const SampleRecord& record) { records_.push_back(record); }

}  // namespace polyamg
