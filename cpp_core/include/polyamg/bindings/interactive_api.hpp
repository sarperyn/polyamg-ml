#pragma once

#include "polyamg/config.hpp"
#include "polyamg/experiment/result_sink.hpp"

#include <vector>

namespace polyamg::bindings {

std::vector<SampleRecord> run_experiment_in_memory(const ExperimentConfig& cfg);

}  // namespace polyamg::bindings
