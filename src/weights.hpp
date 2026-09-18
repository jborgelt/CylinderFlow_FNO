#pragma once
// weights: safetensors-compatible weight file IO for the FNO model.
// Layout: <u64 LE header len><JSON header><raw LE buffers>.
// dtypes F64/F32; complex R stored as R_real + R_imag (safetensors has no
// complex dtype). Zero new dependencies: minimal JSON writer + strict
// parser for exactly this schema. Readable later from Python via the
// safetensors package or plain numpy + json.
#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct WeightTensor {
  std::string name;
  std::vector<int64_t> shape;  // row-major
  std::vector<double> data;    // always double in memory
};

// Throws on IO errors, F32 overflow/NaN, or malformed input.
void saveWeights(const std::string &path,
                 const std::vector<WeightTensor> &tensors,
                 const std::map<std::string, std::string> &metadata,
                 bool fp32);
std::pair<std::vector<WeightTensor>, std::map<std::string, std::string>> loadWeights(const std::string &path);
