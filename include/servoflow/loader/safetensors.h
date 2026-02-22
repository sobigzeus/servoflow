// SPDX-License-Identifier: Apache-2.0
#pragma once

// SafeTensors format loader (CPU-side).
// Ref: https://github.com/huggingface/safetensors
//
// File layout:
//   [8 bytes]  header_len: uint64 little-endian
//   [header_len bytes] JSON header (UTF-8)
//   [data bytes] raw tensor data (concatenated, no padding between tensors)
//
// JSON header structure:
//   {
//     "__metadata__": {"format": "pt"},
//     "tensor_name": {
//       "dtype": "F32" | "F16" | "BF16" | "I8" | ...,
//       "shape": [d0, d1, ...],
//       "data_offsets": [begin, end]   // byte offsets into the data section
//     },
//     ...
//   }

#include "servoflow/core/tensor.h"
#include "servoflow/models/rdt1b/dit_block.h"  // for WeightMap
#include <string>
#include <unordered_map>
#include <vector>

namespace sf {

class SafeTensorsLoader {
public:
    // Load all tensors from a .safetensors file into CPU Tensors.
    // Returns a WeightMap (name → CPU Tensor backed by mmap or heap memory).
    static rdt1b::WeightMap load(const std::string& path);

    // Load only the tensors whose names start with any of the given prefixes.
    // Useful for loading a single shard of a sharded checkpoint efficiently.
    static rdt1b::WeightMap load_filtered(const std::string& path,
                                          const std::vector<std::string>& prefixes);

    // Load metadata (tensor names, shapes, dtypes) without reading tensor data.
    struct TensorMeta {
        std::string          name;
        DType                dtype;
        Shape                shape;
        uint64_t             data_begin;  // offset from start of data section
        uint64_t             data_end;
    };
    static std::vector<TensorMeta> inspect(const std::string& path);

private:
    struct Header {
        uint64_t                    header_len;
        std::vector<TensorMeta>     tensors;
        uint64_t                    data_section_offset;  // = 8 + header_len
    };

    static Header     parse_header(const char* file_data, size_t file_size);
    static DType      parse_dtype(const std::string& s);
    static Shape      parse_shape(const nlohmann::json& arr);
};

// ─────────────────────────────────────────────────────────────────────────────
// Helper used by model loaders: load a single weight from a pre-loaded map,
// cast to target dtype, and upload to the given backend device.
// ─────────────────────────────────────────────────────────────────────────────
Tensor load_weight_from_map(const rdt1b::WeightMap& weights,
                             const std::string& key,
                             DType target_dtype,
                             BackendPtr backend,
                             StreamHandle stream);

}  // namespace sf
