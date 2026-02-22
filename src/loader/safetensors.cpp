// SPDX-License-Identifier: Apache-2.0
#include "servoflow/loader/safetensors.h"

#include <nlohmann/json.hpp>

#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace sf {

// ─────────────────────────────────────────────────────────────────────────────
// RAII wrapper for memory-mapped files (POSIX mmap).
// Using mmap avoids copying the entire file into heap memory.
// ─────────────────────────────────────────────────────────────────────────────
class MmapFile {
public:
    explicit MmapFile(const std::string& path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0)
            throw std::runtime_error("SafeTensors: cannot open " + path);

        struct stat st{};
        if (::fstat(fd_, &st) != 0)
            throw std::runtime_error("SafeTensors: fstat failed for " + path);
        size_ = static_cast<size_t>(st.st_size);

        data_ = static_cast<const char*>(
            ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (data_ == MAP_FAILED) {
            ::close(fd_);
            throw std::runtime_error("SafeTensors: mmap failed for " + path);
        }
        // Hint to the OS that we'll access this sequentially.
        ::madvise(const_cast<char*>(data_), size_, MADV_SEQUENTIAL);
    }

    ~MmapFile() {
        if (data_ && data_ != MAP_FAILED) ::munmap(const_cast<char*>(data_), size_);
        if (fd_ >= 0) ::close(fd_);
    }

    const char* data() const { return data_; }
    size_t      size() const { return size_; }

    MmapFile(const MmapFile&)            = delete;
    MmapFile& operator=(const MmapFile&) = delete;

private:
    int         fd_   = -1;
    const char* data_ = nullptr;
    size_t      size_ = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// Header parsing
// ─────────────────────────────────────────────────────────────────────────────
DType SafeTensorsLoader::parse_dtype(const std::string& s) {
    DType dt = dtype_from_string(s);
    if (dt == DType::Unknown)
        throw std::runtime_error("SafeTensors: unrecognised dtype '" + s + "'");
    return dt;
}

Shape SafeTensorsLoader::parse_shape(const nlohmann::json& arr) {
    std::vector<int64_t> dims;
    dims.reserve(arr.size());
    for (auto& d : arr) dims.push_back(d.get<int64_t>());
    return Shape(dims);
}

SafeTensorsLoader::Header
SafeTensorsLoader::parse_header(const char* file_data, size_t file_size) {
    if (file_size < 8)
        throw std::runtime_error("SafeTensors: file too small");

    uint64_t header_len = 0;
    std::memcpy(&header_len, file_data, 8);

    if (8 + header_len > file_size)
        throw std::runtime_error("SafeTensors: header_len exceeds file size");

    std::string_view json_sv(file_data + 8, static_cast<size_t>(header_len));
    auto j = nlohmann::json::parse(json_sv);

    Header hdr;
    hdr.header_len          = header_len;
    hdr.data_section_offset = 8 + header_len;

    for (auto& [name, info] : j.items()) {
        if (name == "__metadata__") continue;

        TensorMeta meta;
        meta.name       = name;
        meta.dtype      = parse_dtype(info["dtype"].get<std::string>());
        meta.shape      = parse_shape(info["shape"]);
        auto offsets    = info["data_offsets"];
        meta.data_begin = offsets[0].get<uint64_t>();
        meta.data_end   = offsets[1].get<uint64_t>();
        hdr.tensors.push_back(std::move(meta));
    }
    return hdr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────
std::vector<SafeTensorsLoader::TensorMeta>
SafeTensorsLoader::inspect(const std::string& path) {
    MmapFile f(path);
    auto hdr = parse_header(f.data(), f.size());
    return hdr.tensors;
}

rdt1b::WeightMap SafeTensorsLoader::load(const std::string& path) {
    return load_filtered(path, {});
}

rdt1b::WeightMap SafeTensorsLoader::load_filtered(
    const std::string& path, const std::vector<std::string>& prefixes) {

    MmapFile f(path);
    auto hdr = parse_header(f.data(), f.size());

    rdt1b::WeightMap result;
    result.reserve(hdr.tensors.size());

    for (auto& meta : hdr.tensors) {
        // Apply prefix filter.
        if (!prefixes.empty()) {
            bool match = false;
            for (auto& p : prefixes)
                if (meta.name.substr(0, p.size()) == p) { match = true; break; }
            if (!match) continue;
        }

        uint64_t byte_len = meta.data_end - meta.data_begin;
        uint64_t file_off = hdr.data_section_offset + meta.data_begin;

        if (file_off + byte_len > f.size())
            throw std::runtime_error(
                "SafeTensors: data out of bounds for tensor '" + meta.name + "'");

        // Create a CPU tensor that owns a copy of the data (heap allocation).
        // We copy rather than reference mmap memory so that the file can be
        // closed and the tensor remains valid independently.
        void* buf = std::malloc(static_cast<size_t>(byte_len));
        if (!buf) throw std::bad_alloc();
        std::memcpy(buf, f.data() + file_off, static_cast<size_t>(byte_len));

        auto storage = std::make_shared<Storage>(
            buf, static_cast<size_t>(byte_len), kCPU,
            [](void* p) { std::free(p); });

        result[meta.name] = Tensor(std::move(storage), meta.shape, meta.dtype);
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// load_weight_from_map: CPU → GPU with dtype cast
// ─────────────────────────────────────────────────────────────────────────────
Tensor load_weight_from_map(const rdt1b::WeightMap& weights,
                             const std::string& key,
                             DType target_dtype,
                             BackendPtr backend,
                             StreamHandle stream) {
    auto it = weights.find(key);
    if (it == weights.end())
        throw std::runtime_error("load_weight_from_map: missing key '" + key + "'");

    const Tensor& src = it->second;

    if (src.dtype() == target_dtype) {
        // Direct H2D upload.
        Tensor dst = backend->alloc(src.shape(), target_dtype, stream);
        backend->copy(dst, src, stream);
        return dst;
    }

    // Upload in source dtype, then cast on device.
    Tensor tmp = backend->alloc(src.shape(), src.dtype(), stream);
    backend->copy(tmp, src, stream);
    Tensor dst = backend->alloc(src.shape(), target_dtype, stream);
    backend->cast(tmp, dst, stream);
    return dst;
}

}  // namespace sf
