#ifndef AXON_TENSOR_TENSOR_HPP
#define AXON_TENSOR_TENSOR_HPP

#include "axon/tensor/storage.hpp"
#include "axon/tensor/shape.hpp"

namespace axon {
namespace tensor {

// ============================================================================
// CORE MULTI-DIMENSIONAL TENSOR ENGINE
// ============================================================================

class AXON_API Tensor {
private:
    Storage storage_;
    TensorShape shape_;
    int64_t storage_offset_;

public:
    // Default constructor: initializes an empty, unallocated tensor handle
    constexpr Tensor() noexcept
        : storage_(),
          shape_(),
          storage_offset_(0) {}

    // Contiguous owning constructor: computes C-order shape/strides and allocates SIMD-aligned storage
    Tensor(
        const int64_t* sizes,
        size_t ndim,
        axon::core::DType dtype = axon::core::DType::Float32,
        axon::core::DeviceType device = axon::core::DeviceType::CPU
    ) noexcept
        : storage_(),
          shape_(sizes, ndim),
          storage_offset_(0) {
        const size_t total = shape_.total_elements();
        if (AXON_LIKELY(total > 0)) {
            storage_ = Storage(total, dtype, device);
        }
    }

    // Zero-copy view constructor: binds an existing Storage handle, custom shape/strides, and element offset
    Tensor(
        Storage&& storage,
        const TensorShape& shape,
        int64_t storage_offset = 0
    ) noexcept
        : storage_(static_cast<Storage&&>(storage)),
          shape_(shape),
          storage_offset_(storage_offset) {
        AXON_ASSERT(storage_offset_ >= 0);
    }

    // Destructor: deterministically releases underlying Storage if owned
    ~Tensor() noexcept = default;

    // Strictly disable implicit deep copies (use explicit .clone() for deep copies)
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Move constructor: transfers Tensor ownership in O(1) without memory allocation
    Tensor(Tensor&& other) noexcept
        : storage_(static_cast<Storage&&>(other.storage_)),
          shape_(other.shape_),
          storage_offset_(other.storage_offset_) {
        other.shape_ = TensorShape();
        other.storage_offset_ = 0;
    }

    // Move assignment operator: releases current owned storage and claims source Tensor
    Tensor& operator=(Tensor&& other) noexcept {
        if (AXON_LIKELY(this != &other)) {
            storage_ = static_cast<Storage&&>(other.storage_);
            shape_ = other.shape_;
            storage_offset_ = other.storage_offset_;

            other.shape_ = TensorShape();
            other.storage_offset_ = 0;
        }
        return *this;
    }

    // ========================================================================
    // CORE METADATA & TOPOLOGY GETTERS
    // ========================================================================

    AXON_NODISCARD constexpr AXON_ALWAYS_INLINE const TensorShape& shape() const noexcept {
        return shape_;
    }

    AXON_NODISCARD AXON_ALWAYS_INLINE const Storage& storage() const noexcept {
        return storage_;
    }

    AXON_NODISCARD AXON_ALWAYS_INLINE Storage& storage() noexcept {
        return storage_;
    }

    AXON_NODISCARD constexpr AXON_ALWAYS_INLINE int64_t storage_offset() const noexcept {
        return storage_offset_;
    }

    AXON_NODISCARD constexpr AXON_ALWAYS_INLINE size_t ndim() const noexcept {
        return shape_.ndim();
    }

    AXON_NODISCARD constexpr AXON_ALWAYS_INLINE int64_t dim(size_t index) const noexcept {
        return shape_.dim(index);
    }

    AXON_NODISCARD constexpr AXON_ALWAYS_INLINE int64_t stride(size_t index) const noexcept {
        return shape_.stride(index);
    }

    AXON_NODISCARD constexpr AXON_ALWAYS_INLINE const int64_t* dims() const noexcept {
        return shape_.dims();
    }

    AXON_NODISCARD constexpr AXON_ALWAYS_INLINE const int64_t* strides() const noexcept {
        return shape_.strides();
    }

    AXON_NODISCARD AXON_ALWAYS_INLINE axon::core::DType dtype() const noexcept {
        return storage_.dtype();
    }

    AXON_NODISCARD AXON_ALWAYS_INLINE axon::core::DeviceType device() const noexcept {
        return storage_.device();
    }

    AXON_NODISCARD AXON_ALWAYS_INLINE size_t num_elements() const noexcept {
        return storage_.empty() ? 0 : shape_.total_elements();
    }

    AXON_NODISCARD AXON_ALWAYS_INLINE size_t itemsize() const noexcept {
        return storage_.itemsize();
    }

    AXON_NODISCARD AXON_ALWAYS_INLINE size_t nbytes() const noexcept {
        return num_elements() * itemsize();
    }

    AXON_NODISCARD AXON_ALWAYS_INLINE bool empty() const noexcept {
        return storage_.empty();
    }

    AXON_NODISCARD constexpr AXON_ALWAYS_INLINE bool is_contiguous() const noexcept {
        return shape_.is_contiguous();
    }

    AXON_NODISCARD AXON_ALWAYS_INLINE bool owns_memory() const noexcept {
        return storage_.owns_memory();
    }

    // ========================================================================
    // TYPE-SAFE MEMORY ACCESSORS
    // ========================================================================

    // Returns a typed pointer offset by `storage_offset_` with compile-time DType validation
    template <typename T>
    AXON_NODISCARD AXON_ALWAYS_INLINE T* data() noexcept {
        T* base_ptr = storage_.data_as<T>();
        return (base_ptr != nullptr) ? (base_ptr + storage_offset_) : nullptr;
    }

    template <typename T>
    AXON_NODISCARD AXON_ALWAYS_INLINE const T* data() const noexcept {
        const T* base_ptr = storage_.data_as<T>();
        return (base_ptr != nullptr) ? (base_ptr + storage_offset_) : nullptr;
    }

    // Returns an untyped byte-offset pointer to the start of this Tensor's active view
    AXON_NODISCARD AXON_ALWAYS_INLINE void* raw_data() noexcept {
        uint8_t* base_bytes = static_cast<uint8_t*>(storage_.data());
        return (base_bytes != nullptr)
            ? static_cast<void*>(base_bytes + (static_cast<size_t>(storage_offset_) * itemsize()))
            : nullptr;
    }

    AXON_NODISCARD AXON_ALWAYS_INLINE const void* raw_data() const noexcept {
        const uint8_t* base_bytes = static_cast<const uint8_t*>(storage_.data());
        return (base_bytes != nullptr)
            ? static_cast<const void*>(base_bytes + (static_cast<size_t>(storage_offset_) * itemsize()))
            : nullptr;
    }

    // Verifies whether the active view pointer (accounting for storage_offset_) is 64-byte SIMD-aligned
    AXON_NODISCARD AXON_ALWAYS_INLINE bool is_simd_aligned() const noexcept {
        return axon::memory::is_aligned(raw_data(), AXON_SIMD_ALIGNMENT);
    }

    // Zero-initializes the tensor's memory buffer (requires contiguous layout)
    AXON_ALWAYS_INLINE void zero() noexcept {
        AXON_ASSERT(is_contiguous());
        void* ptr = raw_data();
        const size_t bytes = nbytes();
        if (AXON_LIKELY(ptr != nullptr && bytes > 0)) {
            if (is_simd_aligned()) {
                axon::memory::zero_memory_aligned(ptr, bytes);
            } else {
                axon::memory::zero_memory(ptr, bytes);
            }
        }
    }

    // ========================================================================
    // TENSOR VIEW & TRANSFORMATION OPERATIONS
    // ========================================================================

    // Performs a deep copy of the Tensor into a newly allocated contiguous owning Tensor
    AXON_NODISCARD Tensor clone() const;

    // Returns a zero-copy strided Tensor view with `dim0` and `dim1` swapped
    AXON_NODISCARD Tensor transpose(size_t dim0 = 0, size_t dim1 = 1) const;

    // Returns a zero-copy reshaped Tensor view (requires contiguous memory layout and matching element count)
    AXON_NODISCARD Tensor view(const int64_t* sizes, size_t ndim) const;
};

} // namespace tensor
} // namespace axon

#endif // AXON_TENSOR_TENSOR_HPP
