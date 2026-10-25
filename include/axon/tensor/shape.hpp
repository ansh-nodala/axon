#ifndef AXON_TENSOR_SHAPE_HPP
#define AXON_TENSOR_SHAPE_HPP

#include "axon/core/axon_macros.hpp"
#include <stddef.h>
#include <stdint.h>

namespace axon {
namespace tensor {

// ============================================================================
// MAXIMUM TENSOR DIMENSIONALITY CONSTANT
// ============================================================================

// Fixed upper bound on tensor rank to guarantee stack-allocated, cache-resident
// metadata without dynamic heap allocations.
constexpr size_t AXON_MAX_DIMS = 8;

// ============================================================================
// TENSOR SHAPE & STRIDE METADATA ENGINE
// ============================================================================

class AXON_API TensorShape {
private:
    int64_t dims_[AXON_MAX_DIMS];
    int64_t strides_[AXON_MAX_DIMS];
    size_t ndim_;

public:
    // Default constructor: initializes a 0-D scalar tensor shape (ndim = 0, 1 total element)
    constexpr TensorShape() noexcept
        : dims_{0, 0, 0, 0, 0, 0, 0, 0},
          strides_{0, 0, 0, 0, 0, 0, 0, 0},
          ndim_(0) {}

    // Contiguous array constructor: copies dimensions and computes C-order (row-major) strides
    constexpr TensorShape(const int64_t* sizes, size_t ndim) noexcept
        : dims_{0, 0, 0, 0, 0, 0, 0, 0},
          strides_{0, 0, 0, 0, 0, 0, 0, 0},
          ndim_(ndim) {
        AXON_ASSERT(ndim <= AXON_MAX_DIMS);

        if (AXON_LIKELY(ndim_ > 0)) {
            AXON_ASSERT(sizes != nullptr);

            for (size_t i = 0; i < ndim_; ++i) {
                AXON_ASSERT(sizes[i] >= 0);
                dims_[i] = sizes[i];
            }

            int64_t running_stride = 1;
            for (size_t i = ndim_; i > 0; --i) {
                const size_t idx = i - 1;
                strides_[idx] = running_stride;
                running_stride *= dims_[idx];
            }
        }
    }

    // Explicit strided constructor: initializes both dimensions and custom strides for views/slices
    constexpr TensorShape(const int64_t* sizes, const int64_t* strides, size_t ndim) noexcept
        : dims_{0, 0, 0, 0, 0, 0, 0, 0},
          strides_{0, 0, 0, 0, 0, 0, 0, 0},
          ndim_(ndim) {
        AXON_ASSERT(ndim <= AXON_MAX_DIMS);

        if (AXON_LIKELY(ndim_ > 0)) {
            AXON_ASSERT(sizes != nullptr);
            AXON_ASSERT(strides != nullptr);

            for (size_t i = 0; i < ndim_; ++i) {
                AXON_ASSERT(sizes[i] >= 0);
                dims_[i] = sizes[i];
                strides_[i] = strides[i];
            }
        }
    }

    // ========================================================================
    // ZERO-OVERHEAD DIMENSION & STRIDE ACCESSORS
    // ========================================================================

    AXON_NODISCARD constexpr AXON_ALWAYS_INLINE size_t ndim() const noexcept {
        return ndim_;
    }

    AXON_NODISCARD constexpr AXON_ALWAYS_INLINE int64_t dim(size_t index) const noexcept {
        AXON_ASSERT(index < ndim_);
        return dims_[index];
    }

    AXON_NODISCARD constexpr AXON_ALWAYS_INLINE int64_t stride(size_t index) const noexcept {
        AXON_ASSERT(index < ndim_);
        return strides_[index];
    }

    AXON_NODISCARD constexpr AXON_ALWAYS_INLINE const int64_t* dims() const noexcept {
        return dims_;
    }

    AXON_NODISCARD constexpr AXON_ALWAYS_INLINE const int64_t* strides() const noexcept {
        return strides_;
    }

    // ========================================================================
    // LAYOUT & TOPOLOGY INTROSPECTION
    // ========================================================================

    // Computes total scalar elements across all dimensions (returns 1 for a 0-D scalar)
    AXON_NODISCARD constexpr AXON_ALWAYS_INLINE size_t total_elements() const noexcept {
        size_t total = 1;
        for (size_t i = 0; i < ndim_; ++i) {
            total *= static_cast<size_t>(dims_[i]);
        }
        return total;
    }

    // Verifies whether strides strictly match a standard C-order (row-major) contiguous memory layout
    AXON_NODISCARD constexpr AXON_ALWAYS_INLINE bool is_contiguous() const noexcept {
        int64_t expected_stride = 1;
        for (size_t i = ndim_; i > 0; --i) {
            const size_t idx = i - 1;
            if (strides_[idx] != expected_stride) {
                return false;
            }
            expected_stride *= dims_[idx];
        }
        return true;
    }

    // Evaluates to true if this shape represents a 0-D scalar value
    AXON_NODISCARD constexpr AXON_ALWAYS_INLINE bool is_scalar() const noexcept {
        return ndim_ == 0;
    }

    // Computes the flat element offset from multi-dimensional coordinates using current strides
    AXON_NODISCARD constexpr AXON_ALWAYS_INLINE int64_t linear_offset(
        const int64_t* AXON_RESTRICT indices,
        size_t num_indices
    ) const noexcept {
        AXON_ASSERT(num_indices == ndim_);
        int64_t offset = 0;
        for (size_t i = 0; i < ndim_; ++i) {
            AXON_ASSERT(indices[i] >= 0 && indices[i] < dims_[i]);
            offset += indices[i] * strides_[i];
        }
        return offset;
    }

    // Returns a new TensorShape with two dimensions and their corresponding strides swapped
    AXON_NODISCARD constexpr AXON_ALWAYS_INLINE TensorShape transpose(
        size_t dim0 = 0,
        size_t dim1 = 1
    ) const noexcept {
        AXON_ASSERT(dim0 < ndim_ && dim1 < ndim_);
        TensorShape result = *this;
        const int64_t tmp_dim = result.dims_[dim0];
        result.dims_[dim0] = result.dims_[dim1];
        result.dims_[dim1] = tmp_dim;

        const int64_t tmp_stride = result.strides_[dim0];
        result.strides_[dim0] = result.strides_[dim1];
        result.strides_[dim1] = tmp_stride;
        return result;
    }

    // Compares whether two shapes have identical rank and dimensions (ignoring strides)
    AXON_NODISCARD constexpr AXON_ALWAYS_INLINE bool same_dims(const TensorShape& other) const noexcept {
        if (ndim_ != other.ndim_) {
            return false;
        }
        for (size_t i = 0; i < ndim_; ++i) {
            if (dims_[i] != other.dims_[i]) {
                return false;
            }
        }
        return true;
    }

    // Exact equality comparison (both dimensions and strides must match)
    AXON_NODISCARD constexpr AXON_ALWAYS_INLINE bool operator==(const TensorShape& other) const noexcept {
        if (ndim_ != other.ndim_) {
            return false;
        }
        for (size_t i = 0; i < ndim_; ++i) {
            if (dims_[i] != other.dims_[i] || strides_[i] != other.strides_[i]) {
                return false;
            }
        }
        return true;
    }

    AXON_NODISCARD constexpr AXON_ALWAYS_INLINE bool operator!=(const TensorShape& other) const noexcept {
        return !(*this == other);
    }
};

} // namespace tensor
} // namespace axon

#endif // AXON_TENSOR_SHAPE_HPP
