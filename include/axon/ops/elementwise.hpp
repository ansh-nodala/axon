#ifndef AXON_OPS_ELEMENTWISE_HPP
#define AXON_OPS_ELEMENTWISE_HPP

#include "axon/core/axon_macros.hpp"
#include "axon/core/types.hpp"
#include "axon/tensor/tensor.hpp"
#include <stddef.h>
#include <stdint.h>

namespace axon {
namespace ops {

// ============================================================================
// 1. LOW-LEVEL BARE-METAL SIMD KERNEL DECLARATIONS
// ============================================================================

namespace kernels {

// ----------------------------------------------------------------------------
// Float32 (Single-Precision IEEE 754) Compute Kernels
// ----------------------------------------------------------------------------

AXON_API void add_kernel_f32(
    const float* AXON_RESTRICT a,
    const float* AXON_RESTRICT b,
    float* AXON_RESTRICT out,
    size_t count
) noexcept;

AXON_API void sub_kernel_f32(
    const float* AXON_RESTRICT a,
    const float* AXON_RESTRICT b,
    float* AXON_RESTRICT out,
    size_t count
) noexcept;

AXON_API void mul_kernel_f32(
    const float* AXON_RESTRICT a,
    const float* AXON_RESTRICT b,
    float* AXON_RESTRICT out,
    size_t count
) noexcept;

AXON_API void div_kernel_f32(
    const float* AXON_RESTRICT a,
    const float* AXON_RESTRICT b,
    float* AXON_RESTRICT out,
    size_t count
) noexcept;

AXON_API void relu_kernel_f32(
    const float* AXON_RESTRICT in,
    float* AXON_RESTRICT out,
    size_t count
) noexcept;

AXON_API void fill_kernel_f32(
    float* AXON_RESTRICT out,
    float value,
    size_t count
) noexcept;

AXON_API void scale_kernel_f32(
    const float* AXON_RESTRICT in,
    float scalar,
    float* AXON_RESTRICT out,
    size_t count
) noexcept;

// ----------------------------------------------------------------------------
// Float64 (Double-Precision IEEE 754) Compute Kernels
// ----------------------------------------------------------------------------

AXON_API void add_kernel_f64(
    const double* AXON_RESTRICT a,
    const double* AXON_RESTRICT b,
    double* AXON_RESTRICT out,
    size_t count
) noexcept;

AXON_API void sub_kernel_f64(
    const double* AXON_RESTRICT a,
    const double* AXON_RESTRICT b,
    double* AXON_RESTRICT out,
    size_t count
) noexcept;

AXON_API void mul_kernel_f64(
    const double* AXON_RESTRICT a,
    const double* AXON_RESTRICT b,
    double* AXON_RESTRICT out,
    size_t count
) noexcept;

AXON_API void div_kernel_f64(
    const double* AXON_RESTRICT a,
    const double* AXON_RESTRICT b,
    double* AXON_RESTRICT out,
    size_t count
) noexcept;

AXON_API void relu_kernel_f64(
    const double* AXON_RESTRICT in,
    double* AXON_RESTRICT out,
    size_t count
) noexcept;

AXON_API void fill_kernel_f64(
    double* AXON_RESTRICT out,
    double value,
    size_t count
) noexcept;

AXON_API void scale_kernel_f64(
    const double* AXON_RESTRICT in,
    double scalar,
    double* AXON_RESTRICT out,
    size_t count
) noexcept;

// ----------------------------------------------------------------------------
// Int32 (32-Bit Signed Integer) Compute Kernels
// ----------------------------------------------------------------------------

AXON_API void add_kernel_i32(
    const int32_t* AXON_RESTRICT a,
    const int32_t* AXON_RESTRICT b,
    int32_t* AXON_RESTRICT out,
    size_t count
) noexcept;

AXON_API void sub_kernel_i32(
    const int32_t* AXON_RESTRICT a,
    const int32_t* AXON_RESTRICT b,
    int32_t* AXON_RESTRICT out,
    size_t count
) noexcept;

AXON_API void mul_kernel_i32(
    const int32_t* AXON_RESTRICT a,
    const int32_t* AXON_RESTRICT b,
    int32_t* AXON_RESTRICT out,
    size_t count
) noexcept;

AXON_API void div_kernel_i32(
    const int32_t* AXON_RESTRICT a,
    const int32_t* AXON_RESTRICT b,
    int32_t* AXON_RESTRICT out,
    size_t count
) noexcept;

AXON_API void relu_kernel_i32(
    const int32_t* AXON_RESTRICT in,
    int32_t* AXON_RESTRICT out,
    size_t count
) noexcept;

AXON_API void fill_kernel_i32(
    int32_t* AXON_RESTRICT out,
    int32_t value,
    size_t count
) noexcept;

AXON_API void scale_kernel_i32(
    const int32_t* AXON_RESTRICT in,
    int32_t scalar,
    int32_t* AXON_RESTRICT out,
    size_t count
) noexcept;

} // namespace kernels

// ============================================================================
// 2. FAST-PATH TENSOR CONTRACT VALIDATION HELPERS
// ============================================================================

namespace detail {

// Validates invariants for binary element-wise operations with an explicit output tensor
AXON_ALWAYS_INLINE void validate_binary_op(
    const axon::tensor::Tensor& a,
    const axon::tensor::Tensor& b,
    const axon::tensor::Tensor& out
) noexcept {
    AXON_ASSERT(!a.empty() && !b.empty() && !out.empty());
    AXON_ASSERT(a.dtype() == b.dtype() && a.dtype() == out.dtype());
    AXON_ASSERT(a.device() == b.device() && a.device() == out.device());
    AXON_ASSERT(a.shape().same_dims(b.shape()) && a.shape().same_dims(out.shape()));
    AXON_ASSERT(a.is_contiguous() && b.is_contiguous() && out.is_contiguous());
}

// Validates invariants for unary element-wise operations with an explicit output tensor
AXON_ALWAYS_INLINE void validate_unary_op(
    const axon::tensor::Tensor& in,
    const axon::tensor::Tensor& out
) noexcept {
    AXON_ASSERT(!in.empty() && !out.empty());
    AXON_ASSERT(in.dtype() == out.dtype());
    AXON_ASSERT(in.device() == out.device());
    AXON_ASSERT(in.shape().same_dims(out.shape()));
    AXON_ASSERT(in.is_contiguous() && out.is_contiguous());
}

// Validates invariants for binary in-place mutation operations
AXON_ALWAYS_INLINE void validate_inplace_binary_op(
    const axon::tensor::Tensor& self,
    const axon::tensor::Tensor& other
) noexcept {
    AXON_ASSERT(!self.empty() && !other.empty());
    AXON_ASSERT(self.dtype() == other.dtype());
    AXON_ASSERT(self.device() == other.device());
    AXON_ASSERT(self.shape().same_dims(other.shape()));
    AXON_ASSERT(self.is_contiguous() && other.is_contiguous());
}

// Validates invariants for unary in-place mutation operations
AXON_ALWAYS_INLINE void validate_inplace_unary_op(
    const axon::tensor::Tensor& self
) noexcept {
    AXON_ASSERT(!self.empty());
    AXON_ASSERT(self.is_contiguous());
}

} // namespace detail

// ============================================================================
// 3. EXPLICIT OUTPUT (ZERO-ALLOCATION) TENSOR OPERATIONS
// ============================================================================

// Computes out[i] = a[i] + b[i] without allocating memory
AXON_API void add_out(
    const axon::tensor::Tensor& a,
    const axon::tensor::Tensor& b,
    axon::tensor::Tensor& out
) noexcept;

// Computes out[i] = a[i] - b[i] without allocating memory
AXON_API void sub_out(
    const axon::tensor::Tensor& a,
    const axon::tensor::Tensor& b,
    axon::tensor::Tensor& out
) noexcept;

// Computes out[i] = a[i] * b[i] without allocating memory
AXON_API void mul_out(
    const axon::tensor::Tensor& a,
    const axon::tensor::Tensor& b,
    axon::tensor::Tensor& out
) noexcept;

// Computes out[i] = a[i] / b[i] without allocating memory
AXON_API void div_out(
    const axon::tensor::Tensor& a,
    const axon::tensor::Tensor& b,
    axon::tensor::Tensor& out
) noexcept;

// Computes out[i] = max(0, in[i]) without allocating memory
AXON_API void relu_out(
    const axon::tensor::Tensor& in,
    axon::tensor::Tensor& out
) noexcept;

// Computes out[i] = in[i] * scalar without allocating memory
AXON_API void scale_out(
    const axon::tensor::Tensor& in,
    float scalar,
    axon::tensor::Tensor& out
) noexcept;

// ============================================================================
// 4. FUNCTIONAL ALLOCATING TENSOR OPERATIONS
// ============================================================================

// Allocates a new contiguous Tensor and computes out = a + b
AXON_NODISCARD AXON_API axon::tensor::Tensor add(
    const axon::tensor::Tensor& a,
    const axon::tensor::Tensor& b
) noexcept;

// Allocates a new contiguous Tensor and computes out = a - b
AXON_NODISCARD AXON_API axon::tensor::Tensor sub(
    const axon::tensor::Tensor& a,
    const axon::tensor::Tensor& b
) noexcept;

// Allocates a new contiguous Tensor and computes out = a * b
AXON_NODISCARD AXON_API axon::tensor::Tensor mul(
    const axon::tensor::Tensor& a,
    const axon::tensor::Tensor& b
) noexcept;

// Allocates a new contiguous Tensor and computes out = a / b
AXON_NODISCARD AXON_API axon::tensor::Tensor div(
    const axon::tensor::Tensor& a,
    const axon::tensor::Tensor& b
) noexcept;

// Allocates a new contiguous Tensor and computes out = max(0, in)
AXON_NODISCARD AXON_API axon::tensor::Tensor relu(
    const axon::tensor::Tensor& in
) noexcept;

// Allocates a new contiguous Tensor and computes out = in * scalar
AXON_NODISCARD AXON_API axon::tensor::Tensor scale(
    const axon::tensor::Tensor& in,
    float scalar
) noexcept;

// ============================================================================
// 5. IN-PLACE TENSOR MUTATION OPERATIONS
// ============================================================================

// Mutates `self` in-place: self[i] += other[i]
AXON_API void add_(
    axon::tensor::Tensor& self,
    const axon::tensor::Tensor& other
) noexcept;

// Mutates `self` in-place: self[i] -= other[i]
AXON_API void sub_(
    axon::tensor::Tensor& self,
    const axon::tensor::Tensor& other
) noexcept;

// Mutates `self` in-place: self[i] *= other[i]
AXON_API void mul_(
    axon::tensor::Tensor& self,
    const axon::tensor::Tensor& other
) noexcept;

// Mutates `self` in-place: self[i] /= other[i]
AXON_API void div_(
    axon::tensor::Tensor& self,
    const axon::tensor::Tensor& other
) noexcept;

// Mutates `self` in-place: self[i] = max(0, self[i])
AXON_API void relu_(
    axon::tensor::Tensor& self
) noexcept;

// Mutates `self` in-place: self[i] *= scalar
AXON_API void scale_(
    axon::tensor::Tensor& self,
    float scalar
) noexcept;

// Mutates `self` in-place: self[i] = value
AXON_API void fill_(
    axon::tensor::Tensor& self,
    float value
) noexcept;

} // namespace ops
} // namespace axon

#endif // AXON_OPS_ELEMENTWISE_HPP
