#ifndef AXON_OPS_REDUCTIONS_HPP
#define AXON_OPS_REDUCTIONS_HPP

#include "axon/core/axon_macros.hpp"
#include "axon/core/types.hpp"
#include "axon/tensor/tensor.hpp"
#include <stddef.h>
#include <stdint.h>

namespace axon {
namespace ops {
namespace kernels {

// ============================================================================
// 1. LOW-LEVEL BARE-METAL SIMD KERNEL DECLARATIONS
// ============================================================================

// Scalar Reductions (Float32, Float64, Int32)
AXON_NODISCARD AXON_API float sum_kernel_f32(const float* AXON_RESTRICT in, size_t count) noexcept;
AXON_NODISCARD AXON_API double sum_kernel_f64(const double* AXON_RESTRICT in, size_t count) noexcept;
AXON_NODISCARD AXON_API int32_t sum_kernel_i32(const int32_t* AXON_RESTRICT in, size_t count) noexcept;

AXON_NODISCARD AXON_API float max_kernel_f32(const float* AXON_RESTRICT in, size_t count) noexcept;
AXON_NODISCARD AXON_API double max_kernel_f64(const double* AXON_RESTRICT in, size_t count) noexcept;
AXON_NODISCARD AXON_API int32_t max_kernel_i32(const int32_t* AXON_RESTRICT in, size_t count) noexcept;

AXON_NODISCARD AXON_API float min_kernel_f32(const float* AXON_RESTRICT in, size_t count) noexcept;
AXON_NODISCARD AXON_API double min_kernel_f64(const double* AXON_RESTRICT in, size_t count) noexcept;
AXON_NODISCARD AXON_API int32_t min_kernel_i32(const int32_t* AXON_RESTRICT in, size_t count) noexcept;

AXON_NODISCARD AXON_API size_t argmax_kernel_f32(const float* AXON_RESTRICT in, size_t count) noexcept;
AXON_NODISCARD AXON_API size_t argmax_kernel_f64(const double* AXON_RESTRICT in, size_t count) noexcept;
AXON_NODISCARD AXON_API size_t argmax_kernel_i32(const int32_t* AXON_RESTRICT in, size_t count) noexcept;

// Deep Learning Transcendental & Non-Linear Activation Kernels (Float32)
AXON_API void exp_kernel_f32(const float* AXON_RESTRICT in, float* AXON_RESTRICT out, size_t count) noexcept;
AXON_API void sigmoid_kernel_f32(const float* AXON_RESTRICT in, float* AXON_RESTRICT out, size_t count) noexcept;
AXON_API void tanh_kernel_f32(const float* AXON_RESTRICT in, float* AXON_RESTRICT out, size_t count) noexcept;
AXON_API void silu_kernel_f32(const float* AXON_RESTRICT in, float* AXON_RESTRICT out, size_t count) noexcept;
AXON_API void gelu_kernel_f32(const float* AXON_RESTRICT in, float* AXON_RESTRICT out, size_t count) noexcept;

// Fused Row-Wise Normalization & Probability Kernels (Float32 over [outer_rows, row_size])
AXON_API void softmax_row_kernel_f32(const float* AXON_RESTRICT in, float* AXON_RESTRICT out, size_t outer_rows, size_t row_size) noexcept;
AXON_API void rmsnorm_row_kernel_f32(const float* AXON_RESTRICT in, const float* AXON_RESTRICT weight, float* AXON_RESTRICT out, size_t outer_rows, size_t row_size, float eps) noexcept;

} // namespace kernels

// ============================================================================
// 2. HIGH-LEVEL TENSOR API DECLARATIONS (NAMESPACE AXON::OPS)
// ============================================================================

// Full-Tensor Scalar Reductions (Float32 convenience)
AXON_NODISCARD AXON_API float sum_all(const axon::tensor::Tensor& in) noexcept;
AXON_NODISCARD AXON_API float mean_all(const axon::tensor::Tensor& in) noexcept;
AXON_NODISCARD AXON_API float max_all(const axon::tensor::Tensor& in) noexcept;
AXON_NODISCARD AXON_API float min_all(const axon::tensor::Tensor& in) noexcept;
AXON_NODISCARD AXON_API int64_t argmax_all(const axon::tensor::Tensor& in) noexcept;

// Last-Dimension Axis Reductions (Reduces last axis [..., D] -> [...] or [1] for 1D)
AXON_API void sum_last_dim_out(const axon::tensor::Tensor& in, axon::tensor::Tensor& out) noexcept;
AXON_NODISCARD AXON_API axon::tensor::Tensor sum_last_dim(const axon::tensor::Tensor& in) noexcept;
AXON_API void mean_last_dim_out(const axon::tensor::Tensor& in, axon::tensor::Tensor& out) noexcept;
AXON_NODISCARD AXON_API axon::tensor::Tensor mean_last_dim(const axon::tensor::Tensor& in) noexcept;

// Deep Activations & Normalizations (Explicit _out and functional allocating versions)
AXON_API void exp_out(const axon::tensor::Tensor& in, axon::tensor::Tensor& out) noexcept;
AXON_NODISCARD AXON_API axon::tensor::Tensor exp(const axon::tensor::Tensor& in) noexcept;

AXON_API void sigmoid_out(const axon::tensor::Tensor& in, axon::tensor::Tensor& out) noexcept;
AXON_NODISCARD AXON_API axon::tensor::Tensor sigmoid(const axon::tensor::Tensor& in) noexcept;

AXON_API void tanh_out(const axon::tensor::Tensor& in, axon::tensor::Tensor& out) noexcept;
AXON_NODISCARD AXON_API axon::tensor::Tensor tanh_op(const axon::tensor::Tensor& in) noexcept;
AXON_NODISCARD AXON_API axon::tensor::Tensor tanh(const axon::tensor::Tensor& in) noexcept;

AXON_API void silu_out(const axon::tensor::Tensor& in, axon::tensor::Tensor& out) noexcept;
AXON_NODISCARD AXON_API axon::tensor::Tensor silu(const axon::tensor::Tensor& in) noexcept;

AXON_API void gelu_out(const axon::tensor::Tensor& in, axon::tensor::Tensor& out) noexcept;
AXON_NODISCARD AXON_API axon::tensor::Tensor gelu(const axon::tensor::Tensor& in) noexcept;

AXON_API void softmax_out(const axon::tensor::Tensor& in, axon::tensor::Tensor& out) noexcept;
AXON_NODISCARD AXON_API axon::tensor::Tensor softmax(const axon::tensor::Tensor& in) noexcept;

AXON_API void rmsnorm_out(const axon::tensor::Tensor& in, const axon::tensor::Tensor& weight, axon::tensor::Tensor& out, float eps = 1e-5f) noexcept;
AXON_NODISCARD AXON_API axon::tensor::Tensor rmsnorm(const axon::tensor::Tensor& in, const axon::tensor::Tensor& weight, float eps = 1e-5f) noexcept;

} // namespace ops
} // namespace axon

#endif // AXON_OPS_REDUCTIONS_HPP
