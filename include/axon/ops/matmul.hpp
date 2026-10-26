#ifndef AXON_OPS_MATMUL_HPP
#define AXON_OPS_MATMUL_HPP

#include "axon/core/axon_macros.hpp"
#include "axon/core/types.hpp"
#include "axon/tensor/tensor.hpp"
#include <stddef.h>
#include <stdint.h>

namespace axon {
namespace ops {

// ============================================================================
// 1. CACHE BLOCKING CONSTANTS & BARE-METAL GEMM KERNEL DECLARATIONS
// ============================================================================

namespace kernels {

// L1/L2 Cache-Blocking Tile Dimensions tuned for 64-byte cache lines and
// SIMD register file saturation (AVX-512 32x ZMM / AVX2 16x YMM / NEON 32x V-regs)
constexpr size_t GEMM_BLOCK_M = 64;
constexpr size_t GEMM_BLOCK_K = 64;
constexpr size_t GEMM_BLOCK_N = 128;

// ----------------------------------------------------------------------------
// 1D Vector Inner Product (Dot Product) Kernels
// ----------------------------------------------------------------------------

AXON_NODISCARD AXON_API float dot_kernel_f32(
    const float* AXON_RESTRICT a,
    const float* AXON_RESTRICT b,
    size_t count
) noexcept;

AXON_NODISCARD AXON_API double dot_kernel_f64(
    const double* AXON_RESTRICT a,
    const double* AXON_RESTRICT b,
    size_t count
) noexcept;

AXON_NODISCARD AXON_API int32_t dot_kernel_i32(
    const int32_t* AXON_RESTRICT a,
    const int32_t* AXON_RESTRICT b,
    size_t count
) noexcept;

// ----------------------------------------------------------------------------
// 2D Contiguous Row-Major General Matrix Multiplication (GEMM)
// Computes: C[M, N] = A[M, K] * B[K, N]
// ----------------------------------------------------------------------------

AXON_API void gemm_kernel_f32(
    const float* AXON_RESTRICT a,
    const float* AXON_RESTRICT b,
    float* AXON_RESTRICT c,
    size_t M,
    size_t K,
    size_t N
) noexcept;

AXON_API void gemm_kernel_f64(
    const double* AXON_RESTRICT a,
    const double* AXON_RESTRICT b,
    double* AXON_RESTRICT c,
    size_t M,
    size_t K,
    size_t N
) noexcept;

AXON_API void gemm_kernel_i32(
    const int32_t* AXON_RESTRICT a,
    const int32_t* AXON_RESTRICT b,
    int32_t* AXON_RESTRICT c,
    size_t M,
    size_t K,
    size_t N
) noexcept;

// ----------------------------------------------------------------------------
// 2D Transposed-B General Matrix Multiplication (GEMM_NT)
// Computes: C[M, N] = A[M, K] * (B_T[N, K])^T
// Optimized for Dense/Linear layers where weights are stored as [N, K]
// ([out_features, in_features]), converting inner-loop strided loads into
// unit-stride contiguous SIMD FMA reductions.
// ----------------------------------------------------------------------------

AXON_API void gemm_nt_kernel_f32(
    const float* AXON_RESTRICT a,
    const float* AXON_RESTRICT b_t,
    float* AXON_RESTRICT c,
    size_t M,
    size_t K,
    size_t N
) noexcept;

AXON_API void gemm_nt_kernel_f64(
    const double* AXON_RESTRICT a,
    const double* AXON_RESTRICT b_t,
    double* AXON_RESTRICT c,
    size_t M,
    size_t K,
    size_t N
) noexcept;

// ----------------------------------------------------------------------------
// Fused 2D GEMM + Epilogue Bias Broadcast
// Computes: C[M, N] = A[M, K] * B[K, N] + bias[N]
// Initializes C tiles directly from bias[N] in L1 cache to eliminate a
// separate memory read/write pass over the output matrix.
// ----------------------------------------------------------------------------

AXON_API void gemm_bias_kernel_f32(
    const float* AXON_RESTRICT a,
    const float* AXON_RESTRICT b,
    const float* AXON_RESTRICT bias,
    float* AXON_RESTRICT c,
    size_t M,
    size_t K,
    size_t N
) noexcept;

} // namespace kernels

// ============================================================================
// 2. FAST-PATH TOPOLOGY & CONTRACT VALIDATION HELPERS
// ============================================================================

namespace detail {

// Detects whether a 2D Tensor of logical shape [K, N] is a zero-copy transposed
// view of a contiguous [N, K] row-major matrix (i.e., stride(0) == 1, stride(1) == K).
AXON_NODISCARD constexpr AXON_ALWAYS_INLINE bool is_transposed_contiguous_2d(
    const axon::tensor::Tensor& t
) noexcept {
    return (t.ndim() == 2) &&
           (t.stride(0) == 1) &&
           (t.stride(1) == t.dim(0));
}

// Validates invariants for 1D inner product (dot) operations
AXON_ALWAYS_INLINE void validate_dot_op(
    const axon::tensor::Tensor& a,
    const axon::tensor::Tensor& b
) noexcept {
    AXON_ASSERT(!a.empty() && !b.empty());
    AXON_ASSERT(a.ndim() == 1 && b.ndim() == 1);
    AXON_ASSERT(a.dim(0) == b.dim(0));
    AXON_ASSERT(a.dtype() == b.dtype());
    AXON_ASSERT(a.device() == b.device());
    AXON_ASSERT(a.is_contiguous() && b.is_contiguous());
}

// Validates invariants for 2D matrix multiplication: out[M, N] = a[M, K] @ b[K, N]
AXON_ALWAYS_INLINE void validate_gemm_2d_op(
    const axon::tensor::Tensor& a,
    const axon::tensor::Tensor& b,
    const axon::tensor::Tensor& out
) noexcept {
    AXON_ASSERT(!a.empty() && !b.empty() && !out.empty());
    AXON_ASSERT(a.ndim() == 2 && b.ndim() == 2 && out.ndim() == 2);
    AXON_ASSERT(a.dim(1) == b.dim(0));
    AXON_ASSERT(out.dim(0) == a.dim(0) && out.dim(1) == b.dim(1));
    AXON_ASSERT(a.dtype() == b.dtype() && a.dtype() == out.dtype());
    AXON_ASSERT(a.device() == b.device() && a.device() == out.device());
    AXON_ASSERT(a.is_contiguous() && out.is_contiguous());
    AXON_ASSERT(b.is_contiguous() || is_transposed_contiguous_2d(b));
}

// Validates invariants for 3D batched matrix multiplication: out[B, M, N] = a[B, M, K] @ b[B, K, N]
AXON_ALWAYS_INLINE void validate_bmm_3d_op(
    const axon::tensor::Tensor& a,
    const axon::tensor::Tensor& b,
    const axon::tensor::Tensor& out
) noexcept {
    AXON_ASSERT(!a.empty() && !b.empty() && !out.empty());
    AXON_ASSERT(a.ndim() == 3 && b.ndim() == 3 && out.ndim() == 3);
    AXON_ASSERT(a.dim(0) == b.dim(0) && a.dim(0) == out.dim(0));
    AXON_ASSERT(a.dim(2) == b.dim(1));
    AXON_ASSERT(out.dim(1) == a.dim(1) && out.dim(2) == b.dim(2));
    AXON_ASSERT(a.dtype() == b.dtype() && a.dtype() == out.dtype());
    AXON_ASSERT(a.device() == b.device() && a.device() == out.device());
    AXON_ASSERT(a.is_contiguous() && b.is_contiguous() && out.is_contiguous());
}

} // namespace detail

// ============================================================================
// 3. HIGH-LEVEL TENSOR MATRIX MULTIPLICATION & LINEAR ALGEBRA API
// ============================================================================

// Explicit-output matrix multiplication (zero dynamic memory allocation):
// - 1D x 1D: [K] @ [K] -> [1] (or 0-D scalar Tensor)
// - 2D x 2D: [M, K] @ [K, N] -> [M, N] (supports contiguous B or transposed-view B)
// - 3D x 3D: [B, M, K] @ [B, K, N] -> [B, M, N] (batched GEMM)
AXON_API void matmul_out(
    const axon::tensor::Tensor& a,
    const axon::tensor::Tensor& b,
    axon::tensor::Tensor& out
) noexcept;

// Functional allocating matrix multiplication: allocates output Tensor and dispatches matmul_out
AXON_NODISCARD AXON_API axon::tensor::Tensor matmul(
    const axon::tensor::Tensor& a,
    const axon::tensor::Tensor& b
) noexcept;

// Computes the scalar inner product of two 1D Float32 vectors
AXON_NODISCARD AXON_API float dot(
    const axon::tensor::Tensor& a,
    const axon::tensor::Tensor& b
) noexcept;

// Explicit-output Dense/Linear layer transformation:
// Computes out = input @ weight + bias (where bias is optional and may be nullptr).
// Supports 2D input [M, K] and 3D batched input [B, M, K], with weight [K, N]
// (contiguous or transposed view of [N, K]) and 1D bias [N].
AXON_API void linear_out(
    const axon::tensor::Tensor& input,
    const axon::tensor::Tensor& weight,
    const axon::tensor::Tensor* bias,
    axon::tensor::Tensor& out
) noexcept;

// Functional allocating Dense/Linear layer transformation:
// Allocates output Tensor and dispatches linear_out.
AXON_NODISCARD AXON_API axon::tensor::Tensor linear(
    const axon::tensor::Tensor& input,
    const axon::tensor::Tensor& weight,
    const axon::tensor::Tensor* bias = nullptr
) noexcept;

} // namespace ops
} // namespace axon

#endif // AXON_OPS_MATMUL_HPP
