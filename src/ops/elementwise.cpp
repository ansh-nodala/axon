#include "axon/ops/elementwise.hpp"

#if defined(AXON_ARCH_X86_64)
    #include <immintrin.h>
#elif defined(AXON_ARCH_ARM64) && defined(AXON_HAS_NEON)
    #include <arm_neon.h>
#endif

namespace axon {
namespace ops {

// ============================================================================
// 1. LOW-LEVEL BARE-METAL SIMD KERNEL IMPLEMENTATIONS
// ============================================================================

namespace kernels {

// ----------------------------------------------------------------------------
// Float32 (Single-Precision IEEE 754) Compute Kernels
// ----------------------------------------------------------------------------

void add_kernel_f32(
    const float* AXON_RESTRICT a,
    const float* AXON_RESTRICT b,
    float* AXON_RESTRICT out,
    size_t count
) noexcept {
    if (AXON_UNLIKELY(count == 0)) {
        return;
    }
    AXON_ASSERT(a != nullptr && b != nullptr && out != nullptr);

    size_t i = 0;

#if defined(AXON_HAS_AVX512F)
    for (; i + 32 <= count; i += 32) {
        __m512 va0 = _mm512_loadu_ps(a + i);
        __m512 vb0 = _mm512_loadu_ps(b + i);
        __m512 va1 = _mm512_loadu_ps(a + i + 16);
        __m512 vb1 = _mm512_loadu_ps(b + i + 16);
        _mm512_storeu_ps(out + i, _mm512_add_ps(va0, vb0));
        _mm512_storeu_ps(out + i + 16, _mm512_add_ps(va1, vb1));
    }
    for (; i + 16 <= count; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        _mm512_storeu_ps(out + i, _mm512_add_ps(va, vb));
    }
#elif defined(AXON_HAS_AVX2)
    for (; i + 32 <= count; i += 32) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        __m256 va2 = _mm256_loadu_ps(a + i + 16);
        __m256 vb2 = _mm256_loadu_ps(b + i + 16);
        __m256 va3 = _mm256_loadu_ps(a + i + 24);
        __m256 vb3 = _mm256_loadu_ps(b + i + 24);
        _mm256_storeu_ps(out + i, _mm256_add_ps(va0, vb0));
        _mm256_storeu_ps(out + i + 8, _mm256_add_ps(va1, vb1));
        _mm256_storeu_ps(out + i + 16, _mm256_add_ps(va2, vb2));
        _mm256_storeu_ps(out + i + 24, _mm256_add_ps(va3, vb3));
    }
    for (; i + 8 <= count; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
    }
#elif defined(AXON_HAS_NEON)
    for (; i + 16 <= count; i += 16) {
        float32x4_t va0 = vld1q_f32(a + i);
        float32x4_t vb0 = vld1q_f32(b + i);
        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
        float32x4_t va2 = vld1q_f32(a + i + 8);
        float32x4_t vb2 = vld1q_f32(b + i + 8);
        float32x4_t va3 = vld1q_f32(a + i + 12);
        float32x4_t vb3 = vld1q_f32(b + i + 12);
        vst1q_f32(out + i, vaddq_f32(va0, vb0));
        vst1q_f32(out + i + 4, vaddq_f32(va1, vb1));
        vst1q_f32(out + i + 8, vaddq_f32(va2, vb2));
        vst1q_f32(out + i + 12, vaddq_f32(va3, vb3));
    }
    for (; i + 4 <= count; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vaddq_f32(va, vb));
    }
#endif

    for (; i + 4 <= count; i += 4) {
        out[i]     = a[i]     + b[i];
        out[i + 1] = a[i + 1] + b[i + 1];
        out[i + 2] = a[i + 2] + b[i + 2];
        out[i + 3] = a[i + 3] + b[i + 3];
    }
    for (; i < count; ++i) {
        out[i] = a[i] + b[i];
    }
}

void sub_kernel_f32(
    const float* AXON_RESTRICT a,
    const float* AXON_RESTRICT b,
    float* AXON_RESTRICT out,
    size_t count
) noexcept {
    if (AXON_UNLIKELY(count == 0)) {
        return;
    }
    AXON_ASSERT(a != nullptr && b != nullptr && out != nullptr);

    size_t i = 0;

#if defined(AXON_HAS_AVX512F)
    for (; i + 32 <= count; i += 32) {
        __m512 va0 = _mm512_loadu_ps(a + i);
        __m512 vb0 = _mm512_loadu_ps(b + i);
        __m512 va1 = _mm512_loadu_ps(a + i + 16);
        __m512 vb1 = _mm512_loadu_ps(b + i + 16);
        _mm512_storeu_ps(out + i, _mm512_sub_ps(va0, vb0));
        _mm512_storeu_ps(out + i + 16, _mm512_sub_ps(va1, vb1));
    }
    for (; i + 16 <= count; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        _mm512_storeu_ps(out + i, _mm512_sub_ps(va, vb));
    }
#elif defined(AXON_HAS_AVX2)
    for (; i + 32 <= count; i += 32) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        __m256 va2 = _mm256_loadu_ps(a + i + 16);
        __m256 vb2 = _mm256_loadu_ps(b + i + 16);
        __m256 va3 = _mm256_loadu_ps(a + i + 24);
        __m256 vb3 = _mm256_loadu_ps(b + i + 24);
        _mm256_storeu_ps(out + i, _mm256_sub_ps(va0, vb0));
        _mm256_storeu_ps(out + i + 8, _mm256_sub_ps(va1, vb1));
        _mm256_storeu_ps(out + i + 16, _mm256_sub_ps(va2, vb2));
        _mm256_storeu_ps(out + i + 24, _mm256_sub_ps(va3, vb3));
    }
    for (; i + 8 <= count; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_sub_ps(va, vb));
    }
#elif defined(AXON_HAS_NEON)
    for (; i + 16 <= count; i += 16) {
        float32x4_t va0 = vld1q_f32(a + i);
        float32x4_t vb0 = vld1q_f32(b + i);
        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
        float32x4_t va2 = vld1q_f32(a + i + 8);
        float32x4_t vb2 = vld1q_f32(b + i + 8);
        float32x4_t va3 = vld1q_f32(a + i + 12);
        float32x4_t vb3 = vld1q_f32(b + i + 12);
        vst1q_f32(out + i, vsubq_f32(va0, vb0));
        vst1q_f32(out + i + 4, vsubq_f32(va1, vb1));
        vst1q_f32(out + i + 8, vsubq_f32(va2, vb2));
        vst1q_f32(out + i + 12, vsubq_f32(va3, vb3));
    }
    for (; i + 4 <= count; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vsubq_f32(va, vb));
    }
#endif

    for (; i + 4 <= count; i += 4) {
        out[i]     = a[i]     - b[i];
        out[i + 1] = a[i + 1] - b[i + 1];
        out[i + 2] = a[i + 2] - b[i + 2];
        out[i + 3] = a[i + 3] - b[i + 3];
    }
    for (; i < count; ++i) {
        out[i] = a[i] - b[i];
    }
}

void mul_kernel_f32(
    const float* AXON_RESTRICT a,
    const float* AXON_RESTRICT b,
    float* AXON_RESTRICT out,
    size_t count
) noexcept {
    if (AXON_UNLIKELY(count == 0)) {
        return;
    }
    AXON_ASSERT(a != nullptr && b != nullptr && out != nullptr);

    size_t i = 0;

#if defined(AXON_HAS_AVX512F)
    for (; i + 32 <= count; i += 32) {
        __m512 va0 = _mm512_loadu_ps(a + i);
        __m512 vb0 = _mm512_loadu_ps(b + i);
        __m512 va1 = _mm512_loadu_ps(a + i + 16);
        __m512 vb1 = _mm512_loadu_ps(b + i + 16);
        _mm512_storeu_ps(out + i, _mm512_mul_ps(va0, vb0));
        _mm512_storeu_ps(out + i + 16, _mm512_mul_ps(va1, vb1));
    }
    for (; i + 16 <= count; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        _mm512_storeu_ps(out + i, _mm512_mul_ps(va, vb));
    }
#elif defined(AXON_HAS_AVX2)
    for (; i + 32 <= count; i += 32) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        __m256 va2 = _mm256_loadu_ps(a + i + 16);
        __m256 vb2 = _mm256_loadu_ps(b + i + 16);
        __m256 va3 = _mm256_loadu_ps(a + i + 24);
        __m256 vb3 = _mm256_loadu_ps(b + i + 24);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(va0, vb0));
        _mm256_storeu_ps(out + i + 8, _mm256_mul_ps(va1, vb1));
        _mm256_storeu_ps(out + i + 16, _mm256_mul_ps(va2, vb2));
        _mm256_storeu_ps(out + i + 24, _mm256_mul_ps(va3, vb3));
    }
    for (; i + 8 <= count; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(va, vb));
    }
#elif defined(AXON_HAS_NEON)
    for (; i + 16 <= count; i += 16) {
        float32x4_t va0 = vld1q_f32(a + i);
        float32x4_t vb0 = vld1q_f32(b + i);
        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
        float32x4_t va2 = vld1q_f32(a + i + 8);
        float32x4_t vb2 = vld1q_f32(b + i + 8);
        float32x4_t va3 = vld1q_f32(a + i + 12);
        float32x4_t vb3 = vld1q_f32(b + i + 12);
        vst1q_f32(out + i, vmulq_f32(va0, vb0));
        vst1q_f32(out + i + 4, vmulq_f32(va1, vb1));
        vst1q_f32(out + i + 8, vmulq_f32(va2, vb2));
        vst1q_f32(out + i + 12, vmulq_f32(va3, vb3));
    }
    for (; i + 4 <= count; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vmulq_f32(va, vb));
    }
#endif

    for (; i + 4 <= count; i += 4) {
        out[i]     = a[i]     * b[i];
        out[i + 1] = a[i + 1] * b[i + 1];
        out[i + 2] = a[i + 2] * b[i + 2];
        out[i + 3] = a[i + 3] * b[i + 3];
    }
    for (; i < count; ++i) {
        out[i] = a[i] * b[i];
    }
}

void div_kernel_f32(
    const float* AXON_RESTRICT a,
    const float* AXON_RESTRICT b,
    float* AXON_RESTRICT out,
    size_t count
) noexcept {
    if (AXON_UNLIKELY(count == 0)) {
        return;
    }
    AXON_ASSERT(a != nullptr && b != nullptr && out != nullptr);

    size_t i = 0;

#if defined(AXON_HAS_AVX512F)
    for (; i + 32 <= count; i += 32) {
        __m512 va0 = _mm512_loadu_ps(a + i);
        __m512 vb0 = _mm512_loadu_ps(b + i);
        __m512 va1 = _mm512_loadu_ps(a + i + 16);
        __m512 vb1 = _mm512_loadu_ps(b + i + 16);
        _mm512_storeu_ps(out + i, _mm512_div_ps(va0, vb0));
        _mm512_storeu_ps(out + i + 16, _mm512_div_ps(va1, vb1));
    }
    for (; i + 16 <= count; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        _mm512_storeu_ps(out + i, _mm512_div_ps(va, vb));
    }
#elif defined(AXON_HAS_AVX2)
    for (; i + 32 <= count; i += 32) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        __m256 va2 = _mm256_loadu_ps(a + i + 16);
        __m256 vb2 = _mm256_loadu_ps(b + i + 16);
        __m256 va3 = _mm256_loadu_ps(a + i + 24);
        __m256 vb3 = _mm256_loadu_ps(b + i + 24);
        _mm256_storeu_ps(out + i, _mm256_div_ps(va0, vb0));
        _mm256_storeu_ps(out + i + 8, _mm256_div_ps(va1, vb1));
        _mm256_storeu_ps(out + i + 16, _mm256_div_ps(va2, vb2));
        _mm256_storeu_ps(out + i + 24, _mm256_div_ps(va3, vb3));
    }
    for (; i + 8 <= count; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_div_ps(va, vb));
    }
#elif defined(AXON_HAS_NEON)
    for (; i + 16 <= count; i += 16) {
        float32x4_t va0 = vld1q_f32(a + i);
        float32x4_t vb0 = vld1q_f32(b + i);
        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
        float32x4_t va2 = vld1q_f32(a + i + 8);
        float32x4_t vb2 = vld1q_f32(b + i + 8);
        float32x4_t va3 = vld1q_f32(a + i + 12);
        float32x4_t vb3 = vld1q_f32(b + i + 12);
        vst1q_f32(out + i, vdivq_f32(va0, vb0));
        vst1q_f32(out + i + 4, vdivq_f32(va1, vb1));
        vst1q_f32(out + i + 8, vdivq_f32(va2, vb2));
        vst1q_f32(out + i + 12, vdivq_f32(va3, vb3));
    }
    for (; i + 4 <= count; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vdivq_f32(va, vb));
    }
#endif

    for (; i + 4 <= count; i += 4) {
        out[i]     = a[i]     / b[i];
        out[i + 1] = a[i + 1] / b[i + 1];
        out[i + 2] = a[i + 2] / b[i + 2];
        out[i + 3] = a[i + 3] / b[i + 3];
    }
    for (; i < count; ++i) {
        out[i] = a[i] / b[i];
    }
}

void relu_kernel_f32(
    const float* AXON_RESTRICT in,
    float* AXON_RESTRICT out,
    size_t count
) noexcept {
    if (AXON_UNLIKELY(count == 0)) {
        return;
    }
    AXON_ASSERT(in != nullptr && out != nullptr);

    size_t i = 0;

#if defined(AXON_HAS_AVX512F)
    const __m512 vzero = _mm512_setzero_ps();
    for (; i + 32 <= count; i += 32) {
        __m512 v0 = _mm512_loadu_ps(in + i);
        __m512 v1 = _mm512_loadu_ps(in + i + 16);
        _mm512_storeu_ps(out + i, _mm512_max_ps(v0, vzero));
        _mm512_storeu_ps(out + i + 16, _mm512_max_ps(v1, vzero));
    }
    for (; i + 16 <= count; i += 16) {
        __m512 v = _mm512_loadu_ps(in + i);
        _mm512_storeu_ps(out + i, _mm512_max_ps(v, vzero));
    }
#elif defined(AXON_HAS_AVX2)
    const __m256 vzero = _mm256_setzero_ps();
    for (; i + 32 <= count; i += 32) {
        __m256 v0 = _mm256_loadu_ps(in + i);
        __m256 v1 = _mm256_loadu_ps(in + i + 8);
        __m256 v2 = _mm256_loadu_ps(in + i + 16);
        __m256 v3 = _mm256_loadu_ps(in + i + 24);
        _mm256_storeu_ps(out + i, _mm256_max_ps(v0, vzero));
        _mm256_storeu_ps(out + i + 8, _mm256_max_ps(v1, vzero));
        _mm256_storeu_ps(out + i + 16, _mm256_max_ps(v2, vzero));
        _mm256_storeu_ps(out + i + 24, _mm256_max_ps(v3, vzero));
    }
    for (; i + 8 <= count; i += 8) {
        __m256 v = _mm256_loadu_ps(in + i);
        _mm256_storeu_ps(out + i, _mm256_max_ps(v, vzero));
    }
#elif defined(AXON_HAS_NEON)
    const float32x4_t vzero = vdupq_n_f32(0.0f);
    for (; i + 16 <= count; i += 16) {
        float32x4_t v0 = vld1q_f32(in + i);
        float32x4_t v1 = vld1q_f32(in + i + 4);
        float32x4_t v2 = vld1q_f32(in + i + 8);
        float32x4_t v3 = vld1q_f32(in + i + 12);
        vst1q_f32(out + i, vmaxq_f32(v0, vzero));
        vst1q_f32(out + i + 4, vmaxq_f32(v1, vzero));
        vst1q_f32(out + i + 8, vmaxq_f32(v2, vzero));
        vst1q_f32(out + i + 12, vmaxq_f32(v3, vzero));
    }
    for (; i + 4 <= count; i += 4) {
        float32x4_t v = vld1q_f32(in + i);
        vst1q_f32(out + i, vmaxq_f32(v, vzero));
    }
#endif

    for (; i + 4 <= count; i += 4) {
        const float v0 = in[i];
        const float v1 = in[i + 1];
        const float v2 = in[i + 2];
        const float v3 = in[i + 3];
        out[i]     = (v0 > 0.0f) ? v0 : 0.0f;
        out[i + 1] = (v1 > 0.0f) ? v1 : 0.0f;
        out[i + 2] = (v2 > 0.0f) ? v2 : 0.0f;
        out[i + 3] = (v3 > 0.0f) ? v3 : 0.0f;
    }
    for (; i < count; ++i) {
        const float v = in[i];
        out[i] = (v > 0.0f) ? v : 0.0f;
    }
}

void fill_kernel_f32(
    float* AXON_RESTRICT out,
    float value,
    size_t count
) noexcept {
    if (AXON_UNLIKELY(count == 0)) {
        return;
    }
    AXON_ASSERT(out != nullptr);

    size_t i = 0;

#if defined(AXON_HAS_AVX512F)
    const __m512 vval = _mm512_set1_ps(value);
    for (; i + 32 <= count; i += 32) {
        _mm512_storeu_ps(out + i, vval);
        _mm512_storeu_ps(out + i + 16, vval);
    }
    for (; i + 16 <= count; i += 16) {
        _mm512_storeu_ps(out + i, vval);
    }
#elif defined(AXON_HAS_AVX2)
    const __m256 vval = _mm256_set1_ps(value);
    for (; i + 32 <= count; i += 32) {
        _mm256_storeu_ps(out + i, vval);
        _mm256_storeu_ps(out + i + 8, vval);
        _mm256_storeu_ps(out + i + 16, vval);
        _mm256_storeu_ps(out + i + 24, vval);
    }
    for (; i + 8 <= count; i += 8) {
        _mm256_storeu_ps(out + i, vval);
    }
#elif defined(AXON_HAS_NEON)
    const float32x4_t vval = vdupq_n_f32(value);
    for (; i + 16 <= count; i += 16) {
        vst1q_f32(out + i, vval);
        vst1q_f32(out + i + 4, vval);
        vst1q_f32(out + i + 8, vval);
        vst1q_f32(out + i + 12, vval);
    }
    for (; i + 4 <= count; i += 4) {
        vst1q_f32(out + i, vval);
    }
#endif

    for (; i + 4 <= count; i += 4) {
        out[i]     = value;
        out[i + 1] = value;
        out[i + 2] = value;
        out[i + 3] = value;
    }
    for (; i < count; ++i) {
        out[i] = value;
    }
}

void scale_kernel_f32(
    const float* AXON_RESTRICT in,
    float scalar,
    float* AXON_RESTRICT out,
    size_t count
) noexcept {
    if (AXON_UNLIKELY(count == 0)) {
        return;
    }
    AXON_ASSERT(in != nullptr && out != nullptr);

    size_t i = 0;

#if defined(AXON_HAS_AVX512F)
    const __m512 vscalar = _mm512_set1_ps(scalar);
    for (; i + 32 <= count; i += 32) {
        __m512 v0 = _mm512_loadu_ps(in + i);
        __m512 v1 = _mm512_loadu_ps(in + i + 16);
        _mm512_storeu_ps(out + i, _mm512_mul_ps(v0, vscalar));
        _mm512_storeu_ps(out + i + 16, _mm512_mul_ps(v1, vscalar));
    }
    for (; i + 16 <= count; i += 16) {
        __m512 v = _mm512_loadu_ps(in + i);
        _mm512_storeu_ps(out + i, _mm512_mul_ps(v, vscalar));
    }
#elif defined(AXON_HAS_AVX2)
    const __m256 vscalar = _mm256_set1_ps(scalar);
    for (; i + 32 <= count; i += 32) {
        __m256 v0 = _mm256_loadu_ps(in + i);
        __m256 v1 = _mm256_loadu_ps(in + i + 8);
        __m256 v2 = _mm256_loadu_ps(in + i + 16);
        __m256 v3 = _mm256_loadu_ps(in + i + 24);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(v0, vscalar));
        _mm256_storeu_ps(out + i + 8, _mm256_mul_ps(v1, vscalar));
        _mm256_storeu_ps(out + i + 16, _mm256_mul_ps(v2, vscalar));
        _mm256_storeu_ps(out + i + 24, _mm256_mul_ps(v3, vscalar));
    }
    for (; i + 8 <= count; i += 8) {
        __m256 v = _mm256_loadu_ps(in + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(v, vscalar));
    }
#elif defined(AXON_HAS_NEON)
    const float32x4_t vscalar = vdupq_n_f32(scalar);
    for (; i + 16 <= count; i += 16) {
        float32x4_t v0 = vld1q_f32(in + i);
        float32x4_t v1 = vld1q_f32(in + i + 4);
        float32x4_t v2 = vld1q_f32(in + i + 8);
        float32x4_t v3 = vld1q_f32(in + i + 12);
        vst1q_f32(out + i, vmulq_f32(v0, vscalar));
        vst1q_f32(out + i + 4, vmulq_f32(v1, vscalar));
        vst1q_f32(out + i + 8, vmulq_f32(v2, vscalar));
        vst1q_f32(out + i + 12, vmulq_f32(v3, vscalar));
    }
    for (; i + 4 <= count; i += 4) {
        float32x4_t v = vld1q_f32(in + i);
        vst1q_f32(out + i, vmulq_f32(v, vscalar));
    }
#endif

    for (; i + 4 <= count; i += 4) {
        out[i]     = in[i]     * scalar;
        out[i + 1] = in[i + 1] * scalar;
        out[i + 2] = in[i + 2] * scalar;
        out[i + 3] = in[i + 3] * scalar;
    }
    for (; i < count; ++i) {
        out[i] = in[i] * scalar;
    }
}

// ----------------------------------------------------------------------------
// Float64 (Double-Precision IEEE 754) Compute Kernels
// ----------------------------------------------------------------------------

void add_kernel_f64(
    const double* AXON_RESTRICT a,
    const double* AXON_RESTRICT b,
    double* AXON_RESTRICT out,
    size_t count
) noexcept {
    if (AXON_UNLIKELY(count == 0)) {
        return;
    }
    AXON_ASSERT(a != nullptr && b != nullptr && out != nullptr);

    size_t i = 0;

#if defined(AXON_HAS_AVX512F)
    for (; i + 16 <= count; i += 16) {
        __m512d va0 = _mm512_loadu_pd(a + i);
        __m512d vb0 = _mm512_loadu_pd(b + i);
        __m512d va1 = _mm512_loadu_pd(a + i + 8);
        __m512d vb1 = _mm512_loadu_pd(b + i + 8);
        _mm512_storeu_pd(out + i, _mm512_add_pd(va0, vb0));
        _mm512_storeu_pd(out + i + 8, _mm512_add_pd(va1, vb1));
    }
    for (; i + 8 <= count; i += 8) {
        __m512d va = _mm512_loadu_pd(a + i);
        __m512d vb = _mm512_loadu_pd(b + i);
        _mm512_storeu_pd(out + i, _mm512_add_pd(va, vb));
    }
#elif defined(AXON_HAS_AVX2)
    for (; i + 16 <= count; i += 16) {
        __m256d va0 = _mm256_loadu_pd(a + i);
        __m256d vb0 = _mm256_loadu_pd(b + i);
        __m256d va1 = _mm256_loadu_pd(a + i + 4);
        __m256d vb1 = _mm256_loadu_pd(b + i + 4);
        __m256d va2 = _mm256_loadu_pd(a + i + 8);
        __m256d vb2 = _mm256_loadu_pd(b + i + 8);
        __m256d va3 = _mm256_loadu_pd(a + i + 12);
        __m256d vb3 = _mm256_loadu_pd(b + i + 12);
        _mm256_storeu_pd(out + i, _mm256_add_pd(va0, vb0));
        _mm256_storeu_pd(out + i + 4, _mm256_add_pd(va1, vb1));
        _mm256_storeu_pd(out + i + 8, _mm256_add_pd(va2, vb2));
        _mm256_storeu_pd(out + i + 12, _mm256_add_pd(va3, vb3));
    }
    for (; i + 4 <= count; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(out + i, _mm256_add_pd(va, vb));
    }
#elif defined(AXON_HAS_NEON)
    for (; i + 8 <= count; i += 8) {
        float64x2_t va0 = vld1q_f64(a + i);
        float64x2_t vb0 = vld1q_f64(b + i);
        float64x2_t va1 = vld1q_f64(a + i + 2);
        float64x2_t vb1 = vld1q_f64(b + i + 2);
        float64x2_t va2 = vld1q_f64(a + i + 4);
        float64x2_t vb2 = vld1q_f64(b + i + 4);
        float64x2_t va3 = vld1q_f64(a + i + 6);
        float64x2_t vb3 = vld1q_f64(b + i + 6);
        vst1q_f64(out + i, vaddq_f64(va0, vb0));
        vst1q_f64(out + i + 2, vaddq_f64(va1, vb1));
        vst1q_f64(out + i + 4, vaddq_f64(va2, vb2));
        vst1q_f64(out + i + 6, vaddq_f64(va3, vb3));
    }
    for (; i + 2 <= count; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        vst1q_f64(out + i, vaddq_f64(va, vb));
    }
#endif

    for (; i + 4 <= count; i += 4) {
        out[i]     = a[i]     + b[i];
        out[i + 1] = a[i + 1] + b[i + 1];
        out[i + 2] = a[i + 2] + b[i + 2];
        out[i + 3] = a[i + 3] + b[i + 3];
    }
    for (; i < count; ++i) {
        out[i] = a[i] + b[i];
    }
}

void sub_kernel_f64(
    const double* AXON_RESTRICT a,
    const double* AXON_RESTRICT b,
    double* AXON_RESTRICT out,
    size_t count
) noexcept {
    if (AXON_UNLIKELY(count == 0)) {
        return;
    }
    AXON_ASSERT(a != nullptr && b != nullptr && out != nullptr);

    size_t i = 0;

#if defined(AXON_HAS_AVX512F)
    for (; i + 16 <= count; i += 16) {
        __m512d va0 = _mm512_loadu_pd(a + i);
        __m512d vb0 = _mm512_loadu_pd(b + i);
        __m512d va1 = _mm512_loadu_pd(a + i + 8);
        __m512d vb1 = _mm512_loadu_pd(b + i + 8);
        _mm512_storeu_pd(out + i, _mm512_sub_pd(va0, vb0));
        _mm512_storeu_pd(out + i + 8, _mm512_sub_pd(va1, vb1));
    }
    for (; i + 8 <= count; i += 8) {
        __m512d va = _mm512_loadu_pd(a + i);
        __m512d vb = _mm512_loadu_pd(b + i);
        _mm512_storeu_pd(out + i, _mm512_sub_pd(va, vb));
    }
#elif defined(AXON_HAS_AVX2)
    for (; i + 16 <= count; i += 16) {
        __m256d va0 = _mm256_loadu_pd(a + i);
        __m256d vb0 = _mm256_loadu_pd(b + i);
        __m256d va1 = _mm256_loadu_pd(a + i + 4);
        __m256d vb1 = _mm256_loadu_pd(b + i + 4);
        __m256d va2 = _mm256_loadu_pd(a + i + 8);
        __m256d vb2 = _mm256_loadu_pd(b + i + 8);
        __m256d va3 = _mm256_loadu_pd(a + i + 12);
        __m256d vb3 = _mm256_loadu_pd(b + i + 12);
        _mm256_storeu_pd(out + i, _mm256_sub_pd(va0, vb0));
        _mm256_storeu_pd(out + i + 4, _mm256_sub_pd(va1, vb1));
        _mm256_storeu_pd(out + i + 8, _mm256_sub_pd(va2, vb2));
        _mm256_storeu_pd(out + i + 12, _mm256_sub_pd(va3, vb3));
    }
    for (; i + 4 <= count; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(out + i, _mm256_sub_pd(va, vb));
    }
#elif defined(AXON_HAS_NEON)
    for (; i + 8 <= count; i += 8) {
        float64x2_t va0 = vld1q_f64(a + i);
        float64x2_t vb0 = vld1q_f64(b + i);
        float64x2_t va1 = vld1q_f64(a + i + 2);
        float64x2_t vb1 = vld1q_f64(b + i + 2);
        float64x2_t va2 = vld1q_f64(a + i + 4);
        float64x2_t vb2 = vld1q_f64(b + i + 4);
        float64x2_t va3 = vld1q_f64(a + i + 6);
        float64x2_t vb3 = vld1q_f64(b + i + 6);
        vst1q_f64(out + i, vsubq_f64(va0, vb0));
        vst1q_f64(out + i + 2, vsubq_f64(va1, vb1));
        vst1q_f64(out + i + 4, vsubq_f64(va2, vb2));
        vst1q_f64(out + i + 6, vsubq_f64(va3, vb3));
    }
    for (; i + 2 <= count; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        vst1q_f64(out + i, vsubq_f64(va, vb));
    }
#endif

    for (; i + 4 <= count; i += 4) {
        out[i]     = a[i]     - b[i];
        out[i + 1] = a[i + 1] - b[i + 1];
        out[i + 2] = a[i + 2] - b[i + 2];
        out[i + 3] = a[i + 3] - b[i + 3];
    }
    for (; i < count; ++i) {
        out[i] = a[i] - b[i];
    }
}

void mul_kernel_f64(
    const double* AXON_RESTRICT a,
    const double* AXON_RESTRICT b,
    double* AXON_RESTRICT out,
    size_t count
) noexcept {
    if (AXON_UNLIKELY(count == 0)) {
        return;
    }
    AXON_ASSERT(a != nullptr && b != nullptr && out != nullptr);

    size_t i = 0;

#if defined(AXON_HAS_AVX512F)
    for (; i + 16 <= count; i += 16) {
        __m512d va0 = _mm512_loadu_pd(a + i);
        __m512d vb0 = _mm512_loadu_pd(b + i);
        __m512d va1 = _mm512_loadu_pd(a + i + 8);
        __m512d vb1 = _mm512_loadu_pd(b + i + 8);
        _mm512_storeu_pd(out + i, _mm512_mul_pd(va0, vb0));
        _mm512_storeu_pd(out + i + 8, _mm512_mul_pd(va1, vb1));
    }
    for (; i + 8 <= count; i += 8) {
        __m512d va = _mm512_loadu_pd(a + i);
        __m512d vb = _mm512_loadu_pd(b + i);
        _mm512_storeu_pd(out + i, _mm512_mul_pd(va, vb));
    }
#elif defined(AXON_HAS_AVX2)
    for (; i + 16 <= count; i += 16) {
        __m256d va0 = _mm256_loadu_pd(a + i);
        __m256d vb0 = _mm256_loadu_pd(b + i);
        __m256d va1 = _mm256_loadu_pd(a + i + 4);
        __m256d vb1 = _mm256_loadu_pd(b + i + 4);
        __m256d va2 = _mm256_loadu_pd(a + i + 8);
        __m256d vb2 = _mm256_loadu_pd(b + i + 8);
        __m256d va3 = _mm256_loadu_pd(a + i + 12);
        __m256d vb3 = _mm256_loadu_pd(b + i + 12);
        _mm256_storeu_pd(out + i, _mm256_mul_pd(va0, vb0));
        _mm256_storeu_pd(out + i + 4, _mm256_mul_pd(va1, vb1));
        _mm256_storeu_pd(out + i + 8, _mm256_mul_pd(va2, vb2));
        _mm256_storeu_pd(out + i + 12, _mm256_mul_pd(va3, vb3));
    }
    for (; i + 4 <= count; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(out + i, _mm256_mul_pd(va, vb));
    }
#elif defined(AXON_HAS_NEON)
    for (; i + 8 <= count; i += 8) {
        float64x2_t va0 = vld1q_f64(a + i);
        float64x2_t vb0 = vld1q_f64(b + i);
        float64x2_t va1 = vld1q_f64(a + i + 2);
        float64x2_t vb1 = vld1q_f64(b + i + 2);
        float64x2_t va2 = vld1q_f64(a + i + 4);
        float64x2_t vb2 = vld1q_f64(b + i + 4);
        float64x2_t va3 = vld1q_f64(a + i + 6);
        float64x2_t vb3 = vld1q_f64(b + i + 6);
        vst1q_f64(out + i, vmulq_f64(va0, vb0));
        vst1q_f64(out + i + 2, vmulq_f64(va1, vb1));
        vst1q_f64(out + i + 4, vmulq_f64(va2, vb2));
        vst1q_f64(out + i + 6, vmulq_f64(va3, vb3));
    }
    for (; i + 2 <= count; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        vst1q_f64(out + i, vmulq_f64(va, vb));
    }
#endif

    for (; i + 4 <= count; i += 4) {
        out[i]     = a[i]     * b[i];
        out[i + 1] = a[i + 1] * b[i + 1];
        out[i + 2] = a[i + 2] * b[i + 2];
        out[i + 3] = a[i + 3] * b[i + 3];
    }
    for (; i < count; ++i) {
        out[i] = a[i] * b[i];
    }
}

void div_kernel_f64(
    const double* AXON_RESTRICT a,
    const double* AXON_RESTRICT b,
    double* AXON_RESTRICT out,
    size_t count
) noexcept {
    if (AXON_UNLIKELY(count == 0)) {
        return;
    }
    AXON_ASSERT(a != nullptr && b != nullptr && out != nullptr);

    size_t i = 0;

#if defined(AXON_HAS_AVX512F)
    for (; i + 16 <= count; i += 16) {
        __m512d va0 = _mm512_loadu_pd(a + i);
        __m512d vb0 = _mm512_loadu_pd(b + i);
        __m512d va1 = _mm512_loadu_pd(a + i + 8);
        __m512d vb1 = _mm512_loadu_pd(b + i + 8);
        _mm512_storeu_pd(out + i, _mm512_div_pd(va0, vb0));
        _mm512_storeu_pd(out + i + 8, _mm512_div_pd(va1, vb1));
    }
    for (; i + 8 <= count; i += 8) {
        __m512d va = _mm512_loadu_pd(a + i);
        __m512d vb = _mm512_loadu_pd(b + i);
        _mm512_storeu_pd(out + i, _mm512_div_pd(va, vb));
    }
#elif defined(AXON_HAS_AVX2)
    for (; i + 16 <= count; i += 16) {
        __m256d va0 = _mm256_loadu_pd(a + i);
        __m256d vb0 = _mm256_loadu_pd(b + i);
        __m256d va1 = _mm256_loadu_pd(a + i + 4);
        __m256d vb1 = _mm256_loadu_pd(b + i + 4);
        __m256d va2 = _mm256_loadu_pd(a + i + 8);
        __m256d vb2 = _mm256_loadu_pd(b + i + 8);
        __m256d va3 = _mm256_loadu_pd(a + i + 12);
        __m256d vb3 = _mm256_loadu_pd(b + i + 12);
        _mm256_storeu_pd(out + i, _mm256_div_pd(va0, vb0));
        _mm256_storeu_pd(out + i + 4, _mm256_div_pd(va1, vb1));
        _mm256_storeu_pd(out + i + 8, _mm256_div_pd(va2, vb2));
        _mm256_storeu_pd(out + i + 12, _mm256_div_pd(va3, vb3));
    }
    for (; i + 4 <= count; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(out + i, _mm256_div_pd(va, vb));
    }
#elif defined(AXON_HAS_NEON)
    for (; i + 8 <= count; i += 8) {
        float64x2_t va0 = vld1q_f64(a + i);
        float64x2_t vb0 = vld1q_f64(b + i);
        float64x2_t va1 = vld1q_f64(a + i + 2);
        float64x2_t vb1 = vld1q_f64(b + i + 2);
        float64x2_t va2 = vld1q_f64(a + i + 4);
        float64x2_t vb2 = vld1q_f64(b + i + 4);
        float64x2_t va3 = vld1q_f64(a + i + 6);
        float64x2_t vb3 = vld1q_f64(b + i + 6);
        vst1q_f64(out + i, vdivq_f64(va0, vb0));
        vst1q_f64(out + i + 2, vdivq_f64(va1, vb1));
        vst1q_f64(out + i + 4, vdivq_f64(va2, vb2));
        vst1q_f64(out + i + 6, vdivq_f64(va3, vb3));
    }
    for (; i + 2 <= count; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        vst1q_f64(out + i, vdivq_f64(va, vb));
    }
#endif

    for (; i + 4 <= count; i += 4) {
        out[i]     = a[i]     / b[i];
        out[i + 1] = a[i + 1] / b[i + 1];
        out[i + 2] = a[i + 2] / b[i + 2];
        out[i + 3] = a[i + 3] / b[i + 3];
    }
    for (; i < count; ++i) {
        out[i] = a[i] / b[i];
    }
}

void relu_kernel_f64(
    const double* AXON_RESTRICT in,
    double* AXON_RESTRICT out,
    size_t count
) noexcept {
    if (AXON_UNLIKELY(count == 0)) {
        return;
    }
    AXON_ASSERT(in != nullptr && out != nullptr);

    size_t i = 0;

#if defined(AXON_HAS_AVX512F)
    const __m512d vzero = _mm512_setzero_pd();
    for (; i + 16 <= count; i += 16) {
        __m512d v0 = _mm512_loadu_pd(in + i);
        __m512d v1 = _mm512_loadu_pd(in + i + 8);
        _mm512_storeu_pd(out + i, _mm512_max_pd(v0, vzero));
        _mm512_storeu_pd(out + i + 8, _mm512_max_pd(v1, vzero));
    }
    for (; i + 8 <= count; i += 8) {
        __m512d v = _mm512_loadu_pd(in + i);
        _mm512_storeu_pd(out + i, _mm512_max_pd(v, vzero));
    }
#elif defined(AXON_HAS_AVX2)
    const __m256d vzero = _mm256_setzero_pd();
    for (; i + 16 <= count; i += 16) {
        __m256d v0 = _mm256_loadu_pd(in + i);
        __m256d v1 = _mm256_loadu_pd(in + i + 4);
        __m256d v2 = _mm256_loadu_pd(in + i + 8);
        __m256d v3 = _mm256_loadu_pd(in + i + 12);
        _mm256_storeu_pd(out + i, _mm256_max_pd(v0, vzero));
        _mm256_storeu_pd(out + i + 4, _mm256_max_pd(v1, vzero));
        _mm256_storeu_pd(out + i + 8, _mm256_max_pd(v2, vzero));
        _mm256_storeu_pd(out + i + 12, _mm256_max_pd(v3, vzero));
    }
    for (; i + 4 <= count; i += 4) {
        __m256d v = _mm256_loadu_pd(in + i);
        _mm256_storeu_pd(out + i, _mm256_max_pd(v, vzero));
    }
#elif defined(AXON_HAS_NEON)
    const float64x2_t vzero = vdupq_n_f64(0.0);
    for (; i + 8 <= count; i += 8) {
        float64x2_t v0 = vld1q_f64(in + i);
        float64x2_t v1 = vld1q_f64(in + i + 2);
        float64x2_t v2 = vld1q_f64(in + i + 4);
        float64x2_t v3 = vld1q_f64(in + i + 6);
        vst1q_f64(out + i, vmaxq_f64(v0, vzero));
        vst1q_f64(out + i + 2, vmaxq_f64(v1, vzero));
        vst1q_f64(out + i + 4, vmaxq_f64(v2, vzero));
        vst1q_f64(out + i + 6, vmaxq_f64(v3, vzero));
    }
    for (; i + 2 <= count; i += 2) {
        float64x2_t v = vld1q_f64(in + i);
        vst1q_f64(out + i, vmaxq_f64(v, vzero));
    }
#endif

    for (; i + 4 <= count; i += 4) {
        const double v0 = in[i];
        const double v1 = in[i + 1];
        const double v2 = in[i + 2];
        const double v3 = in[i + 3];
        out[i]     = (v0 > 0.0) ? v0 : 0.0;
        out[i + 1] = (v1 > 0.0) ? v1 : 0.0;
        out[i + 2] = (v2 > 0.0) ? v2 : 0.0;
        out[i + 3] = (v3 > 0.0) ? v3 : 0.0;
    }
    for (; i < count; ++i) {
        const double v = in[i];
        out[i] = (v > 0.0) ? v : 0.0;
    }
}

void fill_kernel_f64(
    double* AXON_RESTRICT out,
    double value,
    size_t count
) noexcept {
    if (AXON_UNLIKELY(count == 0)) {
        return;
    }
    AXON_ASSERT(out != nullptr);

    size_t i = 0;

#if defined(AXON_HAS_AVX512F)
    const __m512d vval = _mm512_set1_pd(value);
    for (; i + 16 <= count; i += 16) {
        _mm512_storeu_pd(out + i, vval);
        _mm512_storeu_pd(out + i + 8, vval);
    }
    for (; i + 8 <= count; i += 8) {
        _mm512_storeu_pd(out + i, vval);
    }
#elif defined(AXON_HAS_AVX2)
    const __m256d vval = _mm256_set1_pd(value);
    for (; i + 16 <= count; i += 16) {
        _mm256_storeu_pd(out + i, vval);
        _mm256_storeu_pd(out + i + 4, vval);
        _mm256_storeu_pd(out + i + 8, vval);
        _mm256_storeu_pd(out + i + 12, vval);
    }
    for (; i + 4 <= count; i += 4) {
        _mm256_storeu_pd(out + i, vval);
    }
#elif defined(AXON_HAS_NEON)
    const float64x2_t vval = vdupq_n_f64(value);
    for (; i + 8 <= count; i += 8) {
        vst1q_f64(out + i, vval);
        vst1q_f64(out + i + 2, vval);
        vst1q_f64(out + i + 4, vval);
        vst1q_f64(out + i + 6, vval);
    }
    for (; i + 2 <= count; i += 2) {
        vst1q_f64(out + i, vval);
    }
#endif

    for (; i + 4 <= count; i += 4) {
        out[i]     = value;
        out[i + 1] = value;
        out[i + 2] = value;
        out[i + 3] = value;
    }
    for (; i < count; ++i) {
        out[i] = value;
    }
}

void scale_kernel_f64(
    const double* AXON_RESTRICT in,
    double scalar,
    double* AXON_RESTRICT out,
    size_t count
) noexcept {
    if (AXON_UNLIKELY(count == 0)) {
        return;
    }
    AXON_ASSERT(in != nullptr && out != nullptr);

    size_t i = 0;

#if defined(AXON_HAS_AVX512F)
    const __m512d vscalar = _mm512_set1_pd(scalar);
    for (; i + 16 <= count; i += 16) {
        __m512d v0 = _mm512_loadu_pd(in + i);
        __m512d v1 = _mm512_loadu_pd(in + i + 8);
        _mm512_storeu_pd(out + i, _mm512_mul_pd(v0, vscalar));
        _mm512_storeu_pd(out + i + 8, _mm512_mul_pd(v1, vscalar));
    }
    for (; i + 8 <= count; i += 8) {
        __m512d v = _mm512_loadu_pd(in + i);
        _mm512_storeu_pd(out + i, _mm512_mul_pd(v, vscalar));
    }
#elif defined(AXON_HAS_AVX2)
    const __m256d vscalar = _mm256_set1_pd(scalar);
    for (; i + 16 <= count; i += 16) {
        __m256d v0 = _mm256_loadu_pd(in + i);
        __m256d v1 = _mm256_loadu_pd(in + i + 4);
        __m256d v2 = _mm256_loadu_pd(in + i + 8);
        __m256d v3 = _mm256_loadu_pd(in + i + 12);
        _mm256_storeu_pd(out + i, _mm256_mul_pd(v0, vscalar));
        _mm256_storeu_pd(out + i + 4, _mm256_mul_pd(v1, vscalar));
        _mm256_storeu_pd(out + i + 8, _mm256_mul_pd(v2, vscalar));
        _mm256_storeu_pd(out + i + 12, _mm256_mul_pd(v3, vscalar));
    }
    for (; i + 4 <= count; i += 4) {
        __m256d v = _mm256_loadu_pd(in + i);
        _mm256_storeu_pd(out + i, _mm256_mul_pd(v, vscalar));
    }
#elif defined(AXON_HAS_NEON)
    const float64x2_t vscalar = vdupq_n_f64(scalar);
    for (; i + 8 <= count; i += 8) {
        float64x2_t v0 = vld1q_f64(in + i);
        float64x2_t v1 = vld1q_f64(in + i + 2);
        float64x2_t v2 = vld1q_f64(in + i + 4);
        float64x2_t v3 = vld1q_f64(in + i + 6);
        vst1q_f64(out + i, vmulq_f64(v0, vscalar));
        vst1q_f64(out + i + 2, vmulq_f64(v1, vscalar));
        vst1q_f64(out + i + 4, vmulq_f64(v2, vscalar));
        vst1q_f64(out + i + 6, vmulq_f64(v3, vscalar));
    }
    for (; i + 2 <= count; i += 2) {
        float64x2_t v = vld1q_f64(in + i);
        vst1q_f64(out + i, vmulq_f64(v, vscalar));
    }
#endif

    for (; i + 4 <= count; i += 4) {
        out[i]     = in[i]     * scalar;
        out[i + 1] = in[i + 1] * scalar;
        out[i + 2] = in[i + 2] * scalar;
        out[i + 3] = in[i + 3] * scalar;
    }
    for (; i < count; ++i) {
        out[i] = in[i] * scalar;
    }
}

// ----------------------------------------------------------------------------
// Int32 (32-Bit Signed Integer) Compute Kernels
// ----------------------------------------------------------------------------

void add_kernel_i32(
    const int32_t* AXON_RESTRICT a,
    const int32_t* AXON_RESTRICT b,
    int32_t* AXON_RESTRICT out,
    size_t count
) noexcept {
    if (AXON_UNLIKELY(count == 0)) {
        return;
    }
    AXON_ASSERT(a != nullptr && b != nullptr && out != nullptr);

    size_t i = 0;

#if defined(AXON_HAS_AVX512F)
    for (; i + 32 <= count; i += 32) {
        __m512i va0 = _mm512_loadu_si512(reinterpret_cast<const void*>(a + i));
        __m512i vb0 = _mm512_loadu_si512(reinterpret_cast<const void*>(b + i));
        __m512i va1 = _mm512_loadu_si512(reinterpret_cast<const void*>(a + i + 16));
        __m512i vb1 = _mm512_loadu_si512(reinterpret_cast<const void*>(b + i + 16));
        _mm512_storeu_si512(reinterpret_cast<void*>(out + i), _mm512_add_epi32(va0, vb0));
        _mm512_storeu_si512(reinterpret_cast<void*>(out + i + 16), _mm512_add_epi32(va1, vb1));
    }
    for (; i + 16 <= count; i += 16) {
        __m512i va = _mm512_loadu_si512(reinterpret_cast<const void*>(a + i));
        __m512i vb = _mm512_loadu_si512(reinterpret_cast<const void*>(b + i));
        _mm512_storeu_si512(reinterpret_cast<void*>(out + i), _mm512_add_epi32(va, vb));
    }
#elif defined(AXON_HAS_AVX2)
    for (; i + 32 <= count; i += 32) {
        __m256i va0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        __m256i va1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 8));
        __m256i vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 8));
        __m256i va2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 16));
        __m256i vb2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 16));
        __m256i va3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 24));
        __m256i vb3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 24));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), _mm256_add_epi32(va0, vb0));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i + 8), _mm256_add_epi32(va1, vb1));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i + 16), _mm256_add_epi32(va2, vb2));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i + 24), _mm256_add_epi32(va3, vb3));
    }
    for (; i + 8 <= count; i += 8) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), _mm256_add_epi32(va, vb));
    }
#elif defined(AXON_HAS_NEON)
    for (; i + 16 <= count; i += 16) {
        int32x4_t va0 = vld1q_s32(a + i);
        int32x4_t vb0 = vld1q_s32(b + i);
        int32x4_t va1 = vld1q_s32(a + i + 4);
        int32x4_t vb1 = vld1q_s32(b + i + 4);
        int32x4_t va2 = vld1q_s32(a + i + 8);
        int32x4_t vb2 = vld1q_s32(b + i + 8);
        int32x4_t va3 = vld1q_s32(a + i + 12);
        int32x4_t vb3 = vld1q_s32(b + i + 12);
        vst1q_s32(out + i, vaddq_s32(va0, vb0));
        vst1q_s32(out + i + 4, vaddq_s32(va1, vb1));
        vst1q_s32(out + i + 8, vaddq_s32(va2, vb2));
        vst1q_s32(out + i + 12, vaddq_s32(va3, vb3));
    }
    for (; i + 4 <= count; i += 4) {
        int32x4_t va = vld1q_s32(a + i);
        int32x4_t vb = vld1q_s32(b + i);
        vst1q_s32(out + i, vaddq_s32(va, vb));
    }
#endif

    for (; i + 4 <= count; i += 4) {
        out[i]     = a[i]     + b[i];
        out[i + 1] = a[i + 1] + b[i + 1];
        out[i + 2] = a[i + 2] + b[i + 2];
        out[i + 3] = a[i + 3] + b[i + 3];
    }
    for (; i < count; ++i) {
        out[i] = a[i] + b[i];
    }
}

void sub_kernel_i32(
    const int32_t* AXON_RESTRICT a,
    const int32_t* AXON_RESTRICT b,
    int32_t* AXON_RESTRICT out,
    size_t count
) noexcept {
    if (AXON_UNLIKELY(count == 0)) {
        return;
    }
    AXON_ASSERT(a != nullptr && b != nullptr && out != nullptr);

    size_t i = 0;

#if defined(AXON_HAS_AVX512F)
    for (; i + 32 <= count; i += 32) {
        __m512i va0 = _mm512_loadu_si512(reinterpret_cast<const void*>(a + i));
        __m512i vb0 = _mm512_loadu_si512(reinterpret_cast<const void*>(b + i));
        __m512i va1 = _mm512_loadu_si512(reinterpret_cast<const void*>(a + i + 16));
        __m512i vb1 = _mm512_loadu_si512(reinterpret_cast<const void*>(b + i + 16));
        _mm512_storeu_si512(reinterpret_cast<void*>(out + i), _mm512_sub_epi32(va0, vb0));
        _mm512_storeu_si512(reinterpret_cast<void*>(out + i + 16), _mm512_sub_epi32(va1, vb1));
    }
    for (; i + 16 <= count; i += 16) {
        __m512i va = _mm512_loadu_si512(reinterpret_cast<const void*>(a + i));
        __m512i vb = _mm512_loadu_si512(reinterpret_cast<const void*>(b + i));
        _mm512_storeu_si512(reinterpret_cast<void*>(out + i), _mm512_sub_epi32(va, vb));
    }
#elif defined(AXON_HAS_AVX2)
    for (; i + 32 <= count; i += 32) {
        __m256i va0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        __m256i va1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 8));
        __m256i vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 8));
        __m256i va2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 16));
        __m256i vb2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 16));
        __m256i va3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 24));
        __m256i vb3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 24));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), _mm256_sub_epi32(va0, vb0));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i + 8), _mm256_sub_epi32(va1, vb1));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i + 16), _mm256_sub_epi32(va2, vb2));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i + 24), _mm256_sub_epi32(va3, vb3));
    }
    for (; i + 8 <= count; i += 8) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), _mm256_sub_epi32(va, vb));
    }
#elif defined(AXON_HAS_NEON)
    for (; i + 16 <= count; i += 16) {
        int32x4_t va0 = vld1q_s32(a + i);
        int32x4_t vb0 = vld1q_s32(b + i);
        int32x4_t va1 = vld1q_s32(a + i + 4);
        int32x4_t vb1 = vld1q_s32(b + i + 4);
        int32x4_t va2 = vld1q_s32(a + i + 8);
        int32x4_t vb2 = vld1q_s32(b + i + 8);
        int32x4_t va3 = vld1q_s32(a + i + 12);
        int32x4_t vb3 = vld1q_s32(b + i + 12);
        vst1q_s32(out + i, vsubq_s32(va0, vb0));
        vst1q_s32(out + i + 4, vsubq_s32(va1, vb1));
        vst1q_s32(out + i + 8, vsubq_s32(va2, vb2));
        vst1q_s32(out + i + 12, vsubq_s32(va3, vb3));
    }
    for (; i + 4 <= count; i += 4) {
        int32x4_t va = vld1q_s32(a + i);
        int32x4_t vb = vld1q_s32(b + i);
        vst1q_s32(out + i, vsubq_s32(va, vb));
    }
#endif

    for (; i + 4 <= count; i += 4) {
        out[i]     = a[i]     - b[i];
        out[i + 1] = a[i + 1] - b[i + 1];
        out[i + 2] = a[i + 2] - b[i + 2];
        out[i + 3] = a[i + 3] - b[i + 3];
    }
    for (; i < count; ++i) {
        out[i] = a[i] - b[i];
    }
}

void mul_kernel_i32(
    const int32_t* AXON_RESTRICT a,
    const int32_t* AXON_RESTRICT b,
    int32_t* AXON_RESTRICT out,
    size_t count
) noexcept {
    if (AXON_UNLIKELY(count == 0)) {
        return;
    }
    AXON_ASSERT(a != nullptr && b != nullptr && out != nullptr);

    size_t i = 0;

#if defined(AXON_HAS_AVX512F)
    for (; i + 32 <= count; i += 32) {
        __m512i va0 = _mm512_loadu_si512(reinterpret_cast<const void*>(a + i));
        __m512i vb0 = _mm512_loadu_si512(reinterpret_cast<const void*>(b + i));
        __m512i va1 = _mm512_loadu_si512(reinterpret_cast<const void*>(a + i + 16));
        __m512i vb1 = _mm512_loadu_si512(reinterpret_cast<const void*>(b + i + 16));
        _mm512_storeu_si512(reinterpret_cast<void*>(out + i), _mm512_mullo_epi32(va0, vb0));
        _mm512_storeu_si512(reinterpret_cast<void*>(out + i + 16), _mm512_mullo_epi32(va1, vb1));
    }
    for (; i + 16 <= count; i += 16) {
        __m512i va = _mm512_loadu_si512(reinterpret_cast<const void*>(a + i));
        __m512i vb = _mm512_loadu_si512(reinterpret_cast<const void*>(b + i));
        _mm512_storeu_si512(reinterpret_cast<void*>(out + i), _mm512_mullo_epi32(va, vb));
    }
#elif defined(AXON_HAS_AVX2)
    for (; i + 32 <= count; i += 32) {
        __m256i va0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        __m256i va1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 8));
        __m256i vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 8));
        __m256i va2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 16));
        __m256i vb2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 16));
        __m256i va3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 24));
        __m256i vb3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 24));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), _mm256_mullo_epi32(va0, vb0));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i + 8), _mm256_mullo_epi32(va1, vb1));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i + 16), _mm256_mullo_epi32(va2, vb2));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i + 24), _mm256_mullo_epi32(va3, vb3));
    }
    for (; i + 8 <= count; i += 8) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), _mm256_mullo_epi32(va, vb));
    }
#elif defined(AXON_HAS_NEON)
    for (; i + 16 <= count; i += 16) {
        int32x4_t va0 = vld1q_s32(a + i);
        int32x4_t vb0 = vld1q_s32(b + i);
        int32x4_t va1 = vld1q_s32(a + i + 4);
        int32x4_t vb1 = vld1q_s32(b + i + 4);
        int32x4_t va2 = vld1q_s32(a + i + 8);
        int32x4_t vb2 = vld1q_s32(b + i + 8);
        int32x4_t va3 = vld1q_s32(a + i + 12);
        int32x4_t vb3 = vld1q_s32(b + i + 12);
        vst1q_s32(out + i, vmulq_s32(va0, vb0));
        vst1q_s32(out + i + 4, vmulq_s32(va1, vb1));
        vst1q_s32(out + i + 8, vmulq_s32(va2, vb2));
        vst1q_s32(out + i + 12, vmulq_s32(va3, vb3));
    }
    for (; i + 4 <= count; i += 4) {
        int32x4_t va = vld1q_s32(a + i);
        int32x4_t vb = vld1q_s32(b + i);
        vst1q_s32(out + i, vmulq_s32(va, vb));
    }
#endif

    for (; i + 4 <= count; i += 4) {
        out[i]     = a[i]     * b[i];
        out[i + 1] = a[i + 1] * b[i + 1];
        out[i + 2] = a[i + 2] * b[i + 2];
        out[i + 3] = a[i + 3] * b[i + 3];
    }
    for (; i < count; ++i) {
        out[i] = a[i] * b[i];
    }
}

void div_kernel_i32(
    const int32_t* AXON_RESTRICT a,
    const int32_t* AXON_RESTRICT b,
    int32_t* AXON_RESTRICT out,
    size_t count
) noexcept {
    if (AXON_UNLIKELY(count == 0)) {
        return;
    }
    AXON_ASSERT(a != nullptr && b != nullptr && out != nullptr);

    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        AXON_ASSERT(b[i] != 0 && b[i + 1] != 0 && b[i + 2] != 0 && b[i + 3] != 0);
        out[i]     = a[i]     / b[i];
        out[i + 1] = a[i + 1] / b[i + 1];
        out[i + 2] = a[i + 2] / b[i + 2];
        out[i + 3] = a[i + 3] / b[i + 3];
    }
    for (; i < count; ++i) {
        AXON_ASSERT(b[i] != 0);
        out[i] = a[i] / b[i];
    }
}

void relu_kernel_i32(
    const int32_t* AXON_RESTRICT in,
    int32_t* AXON_RESTRICT out,
    size_t count
) noexcept {
    if (AXON_UNLIKELY(count == 0)) {
        return;
    }
    AXON_ASSERT(in != nullptr && out != nullptr);

    size_t i = 0;

#if defined(AXON_HAS_AVX512F)
    const __m512i vzero = _mm512_setzero_si512();
    for (; i + 32 <= count; i += 32) {
        __m512i v0 = _mm512_loadu_si512(reinterpret_cast<const void*>(in + i));
        __m512i v1 = _mm512_loadu_si512(reinterpret_cast<const void*>(in + i + 16));
        _mm512_storeu_si512(reinterpret_cast<void*>(out + i), _mm512_max_epi32(v0, vzero));
        _mm512_storeu_si512(reinterpret_cast<void*>(out + i + 16), _mm512_max_epi32(v1, vzero));
    }
    for (; i + 16 <= count; i += 16) {
        __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(in + i));
        _mm512_storeu_si512(reinterpret_cast<void*>(out + i), _mm512_max_epi32(v, vzero));
    }
#elif defined(AXON_HAS_AVX2)
    const __m256i vzero = _mm256_setzero_si256();
    for (; i + 32 <= count; i += 32) {
        __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i));
        __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i + 8));
        __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i + 16));
        __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i + 24));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), _mm256_max_epi32(v0, vzero));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i + 8), _mm256_max_epi32(v1, vzero));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i + 16), _mm256_max_epi32(v2, vzero));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i + 24), _mm256_max_epi32(v3, vzero));
    }
    for (; i + 8 <= count; i += 8) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), _mm256_max_epi32(v, vzero));
    }
#elif defined(AXON_HAS_NEON)
    const int32x4_t vzero = vdupq_n_s32(0);
    for (; i + 16 <= count; i += 16) {
        int32x4_t v0 = vld1q_s32(in + i);
        int32x4_t v1 = vld1q_s32(in + i + 4);
        int32x4_t v2 = vld1q_s32(in + i + 8);
        int32x4_t v3 = vld1q_s32(in + i + 12);
        vst1q_s32(out + i, vmaxq_s32(v0, vzero));
        vst1q_s32(out + i + 4, vmaxq_s32(v1, vzero));
        vst1q_s32(out + i + 8, vmaxq_s32(v2, vzero));
        vst1q_s32(out + i + 12, vmaxq_s32(v3, vzero));
    }
    for (; i + 4 <= count; i += 4) {
        int32x4_t v = vld1q_s32(in + i);
        vst1q_s32(out + i, vmaxq_s32(v, vzero));
    }
#endif

    for (; i + 4 <= count; i += 4) {
        const int32_t v0 = in[i];
        const int32_t v1 = in[i + 1];
        const int32_t v2 = in[i + 2];
        const int32_t v3 = in[i + 3];
        out[i]     = (v0 > 0) ? v0 : 0;
        out[i + 1] = (v1 > 0) ? v1 : 0;
        out[i + 2] = (v2 > 0) ? v2 : 0;
        out[i + 3] = (v3 > 0) ? v3 : 0;
    }
    for (; i < count; ++i) {
        const int32_t v = in[i];
        out[i] = (v > 0) ? v : 0;
    }
}

void fill_kernel_i32(
    int32_t* AXON_RESTRICT out,
    int32_t value,
    size_t count
) noexcept {
    if (AXON_UNLIKELY(count == 0)) {
        return;
    }
    AXON_ASSERT(out != nullptr);

    size_t i = 0;

#if defined(AXON_HAS_AVX512F)
    const __m512i vval = _mm512_set1_epi32(value);
    for (; i + 32 <= count; i += 32) {
        _mm512_storeu_si512(reinterpret_cast<void*>(out + i), vval);
        _mm512_storeu_si512(reinterpret_cast<void*>(out + i + 16), vval);
    }
    for (; i + 16 <= count; i += 16) {
        _mm512_storeu_si512(reinterpret_cast<void*>(out + i), vval);
    }
#elif defined(AXON_HAS_AVX2)
    const __m256i vval = _mm256_set1_epi32(value);
    for (; i + 32 <= count; i += 32) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), vval);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i + 8), vval);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i + 16), vval);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i + 24), vval);
    }
    for (; i + 8 <= count; i += 8) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), vval);
    }
#elif defined(AXON_HAS_NEON)
    const int32x4_t vval = vdupq_n_s32(value);
    for (; i + 16 <= count; i += 16) {
        vst1q_s32(out + i, vval);
        vst1q_s32(out + i + 4, vval);
        vst1q_s32(out + i + 8, vval);
        vst1q_s32(out + i + 12, vval);
    }
    for (; i + 4 <= count; i += 4) {
        vst1q_s32(out + i, vval);
    }
#endif

    for (; i + 4 <= count; i += 4) {
        out[i]     = value;
        out[i + 1] = value;
        out[i + 2] = value;
        out[i + 3] = value;
    }
    for (; i < count; ++i) {
        out[i] = value;
    }
}

void scale_kernel_i32(
    const int32_t* AXON_RESTRICT in,
    int32_t scalar,
    int32_t* AXON_RESTRICT out,
    size_t count
) noexcept {
    if (AXON_UNLIKELY(count == 0)) {
        return;
    }
    AXON_ASSERT(in != nullptr && out != nullptr);

    size_t i = 0;

#if defined(AXON_HAS_AVX512F)
    const __m512i vscalar = _mm512_set1_epi32(scalar);
    for (; i + 32 <= count; i += 32) {
        __m512i v0 = _mm512_loadu_si512(reinterpret_cast<const void*>(in + i));
        __m512i v1 = _mm512_loadu_si512(reinterpret_cast<const void*>(in + i + 16));
        _mm512_storeu_si512(reinterpret_cast<void*>(out + i), _mm512_mullo_epi32(v0, vscalar));
        _mm512_storeu_si512(reinterpret_cast<void*>(out + i + 16), _mm512_mullo_epi32(v1, vscalar));
    }
    for (; i + 16 <= count; i += 16) {
        __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(in + i));
        _mm512_storeu_si512(reinterpret_cast<void*>(out + i), _mm512_mullo_epi32(v, vscalar));
    }
#elif defined(AXON_HAS_AVX2)
    const __m256i vscalar = _mm256_set1_epi32(scalar);
    for (; i + 32 <= count; i += 32) {
        __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i));
        __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i + 8));
        __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i + 16));
        __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i + 24));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), _mm256_mullo_epi32(v0, vscalar));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i + 8), _mm256_mullo_epi32(v1, vscalar));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i + 16), _mm256_mullo_epi32(v2, vscalar));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i + 24), _mm256_mullo_epi32(v3, vscalar));
    }
    for (; i + 8 <= count; i += 8) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), _mm256_mullo_epi32(v, vscalar));
    }
#elif defined(AXON_HAS_NEON)
    const int32x4_t vscalar = vdupq_n_s32(scalar);
    for (; i + 16 <= count; i += 16) {
        int32x4_t v0 = vld1q_s32(in + i);
        int32x4_t v1 = vld1q_s32(in + i + 4);
        int32x4_t v2 = vld1q_s32(in + i + 8);
        int32x4_t v3 = vld1q_s32(in + i + 12);
        vst1q_s32(out + i, vmulq_s32(v0, vscalar));
        vst1q_s32(out + i + 4, vmulq_s32(v1, vscalar));
        vst1q_s32(out + i + 8, vmulq_s32(v2, vscalar));
        vst1q_s32(out + i + 12, vmulq_s32(v3, vscalar));
    }
    for (; i + 4 <= count; i += 4) {
        int32x4_t v = vld1q_s32(in + i);
        vst1q_s32(out + i, vmulq_s32(v, vscalar));
    }
#endif

    for (; i + 4 <= count; i += 4) {
        out[i]     = in[i]     * scalar;
        out[i + 1] = in[i + 1] * scalar;
        out[i + 2] = in[i + 2] * scalar;
        out[i + 3] = in[i + 3] * scalar;
    }
    for (; i < count; ++i) {
        out[i] = in[i] * scalar;
    }
}

} // namespace kernels

// ============================================================================
// 2. EXPLICIT OUTPUT (ZERO-ALLOCATION) TENSOR OPERATIONS
// ============================================================================

void add_out(
    const axon::tensor::Tensor& a,
    const axon::tensor::Tensor& b,
    axon::tensor::Tensor& out
) noexcept {
    detail::validate_binary_op(a, b, out);
    const size_t count = out.num_elements();

    switch (out.dtype()) {
        case axon::core::DType::Float32:
            kernels::add_kernel_f32(a.data<float>(), b.data<float>(), out.data<float>(), count);
            break;
        case axon::core::DType::Float64:
            kernels::add_kernel_f64(a.data<double>(), b.data<double>(), out.data<double>(), count);
            break;
        case axon::core::DType::Int32:
            kernels::add_kernel_i32(a.data<int32_t>(), b.data<int32_t>(), out.data<int32_t>(), count);
            break;
        default:
            AXON_DEBUG_TRAP();
            break;
    }
}

void sub_out(
    const axon::tensor::Tensor& a,
    const axon::tensor::Tensor& b,
    axon::tensor::Tensor& out
) noexcept {
    detail::validate_binary_op(a, b, out);
    const size_t count = out.num_elements();

    switch (out.dtype()) {
        case axon::core::DType::Float32:
            kernels::sub_kernel_f32(a.data<float>(), b.data<float>(), out.data<float>(), count);
            break;
        case axon::core::DType::Float64:
            kernels::sub_kernel_f64(a.data<double>(), b.data<double>(), out.data<double>(), count);
            break;
        case axon::core::DType::Int32:
            kernels::sub_kernel_i32(a.data<int32_t>(), b.data<int32_t>(), out.data<int32_t>(), count);
            break;
        default:
            AXON_DEBUG_TRAP();
            break;
    }
}

void mul_out(
    const axon::tensor::Tensor& a,
    const axon::tensor::Tensor& b,
    axon::tensor::Tensor& out
) noexcept {
    detail::validate_binary_op(a, b, out);
    const size_t count = out.num_elements();

    switch (out.dtype()) {
        case axon::core::DType::Float32:
            kernels::mul_kernel_f32(a.data<float>(), b.data<float>(), out.data<float>(), count);
            break;
        case axon::core::DType::Float64:
            kernels::mul_kernel_f64(a.data<double>(), b.data<double>(), out.data<double>(), count);
            break;
        case axon::core::DType::Int32:
            kernels::mul_kernel_i32(a.data<int32_t>(), b.data<int32_t>(), out.data<int32_t>(), count);
            break;
        default:
            AXON_DEBUG_TRAP();
            break;
    }
}

void div_out(
    const axon::tensor::Tensor& a,
    const axon::tensor::Tensor& b,
    axon::tensor::Tensor& out
) noexcept {
    detail::validate_binary_op(a, b, out);
    const size_t count = out.num_elements();

    switch (out.dtype()) {
        case axon::core::DType::Float32:
            kernels::div_kernel_f32(a.data<float>(), b.data<float>(), out.data<float>(), count);
            break;
        case axon::core::DType::Float64:
            kernels::div_kernel_f64(a.data<double>(), b.data<double>(), out.data<double>(), count);
            break;
        case axon::core::DType::Int32:
            kernels::div_kernel_i32(a.data<int32_t>(), b.data<int32_t>(), out.data<int32_t>(), count);
            break;
        default:
            AXON_DEBUG_TRAP();
            break;
    }
}

void relu_out(
    const axon::tensor::Tensor& in,
    axon::tensor::Tensor& out
) noexcept {
    detail::validate_unary_op(in, out);
    const size_t count = out.num_elements();

    switch (out.dtype()) {
        case axon::core::DType::Float32:
            kernels::relu_kernel_f32(in.data<float>(), out.data<float>(), count);
            break;
        case axon::core::DType::Float64:
            kernels::relu_kernel_f64(in.data<double>(), out.data<double>(), count);
            break;
        case axon::core::DType::Int32:
            kernels::relu_kernel_i32(in.data<int32_t>(), out.data<int32_t>(), count);
            break;
        default:
            AXON_DEBUG_TRAP();
            break;
    }
}

void scale_out(
    const axon::tensor::Tensor& in,
    float scalar,
    axon::tensor::Tensor& out
) noexcept {
    detail::validate_unary_op(in, out);
    const size_t count = out.num_elements();

    switch (out.dtype()) {
        case axon::core::DType::Float32:
            kernels::scale_kernel_f32(in.data<float>(), scalar, out.data<float>(), count);
            break;
        case axon::core::DType::Float64:
            kernels::scale_kernel_f64(in.data<double>(), static_cast<double>(scalar), out.data<double>(), count);
            break;
        case axon::core::DType::Int32:
            kernels::scale_kernel_i32(in.data<int32_t>(), static_cast<int32_t>(scalar), out.data<int32_t>(), count);
            break;
        default:
            AXON_DEBUG_TRAP();
            break;
    }
}

// ============================================================================
// 3. FUNCTIONAL ALLOCATING TENSOR OPERATIONS
// ============================================================================

axon::tensor::Tensor add(
    const axon::tensor::Tensor& a,
    const axon::tensor::Tensor& b
) noexcept {
    axon::tensor::Tensor out(a.dims(), a.ndim(), a.dtype(), a.device());
    add_out(a, b, out);
    return out;
}

axon::tensor::Tensor sub(
    const axon::tensor::Tensor& a,
    const axon::tensor::Tensor& b
) noexcept {
    axon::tensor::Tensor out(a.dims(), a.ndim(), a.dtype(), a.device());
    sub_out(a, b, out);
    return out;
}

axon::tensor::Tensor mul(
    const axon::tensor::Tensor& a,
    const axon::tensor::Tensor& b
) noexcept {
    axon::tensor::Tensor out(a.dims(), a.ndim(), a.dtype(), a.device());
    mul_out(a, b, out);
    return out;
}

axon::tensor::Tensor div(
    const axon::tensor::Tensor& a,
    const axon::tensor::Tensor& b
) noexcept {
    axon::tensor::Tensor out(a.dims(), a.ndim(), a.dtype(), a.device());
    div_out(a, b, out);
    return out;
}

axon::tensor::Tensor relu(
    const axon::tensor::Tensor& in
) noexcept {
    axon::tensor::Tensor out(in.dims(), in.ndim(), in.dtype(), in.device());
    relu_out(in, out);
    return out;
}

axon::tensor::Tensor scale(
    const axon::tensor::Tensor& in,
    float scalar
) noexcept {
    axon::tensor::Tensor out(in.dims(), in.ndim(), in.dtype(), in.device());
    scale_out(in, scalar, out);
    return out;
}

// ============================================================================
// 4. IN-PLACE TENSOR MUTATION OPERATIONS (WARNING-FREE ALIASING PATHS)
// ============================================================================

void add_(
    axon::tensor::Tensor& self,
    const axon::tensor::Tensor& other
) noexcept {
    detail::validate_inplace_binary_op(self, other);
    const size_t count = self.num_elements();
    if (AXON_UNLIKELY(count == 0)) {
        return;
    }

    switch (self.dtype()) {
        case axon::core::DType::Float32: {
            float* self_ptr = self.data<float>();
            const float* other_ptr = other.data<float>();
            AXON_ASSERT(self_ptr != nullptr && other_ptr != nullptr);
            size_t i = 0;
#if defined(AXON_HAS_AVX512F)
            for (; i + 16 <= count; i += 16) {
                __m512 v0 = _mm512_loadu_ps(self_ptr + i);
                __m512 v1 = _mm512_loadu_ps(other_ptr + i);
                _mm512_storeu_ps(self_ptr + i, _mm512_add_ps(v0, v1));
            }
#elif defined(AXON_HAS_AVX2)
            for (; i + 8 <= count; i += 8) {
                __m256 v0 = _mm256_loadu_ps(self_ptr + i);
                __m256 v1 = _mm256_loadu_ps(other_ptr + i);
                _mm256_storeu_ps(self_ptr + i, _mm256_add_ps(v0, v1));
            }
#elif defined(AXON_HAS_NEON)
            for (; i + 4 <= count; i += 4) {
                float32x4_t v0 = vld1q_f32(self_ptr + i);
                float32x4_t v1 = vld1q_f32(other_ptr + i);
                vst1q_f32(self_ptr + i, vaddq_f32(v0, v1));
            }
#endif
            for (; i < count; ++i) {
                self_ptr[i] += other_ptr[i];
            }
            break;
        }
        case axon::core::DType::Float64: {
            double* self_ptr = self.data<double>();
            const double* other_ptr = other.data<double>();
            AXON_ASSERT(self_ptr != nullptr && other_ptr != nullptr);
            size_t i = 0;
#if defined(AXON_HAS_AVX512F)
            for (; i + 8 <= count; i += 8) {
                __m512d v0 = _mm512_loadu_pd(self_ptr + i);
                __m512d v1 = _mm512_loadu_pd(other_ptr + i);
                _mm512_storeu_pd(self_ptr + i, _mm512_add_pd(v0, v1));
            }
#elif defined(AXON_HAS_AVX2)
            for (; i + 4 <= count; i += 4) {
                __m256d v0 = _mm256_loadu_pd(self_ptr + i);
                __m256d v1 = _mm256_loadu_pd(other_ptr + i);
                _mm256_storeu_pd(self_ptr + i, _mm256_add_pd(v0, v1));
            }
#elif defined(AXON_HAS_NEON)
            for (; i + 2 <= count; i += 2) {
                float64x2_t v0 = vld1q_f64(self_ptr + i);
                float64x2_t v1 = vld1q_f64(other_ptr + i);
                vst1q_f64(self_ptr + i, vaddq_f64(v0, v1));
            }
#endif
            for (; i < count; ++i) {
                self_ptr[i] += other_ptr[i];
            }
            break;
        }
        case axon::core::DType::Int32: {
            int32_t* self_ptr = self.data<int32_t>();
            const int32_t* other_ptr = other.data<int32_t>();
            AXON_ASSERT(self_ptr != nullptr && other_ptr != nullptr);
            size_t i = 0;
#if defined(AXON_HAS_AVX512F)
            for (; i + 16 <= count; i += 16) {
                __m512i v0 = _mm512_loadu_si512(reinterpret_cast<const void*>(self_ptr + i));
                __m512i v1 = _mm512_loadu_si512(reinterpret_cast<const void*>(other_ptr + i));
                _mm512_storeu_si512(reinterpret_cast<void*>(self_ptr + i), _mm512_add_epi32(v0, v1));
            }
#elif defined(AXON_HAS_AVX2)
            for (; i + 8 <= count; i += 8) {
                __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(self_ptr + i));
                __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_ptr + i));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(self_ptr + i), _mm256_add_epi32(v0, v1));
            }
#elif defined(AXON_HAS_NEON)
            for (; i + 4 <= count; i += 4) {
                int32x4_t v0 = vld1q_s32(self_ptr + i);
                int32x4_t v1 = vld1q_s32(other_ptr + i);
                vst1q_s32(self_ptr + i, vaddq_s32(v0, v1));
            }
#endif
            for (; i < count; ++i) {
                self_ptr[i] += other_ptr[i];
            }
            break;
        }
        default:
            AXON_DEBUG_TRAP();
            break;
    }
}

void sub_(
    axon::tensor::Tensor& self,
    const axon::tensor::Tensor& other
) noexcept {
    detail::validate_inplace_binary_op(self, other);
    const size_t count = self.num_elements();
    if (AXON_UNLIKELY(count == 0)) {
        return;
    }

    switch (self.dtype()) {
        case axon::core::DType::Float32: {
            float* self_ptr = self.data<float>();
            const float* other_ptr = other.data<float>();
            AXON_ASSERT(self_ptr != nullptr && other_ptr != nullptr);
            size_t i = 0;
#if defined(AXON_HAS_AVX512F)
            for (; i + 16 <= count; i += 16) {
                __m512 v0 = _mm512_loadu_ps(self_ptr + i);
                __m512 v1 = _mm512_loadu_ps(other_ptr + i);
                _mm512_storeu_ps(self_ptr + i, _mm512_sub_ps(v0, v1));
            }
#elif defined(AXON_HAS_AVX2)
            for (; i + 8 <= count; i += 8) {
                __m256 v0 = _mm256_loadu_ps(self_ptr + i);
                __m256 v1 = _mm256_loadu_ps(other_ptr + i);
                _mm256_storeu_ps(self_ptr + i, _mm256_sub_ps(v0, v1));
            }
#elif defined(AXON_HAS_NEON)
            for (; i + 4 <= count; i += 4) {
                float32x4_t v0 = vld1q_f32(self_ptr + i);
                float32x4_t v1 = vld1q_f32(other_ptr + i);
                vst1q_f32(self_ptr + i, vsubq_f32(v0, v1));
            }
#endif
            for (; i < count; ++i) {
                self_ptr[i] -= other_ptr[i];
            }
            break;
        }
        case axon::core::DType::Float64: {
            double* self_ptr = self.data<double>();
            const double* other_ptr = other.data<double>();
            AXON_ASSERT(self_ptr != nullptr && other_ptr != nullptr);
            size_t i = 0;
#if defined(AXON_HAS_AVX512F)
            for (; i + 8 <= count; i += 8) {
                __m512d v0 = _mm512_loadu_pd(self_ptr + i);
                __m512d v1 = _mm512_loadu_pd(other_ptr + i);
                _mm512_storeu_pd(self_ptr + i, _mm512_sub_pd(v0, v1));
            }
#elif defined(AXON_HAS_AVX2)
            for (; i + 4 <= count; i += 4) {
                __m256d v0 = _mm256_loadu_pd(self_ptr + i);
                __m256d v1 = _mm256_loadu_pd(other_ptr + i);
                _mm256_storeu_pd(self_ptr + i, _mm256_sub_pd(v0, v1));
            }
#elif defined(AXON_HAS_NEON)
            for (; i + 2 <= count; i += 2) {
                float64x2_t v0 = vld1q_f64(self_ptr + i);
                float64x2_t v1 = vld1q_f64(other_ptr + i);
                vst1q_f64(self_ptr + i, vsubq_f64(v0, v1));
            }
#endif
            for (; i < count; ++i) {
                self_ptr[i] -= other_ptr[i];
            }
            break;
        }
        case axon::core::DType::Int32: {
            int32_t* self_ptr = self.data<int32_t>();
            const int32_t* other_ptr = other.data<int32_t>();
            AXON_ASSERT(self_ptr != nullptr && other_ptr != nullptr);
            size_t i = 0;
#if defined(AXON_HAS_AVX512F)
            for (; i + 16 <= count; i += 16) {
                __m512i v0 = _mm512_loadu_si512(reinterpret_cast<const void*>(self_ptr + i));
                __m512i v1 = _mm512_loadu_si512(reinterpret_cast<const void*>(other_ptr + i));
                _mm512_storeu_si512(reinterpret_cast<void*>(self_ptr + i), _mm512_sub_epi32(v0, v1));
            }
#elif defined(AXON_HAS_AVX2)
            for (; i + 8 <= count; i += 8) {
                __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(self_ptr + i));
                __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_ptr + i));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(self_ptr + i), _mm256_sub_epi32(v0, v1));
            }
#elif defined(AXON_HAS_NEON)
            for (; i + 4 <= count; i += 4) {
                int32x4_t v0 = vld1q_s32(self_ptr + i);
                int32x4_t v1 = vld1q_s32(other_ptr + i);
                vst1q_s32(self_ptr + i, vsubq_s32(v0, v1));
            }
#endif
            for (; i < count; ++i) {
                self_ptr[i] -= other_ptr[i];
            }
            break;
        }
        default:
            AXON_DEBUG_TRAP();
            break;
    }
}

void mul_(
    axon::tensor::Tensor& self,
    const axon::tensor::Tensor& other
) noexcept {
    detail::validate_inplace_binary_op(self, other);
    const size_t count = self.num_elements();
    if (AXON_UNLIKELY(count == 0)) {
        return;
    }

    switch (self.dtype()) {
        case axon::core::DType::Float32: {
            float* self_ptr = self.data<float>();
            const float* other_ptr = other.data<float>();
            AXON_ASSERT(self_ptr != nullptr && other_ptr != nullptr);
            size_t i = 0;
#if defined(AXON_HAS_AVX512F)
            for (; i + 16 <= count; i += 16) {
                __m512 v0 = _mm512_loadu_ps(self_ptr + i);
                __m512 v1 = _mm512_loadu_ps(other_ptr + i);
                _mm512_storeu_ps(self_ptr + i, _mm512_mul_ps(v0, v1));
            }
#elif defined(AXON_HAS_AVX2)
            for (; i + 8 <= count; i += 8) {
                __m256 v0 = _mm256_loadu_ps(self_ptr + i);
                __m256 v1 = _mm256_loadu_ps(other_ptr + i);
                _mm256_storeu_ps(self_ptr + i, _mm256_mul_ps(v0, v1));
            }
#elif defined(AXON_HAS_NEON)
            for (; i + 4 <= count; i += 4) {
                float32x4_t v0 = vld1q_f32(self_ptr + i);
                float32x4_t v1 = vld1q_f32(other_ptr + i);
                vst1q_f32(self_ptr + i, vmulq_f32(v0, v1));
            }
#endif
            for (; i < count; ++i) {
                self_ptr[i] *= other_ptr[i];
            }
            break;
        }
        case axon::core::DType::Float64: {
            double* self_ptr = self.data<double>();
            const double* other_ptr = other.data<double>();
            AXON_ASSERT(self_ptr != nullptr && other_ptr != nullptr);
            size_t i = 0;
#if defined(AXON_HAS_AVX512F)
            for (; i + 8 <= count; i += 8) {
                __m512d v0 = _mm512_loadu_pd(self_ptr + i);
                __m512d v1 = _mm512_loadu_pd(other_ptr + i);
                _mm512_storeu_pd(self_ptr + i, _mm512_mul_pd(v0, v1));
            }
#elif defined(AXON_HAS_AVX2)
            for (; i + 4 <= count; i += 4) {
                __m256d v0 = _mm256_loadu_pd(self_ptr + i);
                __m256d v1 = _mm256_loadu_pd(other_ptr + i);
                _mm256_storeu_pd(self_ptr + i, _mm256_mul_pd(v0, v1));
            }
#elif defined(AXON_HAS_NEON)
            for (; i + 2 <= count; i += 2) {
                float64x2_t v0 = vld1q_f64(self_ptr + i);
                float64x2_t v1 = vld1q_f64(other_ptr + i);
                vst1q_f64(self_ptr + i, vmulq_f64(v0, v1));
            }
#endif
            for (; i < count; ++i) {
                self_ptr[i] *= other_ptr[i];
            }
            break;
        }
        case axon::core::DType::Int32: {
            int32_t* self_ptr = self.data<int32_t>();
            const int32_t* other_ptr = other.data<int32_t>();
            AXON_ASSERT(self_ptr != nullptr && other_ptr != nullptr);
            size_t i = 0;
#if defined(AXON_HAS_AVX512F)
            for (; i + 16 <= count; i += 16) {
                __m512i v0 = _mm512_loadu_si512(reinterpret_cast<const void*>(self_ptr + i));
                __m512i v1 = _mm512_loadu_si512(reinterpret_cast<const void*>(other_ptr + i));
                _mm512_storeu_si512(reinterpret_cast<void*>(self_ptr + i), _mm512_mullo_epi32(v0, v1));
            }
#elif defined(AXON_HAS_AVX2)
            for (; i + 8 <= count; i += 8) {
                __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(self_ptr + i));
                __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(other_ptr + i));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(self_ptr + i), _mm256_mullo_epi32(v0, v1));
            }
#elif defined(AXON_HAS_NEON)
            for (; i + 4 <= count; i += 4) {
                int32x4_t v0 = vld1q_s32(self_ptr + i);
                int32x4_t v1 = vld1q_s32(other_ptr + i);
                vst1q_s32(self_ptr + i, vmulq_s32(v0, v1));
            }
#endif
            for (; i < count; ++i) {
                self_ptr[i] *= other_ptr[i];
            }
            break;
        }
        default:
            AXON_DEBUG_TRAP();
            break;
    }
}

void div_(
    axon::tensor::Tensor& self,
    const axon::tensor::Tensor& other
) noexcept {
    detail::validate_inplace_binary_op(self, other);
    const size_t count = self.num_elements();
    if (AXON_UNLIKELY(count == 0)) {
        return;
    }

    switch (self.dtype()) {
        case axon::core::DType::Float32: {
            float* self_ptr = self.data<float>();
            const float* other_ptr = other.data<float>();
            AXON_ASSERT(self_ptr != nullptr && other_ptr != nullptr);
            size_t i = 0;
#if defined(AXON_HAS_AVX512F)
            for (; i + 16 <= count; i += 16) {
                __m512 v0 = _mm512_loadu_ps(self_ptr + i);
                __m512 v1 = _mm512_loadu_ps(other_ptr + i);
                _mm512_storeu_ps(self_ptr + i, _mm512_div_ps(v0, v1));
            }
#elif defined(AXON_HAS_AVX2)
            for (; i + 8 <= count; i += 8) {
                __m256 v0 = _mm256_loadu_ps(self_ptr + i);
                __m256 v1 = _mm256_loadu_ps(other_ptr + i);
                _mm256_storeu_ps(self_ptr + i, _mm256_div_ps(v0, v1));
            }
#elif defined(AXON_HAS_NEON)
            for (; i + 4 <= count; i += 4) {
                float32x4_t v0 = vld1q_f32(self_ptr + i);
                float32x4_t v1 = vld1q_f32(other_ptr + i);
                vst1q_f32(self_ptr + i, vdivq_f32(v0, v1));
            }
#endif
            for (; i < count; ++i) {
                self_ptr[i] /= other_ptr[i];
            }
            break;
        }
        case axon::core::DType::Float64: {
            double* self_ptr = self.data<double>();
            const double* other_ptr = other.data<double>();
            AXON_ASSERT(self_ptr != nullptr && other_ptr != nullptr);
            size_t i = 0;
#if defined(AXON_HAS_AVX512F)
            for (; i + 8 <= count; i += 8) {
                __m512d v0 = _mm512_loadu_pd(self_ptr + i);
                __m512d v1 = _mm512_loadu_pd(other_ptr + i);
                _mm512_storeu_pd(self_ptr + i, _mm512_div_pd(v0, v1));
            }
#elif defined(AXON_HAS_AVX2)
            for (; i + 4 <= count; i += 4) {
                __m256d v0 = _mm256_loadu_pd(self_ptr + i);
                __m256d v1 = _mm256_loadu_pd(other_ptr + i);
                _mm256_storeu_pd(self_ptr + i, _mm256_div_pd(v0, v1));
            }
#elif defined(AXON_HAS_NEON)
            for (; i + 2 <= count; i += 2) {
                float64x2_t v0 = vld1q_f64(self_ptr + i);
                float64x2_t v1 = vld1q_f64(other_ptr + i);
                vst1q_f64(self_ptr + i, vdivq_f64(v0, v1));
            }
#endif
            for (; i < count; ++i) {
                self_ptr[i] /= other_ptr[i];
            }
            break;
        }
        case axon::core::DType::Int32: {
            int32_t* self_ptr = self.data<int32_t>();
            const int32_t* other_ptr = other.data<int32_t>();
            AXON_ASSERT(self_ptr != nullptr && other_ptr != nullptr);
            size_t i = 0;
            for (; i < count; ++i) {
                AXON_ASSERT(other_ptr[i] != 0);
                self_ptr[i] /= other_ptr[i];
            }
            break;
        }
        default:
            AXON_DEBUG_TRAP();
            break;
    }
}

void relu_(
    axon::tensor::Tensor& self
) noexcept {
    detail::validate_inplace_unary_op(self);
    const size_t count = self.num_elements();
    if (AXON_UNLIKELY(count == 0)) {
        return;
    }

    switch (self.dtype()) {
        case axon::core::DType::Float32: {
            float* ptr = self.data<float>();
            AXON_ASSERT(ptr != nullptr);
            size_t i = 0;
#if defined(AXON_HAS_AVX512F)
            const __m512 vzero = _mm512_setzero_ps();
            for (; i + 16 <= count; i += 16) {
                __m512 v = _mm512_loadu_ps(ptr + i);
                _mm512_storeu_ps(ptr + i, _mm512_max_ps(v, vzero));
            }
#elif defined(AXON_HAS_AVX2)
            const __m256 vzero = _mm256_setzero_ps();
            for (; i + 8 <= count; i += 8) {
                __m256 v = _mm256_loadu_ps(ptr + i);
                _mm256_storeu_ps(ptr + i, _mm256_max_ps(v, vzero));
            }
#elif defined(AXON_HAS_NEON)
            const float32x4_t vzero = vdupq_n_f32(0.0f);
            for (; i + 4 <= count; i += 4) {
                float32x4_t v = vld1q_f32(ptr + i);
                vst1q_f32(ptr + i, vmaxq_f32(v, vzero));
            }
#endif
            for (; i < count; ++i) {
                if (ptr[i] < 0.0f) {
                    ptr[i] = 0.0f;
                }
            }
            break;
        }
        case axon::core::DType::Float64: {
            double* ptr = self.data<double>();
            AXON_ASSERT(ptr != nullptr);
            size_t i = 0;
#if defined(AXON_HAS_AVX512F)
            const __m512d vzero = _mm512_setzero_pd();
            for (; i + 8 <= count; i += 8) {
                __m512d v = _mm512_loadu_pd(ptr + i);
                _mm512_storeu_pd(ptr + i, _mm512_max_pd(v, vzero));
            }
#elif defined(AXON_HAS_AVX2)
            const __m256d vzero = _mm256_setzero_pd();
            for (; i + 4 <= count; i += 4) {
                __m256d v = _mm256_loadu_pd(ptr + i);
                _mm256_storeu_pd(ptr + i, _mm256_max_pd(v, vzero));
            }
#elif defined(AXON_HAS_NEON)
            const float64x2_t vzero = vdupq_n_f64(0.0);
            for (; i + 2 <= count; i += 2) {
                float64x2_t v = vld1q_f64(ptr + i);
                vst1q_f64(ptr + i, vmaxq_f64(v, vzero));
            }
#endif
            for (; i < count; ++i) {
                if (ptr[i] < 0.0) {
                    ptr[i] = 0.0;
                }
            }
            break;
        }
        case axon::core::DType::Int32: {
            int32_t* ptr = self.data<int32_t>();
            AXON_ASSERT(ptr != nullptr);
            size_t i = 0;
#if defined(AXON_HAS_AVX512F)
            const __m512i vzero = _mm512_setzero_si512();
            for (; i + 16 <= count; i += 16) {
                __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(ptr + i));
                _mm512_storeu_si512(reinterpret_cast<void*>(ptr + i), _mm512_max_epi32(v, vzero));
            }
#elif defined(AXON_HAS_AVX2)
            const __m256i vzero = _mm256_setzero_si256();
            for (; i + 8 <= count; i += 8) {
                __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + i));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr + i), _mm256_max_epi32(v, vzero));
            }
#elif defined(AXON_HAS_NEON)
            const int32x4_t vzero = vdupq_n_s32(0);
            for (; i + 4 <= count; i += 4) {
                int32x4_t v = vld1q_s32(ptr + i);
                vst1q_s32(ptr + i, vmaxq_s32(v, vzero));
            }
#endif
            for (; i < count; ++i) {
                if (ptr[i] < 0) {
                    ptr[i] = 0;
                }
            }
            break;
        }
        default:
            AXON_DEBUG_TRAP();
            break;
    }
}

void scale_(
    axon::tensor::Tensor& self,
    float scalar
) noexcept {
    detail::validate_inplace_unary_op(self);
    const size_t count = self.num_elements();
    if (AXON_UNLIKELY(count == 0)) {
        return;
    }

    switch (self.dtype()) {
        case axon::core::DType::Float32: {
            float* ptr = self.data<float>();
            AXON_ASSERT(ptr != nullptr);
            size_t i = 0;
#if defined(AXON_HAS_AVX512F)
            const __m512 vscalar = _mm512_set1_ps(scalar);
            for (; i + 16 <= count; i += 16) {
                __m512 v = _mm512_loadu_ps(ptr + i);
                _mm512_storeu_ps(ptr + i, _mm512_mul_ps(v, vscalar));
            }
#elif defined(AXON_HAS_AVX2)
            const __m256 vscalar = _mm256_set1_ps(scalar);
            for (; i + 8 <= count; i += 8) {
                __m256 v = _mm256_loadu_ps(ptr + i);
                _mm256_storeu_ps(ptr + i, _mm256_mul_ps(v, vscalar));
            }
#elif defined(AXON_HAS_NEON)
            const float32x4_t vscalar = vdupq_n_f32(scalar);
            for (; i + 4 <= count; i += 4) {
                float32x4_t v = vld1q_f32(ptr + i);
                vst1q_f32(ptr + i, vmulq_f32(v, vscalar));
            }
#endif
            for (; i < count; ++i) {
                ptr[i] *= scalar;
            }
            break;
        }
        case axon::core::DType::Float64: {
            double* ptr = self.data<double>();
            AXON_ASSERT(ptr != nullptr);
            const double dscalar = static_cast<double>(scalar);
            size_t i = 0;
#if defined(AXON_HAS_AVX512F)
            const __m512d vscalar = _mm512_set1_pd(dscalar);
            for (; i + 8 <= count; i += 8) {
                __m512d v = _mm512_loadu_pd(ptr + i);
                _mm512_storeu_pd(ptr + i, _mm512_mul_pd(v, vscalar));
            }
#elif defined(AXON_HAS_AVX2)
            const __m256d vscalar = _mm256_set1_pd(dscalar);
            for (; i + 4 <= count; i += 4) {
                __m256d v = _mm256_loadu_pd(ptr + i);
                _mm256_storeu_pd(ptr + i, _mm256_mul_pd(v, vscalar));
            }
#elif defined(AXON_HAS_NEON)
            const float64x2_t vscalar = vdupq_n_f64(dscalar);
            for (; i + 2 <= count; i += 2) {
                float64x2_t v = vld1q_f64(ptr + i);
                vst1q_f64(ptr + i, vmulq_f64(v, vscalar));
            }
#endif
            for (; i < count; ++i) {
                ptr[i] *= dscalar;
            }
            break;
        }
        case axon::core::DType::Int32: {
            int32_t* ptr = self.data<int32_t>();
            AXON_ASSERT(ptr != nullptr);
            const int32_t iscalar = static_cast<int32_t>(scalar);
            size_t i = 0;
#if defined(AXON_HAS_AVX512F)
            const __m512i vscalar = _mm512_set1_epi32(iscalar);
            for (; i + 16 <= count; i += 16) {
                __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(ptr + i));
                _mm512_storeu_si512(reinterpret_cast<void*>(ptr + i), _mm512_mullo_epi32(v, vscalar));
            }
#elif defined(AXON_HAS_AVX2)
            const __m256i vscalar = _mm256_set1_epi32(iscalar);
            for (; i + 8 <= count; i += 8) {
                __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + i));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr + i), _mm256_mullo_epi32(v, vscalar));
            }
#elif defined(AXON_HAS_NEON)
            const int32x4_t vscalar = vdupq_n_s32(iscalar);
            for (; i + 4 <= count; i += 4) {
                int32x4_t v = vld1q_s32(ptr + i);
                vst1q_s32(ptr + i, vmulq_s32(v, vscalar));
            }
#endif
            for (; i < count; ++i) {
                ptr[i] *= iscalar;
            }
            break;
        }
        default:
            AXON_DEBUG_TRAP();
            break;
    }
}

void fill_(
    axon::tensor::Tensor& self,
    float value
) noexcept {
    detail::validate_inplace_unary_op(self);
    const size_t count = self.num_elements();

    switch (self.dtype()) {
        case axon::core::DType::Float32:
            kernels::fill_kernel_f32(self.data<float>(), value, count);
            break;
        case axon::core::DType::Float64:
            kernels::fill_kernel_f64(self.data<double>(), static_cast<double>(value), count);
            break;
        case axon::core::DType::Int32:
            kernels::fill_kernel_i32(self.data<int32_t>(), static_cast<int32_t>(value), count);
            break;
        default:
            AXON_DEBUG_TRAP();
            break;
    }
}

} // namespace ops
} // namespace axon
