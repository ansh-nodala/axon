#include "axon/ops/reductions.hpp"
#include <math.h>

#if defined(AXON_ARCH_X86_64)
#include <immintrin.h>
#elif defined(AXON_ARCH_ARM64) && defined(AXON_HAS_NEON)
#include <arm_neon.h>
#endif

namespace {

#if defined(AXON_ARCH_X86_64)

// Register-level horizontal sum for AVX2 (__m256) without stack allocation
AXON_ALWAYS_INLINE float horizontal_sum_ps_avx2(__m256 v) noexcept {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum4 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum4);
    __m128 sums = _mm_add_ps(sum4, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// Register-level horizontal max for AVX2 (__m256)
AXON_ALWAYS_INLINE float horizontal_max_ps_avx2(__m256 v) noexcept {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 max4 = _mm_max_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(max4);
    __m128 maxs = _mm_max_ps(max4, shuf);
    shuf = _mm_movehl_ps(shuf, maxs);
    maxs = _mm_max_ss(maxs, shuf);
    return _mm_cvtss_f32(maxs);
}

// Register-level horizontal min for AVX2 (__m256)
AXON_ALWAYS_INLINE float horizontal_min_ps_avx2(__m256 v) noexcept {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 min4 = _mm_min_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(min4);
    __m128 mins = _mm_min_ps(min4, shuf);
    shuf = _mm_movehl_ps(shuf, mins);
    mins = _mm_min_ss(mins, shuf);
    return _mm_cvtss_f32(mins);
}

// Register-level horizontal sum for AVX2 double (__m256d)
AXON_ALWAYS_INLINE double horizontal_sum_pd_avx2(__m256d v) noexcept {
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d sum2 = _mm_add_pd(lo, hi);
    __m128d shuf = _mm_unpackhi_pd(sum2, sum2);
    __m128d sums = _mm_add_sd(sum2, shuf);
    return _mm_cvtsd_f64(sums);
}

#endif // AXON_ARCH_X86_64

} // namespace

namespace axon {
namespace ops {
namespace kernels {

// ============================================================================
// 1. SCALAR & REDUCTION KERNEL IMPLEMENTATIONS
// ============================================================================

AXON_NODISCARD AXON_API float sum_kernel_f32(const float* AXON_RESTRICT in, size_t count) noexcept {
    AXON_ASSERT(in != nullptr);
    size_t i = 0;
    float total = 0.0f;

#if defined(AXON_HAS_AVX512F)
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    for (; i + 63 < count; i += 64) {
        acc0 = _mm512_add_ps(acc0, _mm512_loadu_ps(in + i));
        acc1 = _mm512_add_ps(acc1, _mm512_loadu_ps(in + i + 16));
        acc2 = _mm512_add_ps(acc2, _mm512_loadu_ps(in + i + 32));
        acc3 = _mm512_add_ps(acc3, _mm512_loadu_ps(in + i + 48));
    }
    acc0 = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    for (; i + 15 < count; i += 16) {
        acc0 = _mm512_add_ps(acc0, _mm512_loadu_ps(in + i));
    }
    total += _mm512_reduce_add_ps(acc0);
#elif defined(AXON_HAS_AVX2)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    for (; i + 31 < count; i += 32) {
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(in + i));
        acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(in + i + 8));
        acc2 = _mm256_add_ps(acc2, _mm256_loadu_ps(in + i + 16));
        acc3 = _mm256_add_ps(acc3, _mm256_loadu_ps(in + i + 24));
    }
    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    for (; i + 7 < count; i += 8) {
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(in + i));
    }
    total += horizontal_sum_ps_avx2(acc0);
#elif defined(AXON_HAS_NEON)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    for (; i + 15 < count; i += 16) {
        acc0 = vaddq_f32(acc0, vld1q_f32(in + i));
        acc1 = vaddq_f32(acc1, vld1q_f32(in + i + 4));
        acc2 = vaddq_f32(acc2, vld1q_f32(in + i + 8));
        acc3 = vaddq_f32(acc3, vld1q_f32(in + i + 12));
    }
    acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    for (; i + 3 < count; i += 4) {
        acc0 = vaddq_f32(acc0, vld1q_f32(in + i));
    }
    float32_t t[4];
    vst1q_f32(t, acc0);
    total += t[0] + t[1] + t[2] + t[3];
#endif

    for (; i < count; ++i) {
        total += in[i];
    }
    return total;
}

AXON_NODISCARD AXON_API double sum_kernel_f64(const double* AXON_RESTRICT in, size_t count) noexcept {
    AXON_ASSERT(in != nullptr);
    size_t i = 0;
    double total = 0.0;

#if defined(AXON_HAS_AVX512F)
    __m512d acc0 = _mm512_setzero_pd();
    __m512d acc1 = _mm512_setzero_pd();
    for (; i + 15 < count; i += 16) {
        acc0 = _mm512_add_pd(acc0, _mm512_loadu_pd(in + i));
        acc1 = _mm512_add_pd(acc1, _mm512_loadu_pd(in + i + 8));
    }
    acc0 = _mm512_add_pd(acc0, acc1);
    for (; i + 7 < count; i += 8) {
        acc0 = _mm512_add_pd(acc0, _mm512_loadu_pd(in + i));
    }
    total += _mm512_reduce_add_pd(acc0);
#elif defined(AXON_HAS_AVX2)
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    for (; i + 7 < count; i += 8) {
        acc0 = _mm256_add_pd(acc0, _mm256_loadu_pd(in + i));
        acc1 = _mm256_add_pd(acc1, _mm256_loadu_pd(in + i + 4));
    }
    acc0 = _mm256_add_pd(acc0, acc1);
    for (; i + 3 < count; i += 4) {
        acc0 = _mm256_add_pd(acc0, _mm256_loadu_pd(in + i));
    }
    total += horizontal_sum_pd_avx2(acc0);
#endif

    for (; i < count; ++i) {
        total += in[i];
    }
    return total;
}

AXON_NODISCARD AXON_API int32_t sum_kernel_i32(const int32_t* AXON_RESTRICT in, size_t count) noexcept {
    AXON_ASSERT(in != nullptr);
    size_t i = 0;
    int32_t total = 0;

#if defined(AXON_HAS_AVX2)
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    for (; i + 15 < count; i += 16) {
        acc0 = _mm256_add_epi32(acc0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i)));
        acc1 = _mm256_add_epi32(acc1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i + 8)));
    }
    acc0 = _mm256_add_epi32(acc0, acc1);
    for (; i + 7 < count; i += 8) {
        acc0 = _mm256_add_epi32(acc0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i)));
    }
    alignas(32) int32_t buf[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(buf), acc0);
    total += buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
#endif

    for (; i < count; ++i) {
        total += in[i];
    }
    return total;
}

AXON_NODISCARD AXON_API float max_kernel_f32(const float* AXON_RESTRICT in, size_t count) noexcept {
    AXON_ASSERT(in != nullptr && count > 0);
    float max_val = in[0];
    size_t i = 0;

#if defined(AXON_HAS_AVX2)
    if (count >= 8) {
        __m256 vmax0 = _mm256_set1_ps(in[0]);
        __m256 vmax1 = _mm256_set1_ps(in[0]);
        __m256 vmax2 = _mm256_set1_ps(in[0]);
        __m256 vmax3 = _mm256_set1_ps(in[0]);
        for (; i + 31 < count; i += 32) {
            vmax0 = _mm256_max_ps(vmax0, _mm256_loadu_ps(in + i));
            vmax1 = _mm256_max_ps(vmax1, _mm256_loadu_ps(in + i + 8));
            vmax2 = _mm256_max_ps(vmax2, _mm256_loadu_ps(in + i + 16));
            vmax3 = _mm256_max_ps(vmax3, _mm256_loadu_ps(in + i + 24));
        }
        vmax0 = _mm256_max_ps(_mm256_max_ps(vmax0, vmax1), _mm256_max_ps(vmax2, vmax3));
        for (; i + 7 < count; i += 8) {
            vmax0 = _mm256_max_ps(vmax0, _mm256_loadu_ps(in + i));
        }
        float reg_max = horizontal_max_ps_avx2(vmax0);
        if (reg_max > max_val) max_val = reg_max;
    }
#elif defined(AXON_HAS_NEON)
    if (count >= 4) {
        float32x4_t vmax0 = vdupq_n_f32(in[0]);
        float32x4_t vmax1 = vdupq_n_f32(in[0]);
        float32x4_t vmax2 = vdupq_n_f32(in[0]);
        float32x4_t vmax3 = vdupq_n_f32(in[0]);
        for (; i + 15 < count; i += 16) {
            vmax0 = vmaxq_f32(vmax0, vld1q_f32(in + i));
            vmax1 = vmaxq_f32(vmax1, vld1q_f32(in + i + 4));
            vmax2 = vmaxq_f32(vmax2, vld1q_f32(in + i + 8));
            vmax3 = vmaxq_f32(vmax3, vld1q_f32(in + i + 12));
        }
        vmax0 = vmaxq_f32(vmaxq_f32(vmax0, vmax1), vmaxq_f32(vmax2, vmax3));
        for (; i + 3 < count; i += 4) {
            vmax0 = vmaxq_f32(vmax0, vld1q_f32(in + i));
        }
        float32_t t[4];
        vst1q_f32(t, vmax0);
        for (int k = 0; k < 4; ++k) {
            if (t[k] > max_val) max_val = t[k];
        }
    }
#endif

    for (; i < count; ++i) {
        if (in[i] > max_val) max_val = in[i];
    }
    return max_val;
}

AXON_NODISCARD AXON_API double max_kernel_f64(const double* AXON_RESTRICT in, size_t count) noexcept {
    AXON_ASSERT(in != nullptr && count > 0);
    double max_val = in[0];
    for (size_t i = 1; i < count; ++i) {
        if (in[i] > max_val) max_val = in[i];
    }
    return max_val;
}

AXON_NODISCARD AXON_API int32_t max_kernel_i32(const int32_t* AXON_RESTRICT in, size_t count) noexcept {
    AXON_ASSERT(in != nullptr && count > 0);
    int32_t max_val = in[0];
    for (size_t i = 1; i < count; ++i) {
        if (in[i] > max_val) max_val = in[i];
    }
    return max_val;
}

AXON_NODISCARD AXON_API float min_kernel_f32(const float* AXON_RESTRICT in, size_t count) noexcept {
    AXON_ASSERT(in != nullptr && count > 0);
    float min_val = in[0];
    size_t i = 0;

#if defined(AXON_HAS_AVX2)
    if (count >= 8) {
        __m256 vmin0 = _mm256_set1_ps(in[0]);
        __m256 vmin1 = _mm256_set1_ps(in[0]);
        __m256 vmin2 = _mm256_set1_ps(in[0]);
        __m256 vmin3 = _mm256_set1_ps(in[0]);
        for (; i + 31 < count; i += 32) {
            vmin0 = _mm256_min_ps(vmin0, _mm256_loadu_ps(in + i));
            vmin1 = _mm256_min_ps(vmin1, _mm256_loadu_ps(in + i + 8));
            vmin2 = _mm256_min_ps(vmin2, _mm256_loadu_ps(in + i + 16));
            vmin3 = _mm256_min_ps(vmin3, _mm256_loadu_ps(in + i + 24));
        }
        vmin0 = _mm256_min_ps(_mm256_min_ps(vmin0, vmin1), _mm256_min_ps(vmin2, vmin3));
        for (; i + 7 < count; i += 8) {
            vmin0 = _mm256_min_ps(vmin0, _mm256_loadu_ps(in + i));
        }
        float reg_min = horizontal_min_ps_avx2(vmin0);
        if (reg_min < min_val) min_val = reg_min;
    }
#endif

    for (; i < count; ++i) {
        if (in[i] < min_val) min_val = in[i];
    }
    return min_val;
}

AXON_NODISCARD AXON_API double min_kernel_f64(const double* AXON_RESTRICT in, size_t count) noexcept {
    AXON_ASSERT(in != nullptr && count > 0);
    double min_val = in[0];
    for (size_t i = 1; i < count; ++i) {
        if (in[i] < min_val) min_val = in[i];
    }
    return min_val;
}

AXON_NODISCARD AXON_API int32_t min_kernel_i32(const int32_t* AXON_RESTRICT in, size_t count) noexcept {
    AXON_ASSERT(in != nullptr && count > 0);
    int32_t min_val = in[0];
    for (size_t i = 1; i < count; ++i) {
        if (in[i] < min_val) min_val = in[i];
    }
    return min_val;
}

AXON_NODISCARD AXON_API size_t argmax_kernel_f32(const float* AXON_RESTRICT in, size_t count) noexcept {
    AXON_ASSERT(in != nullptr && count > 0);
    size_t max_idx = 0;
    float max_val = in[0];
    for (size_t i = 1; i < count; ++i) {
        if (in[i] > max_val) {
            max_val = in[i];
            max_idx = i;
        }
    }
    return max_idx;
}

AXON_NODISCARD AXON_API size_t argmax_kernel_f64(const double* AXON_RESTRICT in, size_t count) noexcept {
    AXON_ASSERT(in != nullptr && count > 0);
    size_t max_idx = 0;
    double max_val = in[0];
    for (size_t i = 1; i < count; ++i) {
        if (in[i] > max_val) {
            max_val = in[i];
            max_idx = i;
        }
    }
    return max_idx;
}

AXON_NODISCARD AXON_API size_t argmax_kernel_i32(const int32_t* AXON_RESTRICT in, size_t count) noexcept {
    AXON_ASSERT(in != nullptr && count > 0);
    size_t max_idx = 0;
    int32_t max_val = in[0];
    for (size_t i = 1; i < count; ++i) {
        if (in[i] > max_val) {
            max_val = in[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// ============================================================================
// 2. DEEP LEARNING ACTIVATION KERNELS (FLOAT32)
// ============================================================================

AXON_API void exp_kernel_f32(const float* AXON_RESTRICT in, float* AXON_RESTRICT out, size_t count) noexcept {
    AXON_ASSERT(in != nullptr && out != nullptr);
    for (size_t i = 0; i < count; ++i) {
        out[i] = ::expf(in[i]);
    }
}

AXON_API void sigmoid_kernel_f32(const float* AXON_RESTRICT in, float* AXON_RESTRICT out, size_t count) noexcept {
    AXON_ASSERT(in != nullptr && out != nullptr);
    for (size_t i = 0; i < count; ++i) {
        out[i] = 1.0f / (1.0f + ::expf(-in[i]));
    }
}

AXON_API void tanh_kernel_f32(const float* AXON_RESTRICT in, float* AXON_RESTRICT out, size_t count) noexcept {
    AXON_ASSERT(in != nullptr && out != nullptr);
    for (size_t i = 0; i < count; ++i) {
        out[i] = ::tanhf(in[i]);
    }
}

AXON_API void silu_kernel_f32(const float* AXON_RESTRICT in, float* AXON_RESTRICT out, size_t count) noexcept {
    AXON_ASSERT(in != nullptr && out != nullptr);
    for (size_t i = 0; i < count; ++i) {
        float x = in[i];
        out[i] = x / (1.0f + ::expf(-x));
    }
}

AXON_API void gelu_kernel_f32(const float* AXON_RESTRICT in, float* AXON_RESTRICT out, size_t count) noexcept {
    AXON_ASSERT(in != nullptr && out != nullptr);
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    constexpr float kCoeff = 0.044715f;
    for (size_t i = 0; i < count; ++i) {
        float x = in[i];
        float x3 = x * x * x;
        out[i] = 0.5f * x * (1.0f + ::tanhf(kSqrt2OverPi * (x + kCoeff * x3)));
    }
}

// ============================================================================
// 3. FUSED ROW-WISE NORMALIZATION & PROBABILITY KERNELS
// ============================================================================

AXON_API void softmax_row_kernel_f32(const float* AXON_RESTRICT in, float* AXON_RESTRICT out, size_t outer_rows, size_t row_size) noexcept {
    AXON_ASSERT(in != nullptr && out != nullptr);
    for (size_t r = 0; r < outer_rows; ++r) {
        const float* row_in = in + r * row_size;
        float* row_out = out + r * row_size;

        if (row_size == 0) continue;

        // Pass 1: Find max value using SIMD-accelerated max_kernel_f32
        float max_val = max_kernel_f32(row_in, row_size);

        // Pass 2: Compute row_out[c] = ::expf(row_in[c] - max_val) (4x unrolled scalar loop) and accumulate sum
        float sum = 0.0f;
        size_t c = 0;
        size_t unrolled_end = (row_size / 4) * 4;
        for (; c < unrolled_end; c += 4) {
            float e0 = ::expf(row_in[c + 0] - max_val);
            float e1 = ::expf(row_in[c + 1] - max_val);
            float e2 = ::expf(row_in[c + 2] - max_val);
            float e3 = ::expf(row_in[c + 3] - max_val);

            row_out[c + 0] = e0;
            row_out[c + 1] = e1;
            row_out[c + 2] = e2;
            row_out[c + 3] = e3;

            sum += e0 + e1 + e2 + e3;
        }
        for (; c < row_size; ++c) {
            float e = ::expf(row_in[c] - max_val);
            row_out[c] = e;
            sum += e;
        }

        // Pass 3: Compute inv_sum and vectorize final scaling loop
        float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        c = 0;
#if defined(AXON_HAS_AVX512F)
        __m512 vinv512 = _mm512_set1_ps(inv_sum);
        for (; c + 15 < row_size; c += 16) {
            __m512 v = _mm512_loadu_ps(row_out + c);
            v = _mm512_mul_ps(v, vinv512);
            _mm512_storeu_ps(row_out + c, v);
        }
#elif defined(AXON_HAS_AVX2)
        __m256 vinv = _mm256_set1_ps(inv_sum);
        for (; c + 7 < row_size; c += 8) {
            __m256 v = _mm256_loadu_ps(row_out + c);
            v = _mm256_mul_ps(v, vinv);
            _mm256_storeu_ps(row_out + c, v);
        }
#elif defined(AXON_HAS_NEON)
        float32x4_t vinv_neon = vdupq_n_f32(inv_sum);
        for (; c + 3 < row_size; c += 4) {
            float32x4_t v = vld1q_f32(row_out + c);
            v = vmulq_f32(v, vinv_neon);
            vst1q_f32(row_out + c, v);
        }
#endif
        for (; c < row_size; ++c) {
            row_out[c] *= inv_sum;
        }
    }
}

AXON_API void rmsnorm_row_kernel_f32(const float* AXON_RESTRICT in, const float* AXON_RESTRICT weight, float* AXON_RESTRICT out, size_t outer_rows, size_t row_size, float eps) noexcept {
    AXON_ASSERT(in != nullptr && weight != nullptr && out != nullptr);
    for (size_t r = 0; r < outer_rows; ++r) {
        const float* row_in = in + r * row_size;
        float* row_out = out + r * row_size;

        if (row_size == 0) continue;

        // Pass 1: Compute sum of squares using 4x unrolled SIMD FMA
        float sum_sq = 0.0f;
        size_t c = 0;
#if defined(AXON_HAS_AVX512F)
        __m512 acc0_512 = _mm512_setzero_ps();
        __m512 acc1_512 = _mm512_setzero_ps();
        for (; c + 31 < row_size; c += 32) {
            __m512 v0 = _mm512_loadu_ps(row_in + c);
            __m512 v1 = _mm512_loadu_ps(row_in + c + 16);
            acc0_512 = _mm512_fmadd_ps(v0, v0, acc0_512);
            acc1_512 = _mm512_fmadd_ps(v1, v1, acc1_512);
        }
        acc0_512 = _mm512_add_ps(acc0_512, acc1_512);
        for (; c + 15 < row_size; c += 16) {
            __m512 v0 = _mm512_loadu_ps(row_in + c);
            acc0_512 = _mm512_fmadd_ps(v0, v0, acc0_512);
        }
        sum_sq += _mm512_reduce_add_ps(acc0_512);
#elif defined(AXON_HAS_AVX2) && defined(AXON_HAS_FMA)
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        for (; c + 31 < row_size; c += 32) {
            __m256 v0 = _mm256_loadu_ps(row_in + c);
            __m256 v1 = _mm256_loadu_ps(row_in + c + 8);
            __m256 v2 = _mm256_loadu_ps(row_in + c + 16);
            __m256 v3 = _mm256_loadu_ps(row_in + c + 24);
            acc0 = _mm256_fmadd_ps(v0, v0, acc0);
            acc1 = _mm256_fmadd_ps(v1, v1, acc1);
            acc2 = _mm256_fmadd_ps(v2, v2, acc2);
            acc3 = _mm256_fmadd_ps(v3, v3, acc3);
        }
        acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        for (; c + 7 < row_size; c += 8) {
            __m256 v0 = _mm256_loadu_ps(row_in + c);
            acc0 = _mm256_fmadd_ps(v0, v0, acc0);
        }
        sum_sq += horizontal_sum_ps_avx2(acc0);
#elif defined(AXON_HAS_NEON)
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);
        for (; c + 15 < row_size; c += 16) {
            float32x4_t v0 = vld1q_f32(row_in + c);
            float32x4_t v1 = vld1q_f32(row_in + c + 4);
            float32x4_t v2 = vld1q_f32(row_in + c + 8);
            float32x4_t v3 = vld1q_f32(row_in + c + 12);
            acc0 = vmlaq_f32(acc0, v0, v0);
            acc1 = vmlaq_f32(acc1, v1, v1);
            acc2 = vmlaq_f32(acc2, v2, v2);
            acc3 = vmlaq_f32(acc3, v3, v3);
        }
        acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        for (; c + 3 < row_size; c += 4) {
            float32x4_t v0 = vld1q_f32(row_in + c);
            acc0 = vmlaq_f32(acc0, v0, v0);
        }
        float32_t t[4];
        vst1q_f32(t, acc0);
        sum_sq += t[0] + t[1] + t[2] + t[3];
#endif
        for (; c < row_size; ++c) {
            float val = row_in[c];
            sum_sq += val * val;
        }

        // Pass 2: Compute rscale and vectorize normalization & weighting
        float mean_sq = sum_sq / static_cast<float>(row_size);
        float rscale = 1.0f / ::sqrtf(mean_sq + eps);

        c = 0;
#if defined(AXON_HAS_AVX512F)
        __m512 vrscale512 = _mm512_set1_ps(rscale);
        for (; c + 15 < row_size; c += 16) {
            __m512 vin = _mm512_loadu_ps(row_in + c);
            __m512 vw = _mm512_loadu_ps(weight + c);
            __m512 vout = _mm512_mul_ps(_mm512_mul_ps(vin, vrscale512), vw);
            _mm512_storeu_ps(row_out + c, vout);
        }
#elif defined(AXON_HAS_AVX2)
        __m256 vrscale = _mm256_set1_ps(rscale);
        for (; c + 7 < row_size; c += 8) {
            __m256 vin = _mm256_loadu_ps(row_in + c);
            __m256 vw = _mm256_loadu_ps(weight + c);
            __m256 vout = _mm256_mul_ps(_mm256_mul_ps(vin, vrscale), vw);
            _mm256_storeu_ps(row_out + c, vout);
        }
#elif defined(AXON_HAS_NEON)
        float32x4_t vrscale_neon = vdupq_n_f32(rscale);
        for (; c + 3 < row_size; c += 4) {
            float32x4_t vin = vld1q_f32(row_in + c);
            float32x4_t vw = vld1q_f32(weight + c);
            float32x4_t vout = vmulq_f32(vmulq_f32(vin, vrscale_neon), vw);
            vst1q_f32(row_out + c, vout);
        }
#endif
        for (; c < row_size; ++c) {
            row_out[c] = row_in[c] * rscale * weight[c];
        }
    }
}

} // namespace kernels

// ============================================================================
// 4. HIGH-LEVEL TENSOR API IMPLEMENTATIONS
// ============================================================================

AXON_NODISCARD AXON_API float sum_all(const axon::tensor::Tensor& in) noexcept {
    AXON_ASSERT(in.is_contiguous());
    axon::core::DType dtype = in.dtype();
    if (dtype == axon::core::DType::Float32) {
        return kernels::sum_kernel_f32(in.data<float>(), in.num_elements());
    } else if (dtype == axon::core::DType::Float64) {
        return static_cast<float>(kernels::sum_kernel_f64(in.data<double>(), in.num_elements()));
    } else if (dtype == axon::core::DType::Int32) {
        return static_cast<float>(kernels::sum_kernel_i32(in.data<int32_t>(), in.num_elements()));
    }
    AXON_DEBUG_TRAP();
    return 0.0f;
}

AXON_NODISCARD AXON_API float mean_all(const axon::tensor::Tensor& in) noexcept {
    size_t count = in.num_elements();
    if (count == 0) return 0.0f;
    return sum_all(in) / static_cast<float>(count);
}

AXON_NODISCARD AXON_API float max_all(const axon::tensor::Tensor& in) noexcept {
    AXON_ASSERT(in.is_contiguous() && in.num_elements() > 0);
    axon::core::DType dtype = in.dtype();
    if (dtype == axon::core::DType::Float32) {
        return kernels::max_kernel_f32(in.data<float>(), in.num_elements());
    } else if (dtype == axon::core::DType::Float64) {
        return static_cast<float>(kernels::max_kernel_f64(in.data<double>(), in.num_elements()));
    } else if (dtype == axon::core::DType::Int32) {
        return static_cast<float>(kernels::max_kernel_i32(in.data<int32_t>(), in.num_elements()));
    }
    AXON_DEBUG_TRAP();
    return 0.0f;
}

AXON_NODISCARD AXON_API float min_all(const axon::tensor::Tensor& in) noexcept {
    AXON_ASSERT(in.is_contiguous() && in.num_elements() > 0);
    axon::core::DType dtype = in.dtype();
    if (dtype == axon::core::DType::Float32) {
        return kernels::min_kernel_f32(in.data<float>(), in.num_elements());
    } else if (dtype == axon::core::DType::Float64) {
        return static_cast<float>(kernels::min_kernel_f64(in.data<double>(), in.num_elements()));
    } else if (dtype == axon::core::DType::Int32) {
        return static_cast<float>(kernels::min_kernel_i32(in.data<int32_t>(), in.num_elements()));
    }
    AXON_DEBUG_TRAP();
    return 0.0f;
}

AXON_NODISCARD AXON_API int64_t argmax_all(const axon::tensor::Tensor& in) noexcept {
    AXON_ASSERT(in.is_contiguous() && in.num_elements() > 0);
    axon::core::DType dtype = in.dtype();
    if (dtype == axon::core::DType::Float32) {
        return static_cast<int64_t>(kernels::argmax_kernel_f32(in.data<float>(), in.num_elements()));
    } else if (dtype == axon::core::DType::Float64) {
        return static_cast<int64_t>(kernels::argmax_kernel_f64(in.data<double>(), in.num_elements()));
    } else if (dtype == axon::core::DType::Int32) {
        return static_cast<int64_t>(kernels::argmax_kernel_i32(in.data<int32_t>(), in.num_elements()));
    }
    AXON_DEBUG_TRAP();
    return 0;
}

AXON_API void sum_last_dim_out(const axon::tensor::Tensor& in, axon::tensor::Tensor& out) noexcept {
    AXON_ASSERT(in.is_contiguous() && out.is_contiguous());
    size_t ndim = in.ndim();
    AXON_ASSERT(ndim > 0);
    size_t row_size = static_cast<size_t>(in.dim(ndim - 1));
    size_t outer_rows = in.num_elements() / row_size;

    axon::core::DType dtype = in.dtype();
    if (dtype == axon::core::DType::Float32) {
        const float* p_in = in.data<float>();
        float* p_out = out.data<float>();
        for (size_t r = 0; r < outer_rows; ++r) {
            p_out[r] = kernels::sum_kernel_f32(p_in + r * row_size, row_size);
        }
    } else if (dtype == axon::core::DType::Float64) {
        const double* p_in = in.data<double>();
        double* p_out = out.data<double>();
        for (size_t r = 0; r < outer_rows; ++r) {
            p_out[r] = kernels::sum_kernel_f64(p_in + r * row_size, row_size);
        }
    } else if (dtype == axon::core::DType::Int32) {
        const int32_t* p_in = in.data<int32_t>();
        int32_t* p_out = out.data<int32_t>();
        for (size_t r = 0; r < outer_rows; ++r) {
            p_out[r] = kernels::sum_kernel_i32(p_in + r * row_size, row_size);
        }
    }
}

AXON_NODISCARD AXON_API axon::tensor::Tensor sum_last_dim(const axon::tensor::Tensor& in) noexcept {
    size_t ndim = in.ndim();
    AXON_ASSERT(ndim > 0);
    int64_t out_dims[8];
    size_t out_ndim = (ndim == 1) ? 1 : (ndim - 1);
    if (ndim == 1) {
        out_dims[0] = 1;
    } else {
        for (size_t i = 0; i < out_ndim; ++i) {
            out_dims[i] = in.dim(i);
        }
    }
    axon::tensor::Tensor out(out_dims, out_ndim, in.dtype(), in.device());
    sum_last_dim_out(in, out);
    return out;
}

AXON_API void mean_last_dim_out(const axon::tensor::Tensor& in, axon::tensor::Tensor& out) noexcept {
    AXON_ASSERT(in.is_contiguous() && out.is_contiguous());
    size_t ndim = in.ndim();
    AXON_ASSERT(ndim > 0);
    size_t row_size = static_cast<size_t>(in.dim(ndim - 1));
    size_t outer_rows = in.num_elements() / row_size;

    axon::core::DType dtype = in.dtype();
    if (dtype == axon::core::DType::Float32) {
        const float* p_in = in.data<float>();
        float* p_out = out.data<float>();
        float inv_row_size = row_size > 0 ? (1.0f / static_cast<float>(row_size)) : 0.0f;
        for (size_t r = 0; r < outer_rows; ++r) {
            float sum = kernels::sum_kernel_f32(p_in + r * row_size, row_size);
            p_out[r] = sum * inv_row_size;
        }
    } else if (dtype == axon::core::DType::Float64) {
        const double* p_in = in.data<double>();
        double* p_out = out.data<double>();
        double inv_row_size = row_size > 0 ? (1.0 / static_cast<double>(row_size)) : 0.0;
        for (size_t r = 0; r < outer_rows; ++r) {
            double sum = kernels::sum_kernel_f64(p_in + r * row_size, row_size);
            p_out[r] = sum * inv_row_size;
        }
    } else if (dtype == axon::core::DType::Int32) {
        const int32_t* p_in = in.data<int32_t>();
        int32_t* p_out = out.data<int32_t>();
        int32_t inv_row_size = row_size > 0 ? static_cast<int32_t>(row_size) : 1;
        for (size_t r = 0; r < outer_rows; ++r) {
            int32_t sum = kernels::sum_kernel_i32(p_in + r * row_size, row_size);
            p_out[r] = sum / inv_row_size;
        }
    }
}

AXON_NODISCARD AXON_API axon::tensor::Tensor mean_last_dim(const axon::tensor::Tensor& in) noexcept {
    size_t ndim = in.ndim();
    AXON_ASSERT(ndim > 0);
    int64_t out_dims[8];
    size_t out_ndim = (ndim == 1) ? 1 : (ndim - 1);
    if (ndim == 1) {
        out_dims[0] = 1;
    } else {
        for (size_t i = 0; i < out_ndim; ++i) {
            out_dims[i] = in.dim(i);
        }
    }
    axon::tensor::Tensor out(out_dims, out_ndim, in.dtype(), in.device());
    mean_last_dim_out(in, out);
    return out;
}

AXON_API void exp_out(const axon::tensor::Tensor& in, axon::tensor::Tensor& out) noexcept {
    AXON_ASSERT(in.dtype() == axon::core::DType::Float32 && out.dtype() == axon::core::DType::Float32);
    AXON_ASSERT(in.is_contiguous() && out.is_contiguous() && in.num_elements() == out.num_elements());
    kernels::exp_kernel_f32(in.data<float>(), out.data<float>(), in.num_elements());
}

AXON_NODISCARD AXON_API axon::tensor::Tensor exp(const axon::tensor::Tensor& in) noexcept {
    axon::tensor::Tensor out(in.shape().dims(), in.ndim(), in.dtype(), in.device());
    exp_out(in, out);
    return out;
}

AXON_API void sigmoid_out(const axon::tensor::Tensor& in, axon::tensor::Tensor& out) noexcept {
    AXON_ASSERT(in.dtype() == axon::core::DType::Float32 && out.dtype() == axon::core::DType::Float32);
    AXON_ASSERT(in.is_contiguous() && out.is_contiguous() && in.num_elements() == out.num_elements());
    kernels::sigmoid_kernel_f32(in.data<float>(), out.data<float>(), in.num_elements());
}

AXON_NODISCARD AXON_API axon::tensor::Tensor sigmoid(const axon::tensor::Tensor& in) noexcept {
    axon::tensor::Tensor out(in.shape().dims(), in.ndim(), in.dtype(), in.device());
    sigmoid_out(in, out);
    return out;
}

AXON_API void tanh_out(const axon::tensor::Tensor& in, axon::tensor::Tensor& out) noexcept {
    AXON_ASSERT(in.dtype() == axon::core::DType::Float32 && out.dtype() == axon::core::DType::Float32);
    AXON_ASSERT(in.is_contiguous() && out.is_contiguous() && in.num_elements() == out.num_elements());
    kernels::tanh_kernel_f32(in.data<float>(), out.data<float>(), in.num_elements());
}

AXON_NODISCARD AXON_API axon::tensor::Tensor tanh_op(const axon::tensor::Tensor& in) noexcept {
    axon::tensor::Tensor out(in.shape().dims(), in.ndim(), in.dtype(), in.device());
    tanh_out(in, out);
    return out;
}

AXON_NODISCARD AXON_API axon::tensor::Tensor tanh(const axon::tensor::Tensor& in) noexcept {
    return tanh_op(in);
}

AXON_API void silu_out(const axon::tensor::Tensor& in, axon::tensor::Tensor& out) noexcept {
    AXON_ASSERT(in.dtype() == axon::core::DType::Float32 && out.dtype() == axon::core::DType::Float32);
    AXON_ASSERT(in.is_contiguous() && out.is_contiguous() && in.num_elements() == out.num_elements());
    kernels::silu_kernel_f32(in.data<float>(), out.data<float>(), in.num_elements());
}

AXON_NODISCARD AXON_API axon::tensor::Tensor silu(const axon::tensor::Tensor& in) noexcept {
    axon::tensor::Tensor out(in.shape().dims(), in.ndim(), in.dtype(), in.device());
    silu_out(in, out);
    return out;
}

AXON_API void gelu_out(const axon::tensor::Tensor& in, axon::tensor::Tensor& out) noexcept {
    AXON_ASSERT(in.dtype() == axon::core::DType::Float32 && out.dtype() == axon::core::DType::Float32);
    AXON_ASSERT(in.is_contiguous() && out.is_contiguous() && in.num_elements() == out.num_elements());
    kernels::gelu_kernel_f32(in.data<float>(), out.data<float>(), in.num_elements());
}

AXON_NODISCARD AXON_API axon::tensor::Tensor gelu(const axon::tensor::Tensor& in) noexcept {
    axon::tensor::Tensor out(in.shape().dims(), in.ndim(), in.dtype(), in.device());
    gelu_out(in, out);
    return out;
}

AXON_API void softmax_out(const axon::tensor::Tensor& in, axon::tensor::Tensor& out) noexcept {
    AXON_ASSERT(in.dtype() == axon::core::DType::Float32 && out.dtype() == axon::core::DType::Float32);
    AXON_ASSERT(in.is_contiguous() && out.is_contiguous() && in.num_elements() == out.num_elements());

    size_t ndim = in.ndim();
    AXON_ASSERT(ndim > 0);
    size_t row_size = static_cast<size_t>(in.dim(ndim - 1));
    size_t outer_rows = in.num_elements() / row_size;

    kernels::softmax_row_kernel_f32(in.data<float>(), out.data<float>(), outer_rows, row_size);
}

AXON_NODISCARD AXON_API axon::tensor::Tensor softmax(const axon::tensor::Tensor& in) noexcept {
    axon::tensor::Tensor out(in.shape().dims(), in.ndim(), in.dtype(), in.device());
    softmax_out(in, out);
    return out;
}

AXON_API void rmsnorm_out(const axon::tensor::Tensor& in, const axon::tensor::Tensor& weight, axon::tensor::Tensor& out, float eps) noexcept {
    AXON_ASSERT(in.dtype() == axon::core::DType::Float32);
    AXON_ASSERT(weight.dtype() == axon::core::DType::Float32);
    AXON_ASSERT(out.dtype() == axon::core::DType::Float32);
    AXON_ASSERT(in.is_contiguous() && weight.is_contiguous() && out.is_contiguous());

    size_t ndim = in.ndim();
    AXON_ASSERT(ndim > 0);
    size_t row_size = static_cast<size_t>(in.dim(ndim - 1));
    AXON_ASSERT(weight.num_elements() == row_size);
    size_t outer_rows = in.num_elements() / row_size;

    kernels::rmsnorm_row_kernel_f32(in.data<float>(), weight.data<float>(), out.data<float>(), outer_rows, row_size, eps);
}

AXON_NODISCARD AXON_API axon::tensor::Tensor rmsnorm(const axon::tensor::Tensor& in, const axon::tensor::Tensor& weight, float eps) noexcept {
    axon::tensor::Tensor out(in.shape().dims(), in.ndim(), in.dtype(), in.device());
    rmsnorm_out(in, weight, out, eps);
    return out;
}

} // namespace ops
} // namespace axon
