#include "axon/ops/matmul.hpp"

#if defined(AXON_ARCH_X86_64)
    #include <immintrin.h>
#elif defined(AXON_ARCH_ARM64) && defined(AXON_HAS_NEON)
    #include <arm_neon.h>
#endif

namespace axon {
namespace ops {

// ============================================================================
// 1. INTERNAL SIMD REDUCTION & MICRO-KERNEL HELPERS
// ============================================================================

namespace {

#if defined(AXON_ARCH_X86_64) && (defined(AXON_HAS_AVX2) || defined(AXON_HAS_AVX512F))

AXON_NODISCARD AXON_ALWAYS_INLINE float hsum256_ps(__m256 v) noexcept {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

AXON_NODISCARD AXON_ALWAYS_INLINE double hsum256_pd(__m256d v) noexcept {
    __m128d vlow  = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1);
    vlow = _mm_add_pd(vlow, vhigh);
    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return _mm_cvtsd_f64(_mm_add_sd(vlow, high64));
}

AXON_NODISCARD AXON_ALWAYS_INLINE int32_t hsum256_epi32(__m256i v) noexcept {
    __m128i vlow   = _mm256_castsi256_si128(v);
    __m128i vhigh  = _mm256_extracti128_si256(v, 1);
    __m128i sum128 = _mm_add_epi32(vlow, vhigh);
    __m128i hi64   = _mm_unpackhi_epi64(sum128, sum128);
    sum128 = _mm_add_epi32(sum128, hi64);
    __m128i hi32   = _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 1, 1, 1));
    sum128 = _mm_add_epi32(sum128, hi32);
    return _mm_cvtsi128_si32(sum128);
}

#if !defined(AXON_HAS_AVX512F)
AXON_NODISCARD AXON_ALWAYS_INLINE __m256 fmadd256_ps(__m256 a, __m256 b, __m256 c) noexcept {
#if defined(AXON_HAS_FMA)
    return _mm256_fmadd_ps(a, b, c);
#else
    return _mm256_add_ps(c, _mm256_mul_ps(a, b));
#endif
}

AXON_NODISCARD AXON_ALWAYS_INLINE __m256d fmadd256_pd(__m256d a, __m256d b, __m256d c) noexcept {
#if defined(AXON_HAS_FMA)
    return _mm256_fmadd_pd(a, b, c);
#else
    return _mm256_add_pd(c, _mm256_mul_pd(a, b));
#endif
}
#endif

#if defined(AXON_HAS_AVX512F)
AXON_NODISCARD AXON_ALWAYS_INLINE float hsum512_ps(__m512 v) noexcept {
    __m256 vlow  = _mm512_castps512_ps256(v);
    __m256 vhigh = _mm512_extractf32x8_ps(v, 1);
    return hsum256_ps(_mm256_add_ps(vlow, vhigh));
}

AXON_NODISCARD AXON_ALWAYS_INLINE double hsum512_pd(__m512d v) noexcept {
    __m256d vlow  = _mm512_castpd512_pd256(v);
    __m256d vhigh = _mm512_extractf64x4_pd(v, 1);
    return hsum256_pd(_mm256_add_pd(vlow, vhigh));
}

AXON_NODISCARD AXON_ALWAYS_INLINE int32_t hsum512_epi32(__m512i v) noexcept {
    __m256i vlow  = _mm512_castsi512_si256(v);
    __m256i vhigh = _mm512_extracti32x8_epi32(v, 1);
    return hsum256_epi32(_mm256_add_epi32(vlow, vhigh));
}
#endif

#endif // AXON_ARCH_X86_64

// Adds a 1D bias vector [N] to every row of a 2D matrix C [M, N] in-place
AXON_ALWAYS_INLINE void add_bias_rows_f32(
    float* AXON_RESTRICT c,
    const float* AXON_RESTRICT bias,
    size_t M,
    size_t N
) noexcept {
    for (size_t i = 0; i < M; ++i) {
        float* c_row = c + i * N;
        size_t j = 0;
#if defined(AXON_HAS_AVX512F)
        for (; j + 16 <= N; j += 16) {
            __m512 vc = _mm512_loadu_ps(c_row + j);
            __m512 vb = _mm512_loadu_ps(bias + j);
            _mm512_storeu_ps(c_row + j, _mm512_add_ps(vc, vb));
        }
#elif defined(AXON_HAS_AVX2)
        for (; j + 8 <= N; j += 8) {
            __m256 vc = _mm256_loadu_ps(c_row + j);
            __m256 vb = _mm256_loadu_ps(bias + j);
            _mm256_storeu_ps(c_row + j, _mm256_add_ps(vc, vb));
        }
#elif defined(AXON_HAS_NEON)
        for (; j + 4 <= N; j += 4) {
            float32x4_t vc = vld1q_f32(c_row + j);
            float32x4_t vb = vld1q_f32(bias + j);
            vst1q_f32(c_row + j, vaddq_f32(vc, vb));
        }
#endif
        for (; j < N; ++j) {
            c_row[j] += bias[j];
        }
    }
}

AXON_ALWAYS_INLINE void add_bias_rows_f64(
    double* AXON_RESTRICT c,
    const double* AXON_RESTRICT bias,
    size_t M,
    size_t N
) noexcept {
    for (size_t i = 0; i < M; ++i) {
        double* c_row = c + i * N;
        size_t j = 0;
#if defined(AXON_HAS_AVX512F)
        for (; j + 8 <= N; j += 8) {
            __m512d vc = _mm512_loadu_pd(c_row + j);
            __m512d vb = _mm512_loadu_pd(bias + j);
            _mm512_storeu_pd(c_row + j, _mm512_add_pd(vc, vb));
        }
#elif defined(AXON_HAS_AVX2)
        for (; j + 4 <= N; j += 4) {
            __m256d vc = _mm256_loadu_pd(c_row + j);
            __m256d vb = _mm256_loadu_pd(bias + j);
            _mm256_storeu_pd(c_row + j, _mm256_add_pd(vc, vb));
        }
#elif defined(AXON_HAS_NEON)
        for (; j + 2 <= N; j += 2) {
            float64x2_t vc = vld1q_f64(c_row + j);
            float64x2_t vb = vld1q_f64(bias + j);
            vst1q_f64(c_row + j, vaddq_f64(vc, vb));
        }
#endif
        for (; j < N; ++j) {
            c_row[j] += bias[j];
        }
    }
}

AXON_ALWAYS_INLINE void add_bias_rows_i32(
    int32_t* AXON_RESTRICT c,
    const int32_t* AXON_RESTRICT bias,
    size_t M,
    size_t N
) noexcept {
    for (size_t i = 0; i < M; ++i) {
        int32_t* c_row = c + i * N;
        size_t j = 0;
#if defined(AXON_HAS_AVX512F)
        for (; j + 16 <= N; j += 16) {
            __m512i vc = _mm512_loadu_si512(reinterpret_cast<const void*>(c_row + j));
            __m512i vb = _mm512_loadu_si512(reinterpret_cast<const void*>(bias + j));
            _mm512_storeu_si512(reinterpret_cast<void*>(c_row + j), _mm512_add_epi32(vc, vb));
        }
#elif defined(AXON_HAS_AVX2)
        for (; j + 8 <= N; j += 8) {
            __m256i vc = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c_row + j));
            __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias + j));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c_row + j), _mm256_add_epi32(vc, vb));
        }
#elif defined(AXON_HAS_NEON)
        for (; j + 4 <= N; j += 4) {
            int32x4_t vc = vld1q_s32(c_row + j);
            int32x4_t vb = vld1q_s32(bias + j);
            vst1q_s32(c_row + j, vaddq_s32(vc, vb));
        }
#endif
        for (; j < N; ++j) {
            c_row[j] += bias[j];
        }
    }
}

} // anonymous namespace

// ============================================================================
// 2. LOW-LEVEL BARE-METAL GEMM & DOT-PRODUCT KERNEL IMPLEMENTATIONS
// ============================================================================

namespace kernels {

// ----------------------------------------------------------------------------
// 1D Vector Inner Product (Dot Product) Kernels
// ----------------------------------------------------------------------------

float dot_kernel_f32(
    const float* AXON_RESTRICT a,
    const float* AXON_RESTRICT b,
    size_t count
) noexcept {
    if (AXON_UNLIKELY(count == 0)) {
        return 0.0f;
    }
    AXON_ASSERT(a != nullptr && b != nullptr);

    size_t i = 0;
    float sum = 0.0f;

#if defined(AXON_HAS_AVX512F)
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();

    for (; i + 64 <= count; i += 64) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i),      _mm512_loadu_ps(b + i),      acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), acc1);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(b + i + 32), acc2);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(b + i + 48), acc3);
    }
    acc0 = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    for (; i + 16 <= count; i += 16) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
    }
    sum = hsum512_ps(acc0);
#elif defined(AXON_HAS_AVX2)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    for (; i + 32 <= count; i += 32) {
        acc0 = fmadd256_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i),      acc0);
        acc1 = fmadd256_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8),  acc1);
        acc2 = fmadd256_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), acc2);
        acc3 = fmadd256_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), acc3);
    }
    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    for (; i + 8 <= count; i += 8) {
        acc0 = fmadd256_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    }
    sum = hsum256_ps(acc0);
#elif defined(AXON_HAS_NEON)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    for (; i + 16 <= count; i += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i),      vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    for (; i + 4 <= count; i += 4) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    sum = vaddvq_f32(acc0);
#endif

    float tail0 = 0.0f;
    float tail1 = 0.0f;
    for (; i + 2 <= count; i += 2) {
        tail0 += a[i]     * b[i];
        tail1 += a[i + 1] * b[i + 1];
    }
    sum += (tail0 + tail1);
    for (; i < count; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

double dot_kernel_f64(
    const double* AXON_RESTRICT a,
    const double* AXON_RESTRICT b,
    size_t count
) noexcept {
    if (AXON_UNLIKELY(count == 0)) {
        return 0.0;
    }
    AXON_ASSERT(a != nullptr && b != nullptr);

    size_t i = 0;
    double sum = 0.0;

#if defined(AXON_HAS_AVX512F)
    __m512d acc0 = _mm512_setzero_pd();
    __m512d acc1 = _mm512_setzero_pd();
    __m512d acc2 = _mm512_setzero_pd();
    __m512d acc3 = _mm512_setzero_pd();

    for (; i + 32 <= count; i += 32) {
        acc0 = _mm512_fmadd_pd(_mm512_loadu_pd(a + i),      _mm512_loadu_pd(b + i),      acc0);
        acc1 = _mm512_fmadd_pd(_mm512_loadu_pd(a + i + 8),  _mm512_loadu_pd(b + i + 8),  acc1);
        acc2 = _mm512_fmadd_pd(_mm512_loadu_pd(a + i + 16), _mm512_loadu_pd(b + i + 16), acc2);
        acc3 = _mm512_fmadd_pd(_mm512_loadu_pd(a + i + 24), _mm512_loadu_pd(b + i + 24), acc3);
    }
    acc0 = _mm512_add_pd(_mm512_add_pd(acc0, acc1), _mm512_add_pd(acc2, acc3));
    for (; i + 8 <= count; i += 8) {
        acc0 = _mm512_fmadd_pd(_mm512_loadu_pd(a + i), _mm512_loadu_pd(b + i), acc0);
    }
    sum = hsum512_pd(acc0);
#elif defined(AXON_HAS_AVX2)
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    __m256d acc2 = _mm256_setzero_pd();
    __m256d acc3 = _mm256_setzero_pd();

    for (; i + 16 <= count; i += 16) {
        acc0 = fmadd256_pd(_mm256_loadu_pd(a + i),      _mm256_loadu_pd(b + i),      acc0);
        acc1 = fmadd256_pd(_mm256_loadu_pd(a + i + 4),  _mm256_loadu_pd(b + i + 4),  acc1);
        acc2 = fmadd256_pd(_mm256_loadu_pd(a + i + 8),  _mm256_loadu_pd(b + i + 8),  acc2);
        acc3 = fmadd256_pd(_mm256_loadu_pd(a + i + 12), _mm256_loadu_pd(b + i + 12), acc3);
    }
    acc0 = _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3));
    for (; i + 4 <= count; i += 4) {
        acc0 = fmadd256_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i), acc0);
    }
    sum = hsum256_pd(acc0);
#elif defined(AXON_HAS_NEON)
    float64x2_t acc0 = vdupq_n_f64(0.0);
    float64x2_t acc1 = vdupq_n_f64(0.0);
    float64x2_t acc2 = vdupq_n_f64(0.0);
    float64x2_t acc3 = vdupq_n_f64(0.0);

    for (; i + 8 <= count; i += 8) {
        acc0 = vfmaq_f64(acc0, vld1q_f64(a + i),     vld1q_f64(b + i));
        acc1 = vfmaq_f64(acc1, vld1q_f64(a + i + 2), vld1q_f64(b + i + 2));
        acc2 = vfmaq_f64(acc2, vld1q_f64(a + i + 4), vld1q_f64(b + i + 4));
        acc3 = vfmaq_f64(acc3, vld1q_f64(a + i + 6), vld1q_f64(b + i + 6));
    }
    acc0 = vaddq_f64(vaddq_f64(acc0, acc1), vaddq_f64(acc2, acc3));
    for (; i + 2 <= count; i += 2) {
        acc0 = vfmaq_f64(acc0, vld1q_f64(a + i), vld1q_f64(b + i));
    }
    sum = vaddvq_f64(acc0);
#endif

    double tail0 = 0.0;
    double tail1 = 0.0;
    for (; i + 2 <= count; i += 2) {
        tail0 += a[i]     * b[i];
        tail1 += a[i + 1] * b[i + 1];
    }
    sum += (tail0 + tail1);
    for (; i < count; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

int32_t dot_kernel_i32(
    const int32_t* AXON_RESTRICT a,
    const int32_t* AXON_RESTRICT b,
    size_t count
) noexcept {
    if (AXON_UNLIKELY(count == 0)) {
        return 0;
    }
    AXON_ASSERT(a != nullptr && b != nullptr);

    size_t i = 0;
    int32_t sum = 0;

#if defined(AXON_HAS_AVX512F)
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    __m512i acc2 = _mm512_setzero_si512();
    __m512i acc3 = _mm512_setzero_si512();

    for (; i + 64 <= count; i += 64) {
        __m512i va0 = _mm512_loadu_si512(reinterpret_cast<const void*>(a + i));
        __m512i vb0 = _mm512_loadu_si512(reinterpret_cast<const void*>(b + i));
        __m512i va1 = _mm512_loadu_si512(reinterpret_cast<const void*>(a + i + 16));
        __m512i vb1 = _mm512_loadu_si512(reinterpret_cast<const void*>(b + i + 16));
        __m512i va2 = _mm512_loadu_si512(reinterpret_cast<const void*>(a + i + 32));
        __m512i vb2 = _mm512_loadu_si512(reinterpret_cast<const void*>(b + i + 32));
        __m512i va3 = _mm512_loadu_si512(reinterpret_cast<const void*>(a + i + 48));
        __m512i vb3 = _mm512_loadu_si512(reinterpret_cast<const void*>(b + i + 48));

        acc0 = _mm512_add_epi32(acc0, _mm512_mullo_epi32(va0, vb0));
        acc1 = _mm512_add_epi32(acc1, _mm512_mullo_epi32(va1, vb1));
        acc2 = _mm512_add_epi32(acc2, _mm512_mullo_epi32(va2, vb2));
        acc3 = _mm512_add_epi32(acc3, _mm512_mullo_epi32(va3, vb3));
    }
    acc0 = _mm512_add_epi32(_mm512_add_epi32(acc0, acc1), _mm512_add_epi32(acc2, acc3));
    for (; i + 16 <= count; i += 16) {
        __m512i va = _mm512_loadu_si512(reinterpret_cast<const void*>(a + i));
        __m512i vb = _mm512_loadu_si512(reinterpret_cast<const void*>(b + i));
        acc0 = _mm512_add_epi32(acc0, _mm512_mullo_epi32(va, vb));
    }
    sum = hsum512_epi32(acc0);
#elif defined(AXON_HAS_AVX2)
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();

    for (; i + 32 <= count; i += 32) {
        __m256i va0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        __m256i va1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 8));
        __m256i vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 8));
        __m256i va2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 16));
        __m256i vb2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 16));
        __m256i va3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 24));
        __m256i vb3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 24));

        acc0 = _mm256_add_epi32(acc0, _mm256_mullo_epi32(va0, vb0));
        acc1 = _mm256_add_epi32(acc1, _mm256_mullo_epi32(va1, vb1));
        acc2 = _mm256_add_epi32(acc2, _mm256_mullo_epi32(va2, vb2));
        acc3 = _mm256_add_epi32(acc3, _mm256_mullo_epi32(va3, vb3));
    }
    acc0 = _mm256_add_epi32(_mm256_add_epi32(acc0, acc1), _mm256_add_epi32(acc2, acc3));
    for (; i + 8 <= count; i += 8) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        acc0 = _mm256_add_epi32(acc0, _mm256_mullo_epi32(va, vb));
    }
    sum = hsum256_epi32(acc0);
#elif defined(AXON_HAS_NEON)
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    int32x4_t acc2 = vdupq_n_s32(0);
    int32x4_t acc3 = vdupq_n_s32(0);

    for (; i + 16 <= count; i += 16) {
        acc0 = vmlaq_s32(acc0, vld1q_s32(a + i),      vld1q_s32(b + i));
        acc1 = vmlaq_s32(acc1, vld1q_s32(a + i + 4),  vld1q_s32(b + i + 4));
        acc2 = vmlaq_s32(acc2, vld1q_s32(a + i + 8),  vld1q_s32(b + i + 8));
        acc3 = vmlaq_s32(acc3, vld1q_s32(a + i + 12), vld1q_s32(b + i + 12));
    }
    acc0 = vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3));
    for (; i + 4 <= count; i += 4) {
        acc0 = vmlaq_s32(acc0, vld1q_s32(a + i), vld1q_s32(b + i));
    }
    sum = vaddvq_s32(acc0);
#endif

    int32_t tail0 = 0;
    int32_t tail1 = 0;
    for (; i + 2 <= count; i += 2) {
        tail0 += a[i]     * b[i];
        tail1 += a[i + 1] * b[i + 1];
    }
    sum += (tail0 + tail1);
    for (; i < count; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

// ----------------------------------------------------------------------------
// 2D Contiguous Row-Major General Matrix Multiplication (GEMM)
// Computes: C[M, N] = A[M, K] * B[K, N]
// ----------------------------------------------------------------------------

void gemm_kernel_f32(
    const float* AXON_RESTRICT a,
    const float* AXON_RESTRICT b,
    float* AXON_RESTRICT c,
    size_t M,
    size_t K,
    size_t N
) noexcept {
    if (AXON_UNLIKELY(M == 0 || N == 0)) {
        return;
    }
    AXON_ASSERT(c != nullptr);
    axon::memory::zero_elements(c, M * N);

    if (AXON_UNLIKELY(K == 0)) {
        return;
    }
    AXON_ASSERT(a != nullptr && b != nullptr);

    for (size_t i0 = 0; i0 < M; i0 += GEMM_BLOCK_M) {
        const size_t i_end = (i0 + GEMM_BLOCK_M < M) ? (i0 + GEMM_BLOCK_M) : M;

        for (size_t k0 = 0; k0 < K; k0 += GEMM_BLOCK_K) {
            const size_t k_end = (k0 + GEMM_BLOCK_K < K) ? (k0 + GEMM_BLOCK_K) : K;

            for (size_t j0 = 0; j0 < N; j0 += GEMM_BLOCK_N) {
                const size_t j_end = (j0 + GEMM_BLOCK_N < N) ? (j0 + GEMM_BLOCK_N) : N;

                for (size_t i = i0; i < i_end; ++i) {
                    float* c_row = c + i * N;
                    const float* a_row = a + i * K;

                    for (size_t k = k0; k < k_end; ++k) {
                        const float a_val = a_row[k];
                        const float* b_row = b + k * N;
                        size_t j = j0;

#if defined(AXON_HAS_AVX512F)
                        const __m512 va = _mm512_set1_ps(a_val);
                        for (; j + 32 <= j_end; j += 32) {
                            __m512 vc0 = _mm512_loadu_ps(c_row + j);
                            __m512 vb0 = _mm512_loadu_ps(b_row + j);
                            __m512 vc1 = _mm512_loadu_ps(c_row + j + 16);
                            __m512 vb1 = _mm512_loadu_ps(b_row + j + 16);
                            _mm512_storeu_ps(c_row + j,      _mm512_fmadd_ps(va, vb0, vc0));
                            _mm512_storeu_ps(c_row + j + 16, _mm512_fmadd_ps(va, vb1, vc1));
                        }
                        for (; j + 16 <= j_end; j += 16) {
                            __m512 vc = _mm512_loadu_ps(c_row + j);
                            __m512 vb = _mm512_loadu_ps(b_row + j);
                            _mm512_storeu_ps(c_row + j, _mm512_fmadd_ps(va, vb, vc));
                        }
#elif defined(AXON_HAS_AVX2)
                        const __m256 va = _mm256_set1_ps(a_val);
                        for (; j + 32 <= j_end; j += 32) {
                            __m256 vc0 = _mm256_loadu_ps(c_row + j);
                            __m256 vb0 = _mm256_loadu_ps(b_row + j);
                            __m256 vc1 = _mm256_loadu_ps(c_row + j + 8);
                            __m256 vb1 = _mm256_loadu_ps(b_row + j + 8);
                            __m256 vc2 = _mm256_loadu_ps(c_row + j + 16);
                            __m256 vb2 = _mm256_loadu_ps(b_row + j + 16);
                            __m256 vc3 = _mm256_loadu_ps(c_row + j + 24);
                            __m256 vb3 = _mm256_loadu_ps(b_row + j + 24);
                            _mm256_storeu_ps(c_row + j,      fmadd256_ps(va, vb0, vc0));
                            _mm256_storeu_ps(c_row + j + 8,  fmadd256_ps(va, vb1, vc1));
                            _mm256_storeu_ps(c_row + j + 16, fmadd256_ps(va, vb2, vc2));
                            _mm256_storeu_ps(c_row + j + 24, fmadd256_ps(va, vb3, vc3));
                        }
                        for (; j + 8 <= j_end; j += 8) {
                            __m256 vc = _mm256_loadu_ps(c_row + j);
                            __m256 vb = _mm256_loadu_ps(b_row + j);
                            _mm256_storeu_ps(c_row + j, fmadd256_ps(va, vb, vc));
                        }
#elif defined(AXON_HAS_NEON)
                        const float32x4_t va = vdupq_n_f32(a_val);
                        for (; j + 16 <= j_end; j += 16) {
                            float32x4_t vc0 = vld1q_f32(c_row + j);
                            float32x4_t vb0 = vld1q_f32(b_row + j);
                            float32x4_t vc1 = vld1q_f32(c_row + j + 4);
                            float32x4_t vb1 = vld1q_f32(b_row + j + 4);
                            float32x4_t vc2 = vld1q_f32(c_row + j + 8);
                            float32x4_t vb2 = vld1q_f32(b_row + j + 8);
                            float32x4_t vc3 = vld1q_f32(c_row + j + 12);
                            float32x4_t vb3 = vld1q_f32(b_row + j + 12);
                            vst1q_f32(c_row + j,      vfmaq_f32(vc0, va, vb0));
                            vst1q_f32(c_row + j + 4,  vfmaq_f32(vc1, va, vb1));
                            vst1q_f32(c_row + j + 8,  vfmaq_f32(vc2, va, vb2));
                            vst1q_f32(c_row + j + 12, vfmaq_f32(vc3, va, vb3));
                        }
                        for (; j + 4 <= j_end; j += 4) {
                            float32x4_t vc = vld1q_f32(c_row + j);
                            float32x4_t vb = vld1q_f32(b_row + j);
                            vst1q_f32(c_row + j, vfmaq_f32(vc, va, vb));
                        }
#endif
                        for (; j < j_end; ++j) {
                            c_row[j] += a_val * b_row[j];
                        }
                    }
                }
            }
        }
    }
}

void gemm_kernel_f64(
    const double* AXON_RESTRICT a,
    const double* AXON_RESTRICT b,
    double* AXON_RESTRICT c,
    size_t M,
    size_t K,
    size_t N
) noexcept {
    if (AXON_UNLIKELY(M == 0 || N == 0)) {
        return;
    }
    AXON_ASSERT(c != nullptr);
    axon::memory::zero_elements(c, M * N);

    if (AXON_UNLIKELY(K == 0)) {
        return;
    }
    AXON_ASSERT(a != nullptr && b != nullptr);

    for (size_t i0 = 0; i0 < M; i0 += GEMM_BLOCK_M) {
        const size_t i_end = (i0 + GEMM_BLOCK_M < M) ? (i0 + GEMM_BLOCK_M) : M;

        for (size_t k0 = 0; k0 < K; k0 += GEMM_BLOCK_K) {
            const size_t k_end = (k0 + GEMM_BLOCK_K < K) ? (k0 + GEMM_BLOCK_K) : K;

            for (size_t j0 = 0; j0 < N; j0 += GEMM_BLOCK_N) {
                const size_t j_end = (j0 + GEMM_BLOCK_N < N) ? (j0 + GEMM_BLOCK_N) : N;

                for (size_t i = i0; i < i_end; ++i) {
                    double* c_row = c + i * N;
                    const double* a_row = a + i * K;

                    for (size_t k = k0; k < k_end; ++k) {
                        const double a_val = a_row[k];
                        const double* b_row = b + k * N;
                        size_t j = j0;

#if defined(AXON_HAS_AVX512F)
                        const __m512d va = _mm512_set1_pd(a_val);
                        for (; j + 16 <= j_end; j += 16) {
                            __m512d vc0 = _mm512_loadu_pd(c_row + j);
                            __m512d vb0 = _mm512_loadu_pd(b_row + j);
                            __m512d vc1 = _mm512_loadu_pd(c_row + j + 8);
                            __m512d vb1 = _mm512_loadu_pd(b_row + j + 8);
                            _mm512_storeu_pd(c_row + j,     _mm512_fmadd_pd(va, vb0, vc0));
                            _mm512_storeu_pd(c_row + j + 8, _mm512_fmadd_pd(va, vb1, vc1));
                        }
                        for (; j + 8 <= j_end; j += 8) {
                            __m512d vc = _mm512_loadu_pd(c_row + j);
                            __m512d vb = _mm512_loadu_pd(b_row + j);
                            _mm512_storeu_pd(c_row + j, _mm512_fmadd_pd(va, vb, vc));
                        }
#elif defined(AXON_HAS_AVX2)
                        const __m256d va = _mm256_set1_pd(a_val);
                        for (; j + 16 <= j_end; j += 16) {
                            __m256d vc0 = _mm256_loadu_pd(c_row + j);
                            __m256d vb0 = _mm256_loadu_pd(b_row + j);
                            __m256d vc1 = _mm256_loadu_pd(c_row + j + 4);
                            __m256d vb1 = _mm256_loadu_pd(b_row + j + 4);
                            __m256d vc2 = _mm256_loadu_pd(c_row + j + 8);
                            __m256d vb2 = _mm256_loadu_pd(b_row + j + 8);
                            __m256d vc3 = _mm256_loadu_pd(c_row + j + 12);
                            __m256d vb3 = _mm256_loadu_pd(b_row + j + 12);
                            _mm256_storeu_pd(c_row + j,      fmadd256_pd(va, vb0, vc0));
                            _mm256_storeu_pd(c_row + j + 4,  fmadd256_pd(va, vb1, vc1));
                            _mm256_storeu_pd(c_row + j + 8,  fmadd256_pd(va, vb2, vc2));
                            _mm256_storeu_pd(c_row + j + 12, fmadd256_pd(va, vb3, vc3));
                        }
                        for (; j + 4 <= j_end; j += 4) {
                            __m256d vc = _mm256_loadu_pd(c_row + j);
                            __m256d vb = _mm256_loadu_pd(b_row + j);
                            _mm256_storeu_pd(c_row + j, fmadd256_pd(va, vb, vc));
                        }
#elif defined(AXON_HAS_NEON)
                        const float64x2_t va = vdupq_n_f64(a_val);
                        for (; j + 8 <= j_end; j += 8) {
                            float64x2_t vc0 = vld1q_f64(c_row + j);
                            float64x2_t vb0 = vld1q_f64(b_row + j);
                            float64x2_t vc1 = vld1q_f64(c_row + j + 2);
                            float64x2_t vb1 = vld1q_f64(b_row + j + 2);
                            float64x2_t vc2 = vld1q_f64(c_row + j + 4);
                            float64x2_t vb2 = vld1q_f64(b_row + j + 4);
                            float64x2_t vc3 = vld1q_f64(c_row + j + 6);
                            float64x2_t vb3 = vld1q_f64(b_row + j + 6);
                            vst1q_f64(c_row + j,     vfmaq_f64(vc0, va, vb0));
                            vst1q_f64(c_row + j + 2, vfmaq_f64(vc1, va, vb1));
                            vst1q_f64(c_row + j + 4, vfmaq_f64(vc2, va, vb2));
                            vst1q_f64(c_row + j + 6, vfmaq_f64(vc3, va, vb3));
                        }
                        for (; j + 2 <= j_end; j += 2) {
                            float64x2_t vc = vld1q_f64(c_row + j);
                            float64x2_t vb = vld1q_f64(b_row + j);
                            vst1q_f64(c_row + j, vfmaq_f64(vc, va, vb));
                        }
#endif
                        for (; j < j_end; ++j) {
                            c_row[j] += a_val * b_row[j];
                        }
                    }
                }
            }
        }
    }
}

void gemm_kernel_i32(
    const int32_t* AXON_RESTRICT a,
    const int32_t* AXON_RESTRICT b,
    int32_t* AXON_RESTRICT c,
    size_t M,
    size_t K,
    size_t N
) noexcept {
    if (AXON_UNLIKELY(M == 0 || N == 0)) {
        return;
    }
    AXON_ASSERT(c != nullptr);
    axon::memory::zero_elements(c, M * N);

    if (AXON_UNLIKELY(K == 0)) {
        return;
    }
    AXON_ASSERT(a != nullptr && b != nullptr);

    for (size_t i0 = 0; i0 < M; i0 += GEMM_BLOCK_M) {
        const size_t i_end = (i0 + GEMM_BLOCK_M < M) ? (i0 + GEMM_BLOCK_M) : M;

        for (size_t k0 = 0; k0 < K; k0 += GEMM_BLOCK_K) {
            const size_t k_end = (k0 + GEMM_BLOCK_K < K) ? (k0 + GEMM_BLOCK_K) : K;

            for (size_t j0 = 0; j0 < N; j0 += GEMM_BLOCK_N) {
                const size_t j_end = (j0 + GEMM_BLOCK_N < N) ? (j0 + GEMM_BLOCK_N) : N;

                for (size_t i = i0; i < i_end; ++i) {
                    int32_t* c_row = c + i * N;
                    const int32_t* a_row = a + i * K;

                    for (size_t k = k0; k < k_end; ++k) {
                        const int32_t a_val = a_row[k];
                        const int32_t* b_row = b + k * N;
                        size_t j = j0;

#if defined(AXON_HAS_AVX512F)
                        const __m512i va = _mm512_set1_epi32(a_val);
                        for (; j + 32 <= j_end; j += 32) {
                            __m512i vc0 = _mm512_loadu_si512(reinterpret_cast<const void*>(c_row + j));
                            __m512i vb0 = _mm512_loadu_si512(reinterpret_cast<const void*>(b_row + j));
                            __m512i vc1 = _mm512_loadu_si512(reinterpret_cast<const void*>(c_row + j + 16));
                            __m512i vb1 = _mm512_loadu_si512(reinterpret_cast<const void*>(b_row + j + 16));
                            _mm512_storeu_si512(reinterpret_cast<void*>(c_row + j),      _mm512_add_epi32(vc0, _mm512_mullo_epi32(va, vb0)));
                            _mm512_storeu_si512(reinterpret_cast<void*>(c_row + j + 16), _mm512_add_epi32(vc1, _mm512_mullo_epi32(va, vb1)));
                        }
                        for (; j + 16 <= j_end; j += 16) {
                            __m512i vc = _mm512_loadu_si512(reinterpret_cast<const void*>(c_row + j));
                            __m512i vb = _mm512_loadu_si512(reinterpret_cast<const void*>(b_row + j));
                            _mm512_storeu_si512(reinterpret_cast<void*>(c_row + j), _mm512_add_epi32(vc, _mm512_mullo_epi32(va, vb)));
                        }
#elif defined(AXON_HAS_AVX2)
                        const __m256i va = _mm256_set1_epi32(a_val);
                        for (; j + 32 <= j_end; j += 32) {
                            __m256i vc0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c_row + j));
                            __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_row + j));
                            __m256i vc1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c_row + j + 8));
                            __m256i vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_row + j + 8));
                            __m256i vc2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c_row + j + 16));
                            __m256i vb2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_row + j + 16));
                            __m256i vc3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c_row + j + 24));
                            __m256i vb3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_row + j + 24));
                            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c_row + j),      _mm256_add_epi32(vc0, _mm256_mullo_epi32(va, vb0)));
                            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c_row + j + 8),  _mm256_add_epi32(vc1, _mm256_mullo_epi32(va, vb1)));
                            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c_row + j + 16), _mm256_add_epi32(vc2, _mm256_mullo_epi32(va, vb2)));
                            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c_row + j + 24), _mm256_add_epi32(vc3, _mm256_mullo_epi32(va, vb3)));
                        }
                        for (; j + 8 <= j_end; j += 8) {
                            __m256i vc = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c_row + j));
                            __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_row + j));
                            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c_row + j), _mm256_add_epi32(vc, _mm256_mullo_epi32(va, vb)));
                        }
#elif defined(AXON_HAS_NEON)
                        const int32x4_t va = vdupq_n_s32(a_val);
                        for (; j + 16 <= j_end; j += 16) {
                            int32x4_t vc0 = vld1q_s32(c_row + j);
                            int32x4_t vb0 = vld1q_s32(b_row + j);
                            int32x4_t vc1 = vld1q_s32(c_row + j + 4);
                            int32x4_t vb1 = vld1q_s32(b_row + j + 4);
                            int32x4_t vc2 = vld1q_s32(c_row + j + 8);
                            int32x4_t vb2 = vld1q_s32(b_row + j + 8);
                            int32x4_t vc3 = vld1q_s32(c_row + j + 12);
                            int32x4_t vb3 = vld1q_s32(b_row + j + 12);
                            vst1q_s32(c_row + j,      vmlaq_s32(vc0, va, vb0));
                            vst1q_s32(c_row + j + 4,  vmlaq_s32(vc1, va, vb1));
                            vst1q_s32(c_row + j + 8,  vmlaq_s32(vc2, va, vb2));
                            vst1q_s32(c_row + j + 12, vmlaq_s32(vc3, va, vb3));
                        }
                        for (; j + 4 <= j_end; j += 4) {
                            int32x4_t vc = vld1q_s32(c_row + j);
                            int32x4_t vb = vld1q_s32(b_row + j);
                            vst1q_s32(c_row + j, vmlaq_s32(vc, va, vb));
                        }
#endif
                        for (; j < j_end; ++j) {
                            c_row[j] += a_val * b_row[j];
                        }
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// 2D Transposed-B General Matrix Multiplication (GEMM_NT)
// Computes: C[M, N] = A[M, K] * (B_T[N, K])^T
// ----------------------------------------------------------------------------

void gemm_nt_kernel_f32(
    const float* AXON_RESTRICT a,
    const float* AXON_RESTRICT b_t,
    float* AXON_RESTRICT c,
    size_t M,
    size_t K,
    size_t N
) noexcept {
    if (AXON_UNLIKELY(M == 0 || N == 0)) {
        return;
    }
    AXON_ASSERT(c != nullptr);
    if (AXON_UNLIKELY(K == 0)) {
        axon::memory::zero_elements(c, M * N);
        return;
    }
    AXON_ASSERT(a != nullptr && b_t != nullptr);

    for (size_t i0 = 0; i0 < M; i0 += GEMM_BLOCK_M) {
        const size_t i_end = (i0 + GEMM_BLOCK_M < M) ? (i0 + GEMM_BLOCK_M) : M;

        for (size_t j0 = 0; j0 < N; j0 += GEMM_BLOCK_N) {
            const size_t j_end = (j0 + GEMM_BLOCK_N < N) ? (j0 + GEMM_BLOCK_N) : N;

            for (size_t i = i0; i < i_end; ++i) {
                const float* a_row = a + i * K;
                float* c_row = c + i * N;

                for (size_t j = j0; j < j_end; ++j) {
                    c_row[j] = dot_kernel_f32(a_row, b_t + j * K, K);
                }
            }
        }
    }
}

void gemm_nt_kernel_f64(
    const double* AXON_RESTRICT a,
    const double* AXON_RESTRICT b_t,
    double* AXON_RESTRICT c,
    size_t M,
    size_t K,
    size_t N
) noexcept {
    if (AXON_UNLIKELY(M == 0 || N == 0)) {
        return;
    }
    AXON_ASSERT(c != nullptr);
    if (AXON_UNLIKELY(K == 0)) {
        axon::memory::zero_elements(c, M * N);
        return;
    }
    AXON_ASSERT(a != nullptr && b_t != nullptr);

    for (size_t i0 = 0; i0 < M; i0 += GEMM_BLOCK_M) {
        const size_t i_end = (i0 + GEMM_BLOCK_M < M) ? (i0 + GEMM_BLOCK_M) : M;

        for (size_t j0 = 0; j0 < N; j0 += GEMM_BLOCK_N) {
            const size_t j_end = (j0 + GEMM_BLOCK_N < N) ? (j0 + GEMM_BLOCK_N) : N;

            for (size_t i = i0; i < i_end; ++i) {
                const double* a_row = a + i * K;
                double* c_row = c + i * N;

                for (size_t j = j0; j < j_end; ++j) {
                    c_row[j] = dot_kernel_f64(a_row, b_t + j * K, K);
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Fused 2D GEMM + Epilogue Bias Broadcast
// Computes: C[M, N] = A[M, K] * B[K, N] + bias[N]
// ----------------------------------------------------------------------------

void gemm_bias_kernel_f32(
    const float* AXON_RESTRICT a,
    const float* AXON_RESTRICT b,
    const float* AXON_RESTRICT bias,
    float* AXON_RESTRICT c,
    size_t M,
    size_t K,
    size_t N
) noexcept {
    if (AXON_UNLIKELY(M == 0 || N == 0)) {
        return;
    }
    AXON_ASSERT(c != nullptr);

    if (bias != nullptr) {
        for (size_t i = 0; i < M; ++i) {
            axon::memory::copy_elements(c + i * N, bias, N);
        }
    } else {
        axon::memory::zero_elements(c, M * N);
    }

    if (AXON_UNLIKELY(K == 0)) {
        return;
    }
    AXON_ASSERT(a != nullptr && b != nullptr);

    for (size_t i0 = 0; i0 < M; i0 += GEMM_BLOCK_M) {
        const size_t i_end = (i0 + GEMM_BLOCK_M < M) ? (i0 + GEMM_BLOCK_M) : M;

        for (size_t k0 = 0; k0 < K; k0 += GEMM_BLOCK_K) {
            const size_t k_end = (k0 + GEMM_BLOCK_K < K) ? (k0 + GEMM_BLOCK_K) : K;

            for (size_t j0 = 0; j0 < N; j0 += GEMM_BLOCK_N) {
                const size_t j_end = (j0 + GEMM_BLOCK_N < N) ? (j0 + GEMM_BLOCK_N) : N;

                for (size_t i = i0; i < i_end; ++i) {
                    float* c_row = c + i * N;
                    const float* a_row = a + i * K;

                    for (size_t k = k0; k < k_end; ++k) {
                        const float a_val = a_row[k];
                        const float* b_row = b + k * N;
                        size_t j = j0;

#if defined(AXON_HAS_AVX512F)
                        const __m512 va = _mm512_set1_ps(a_val);
                        for (; j + 32 <= j_end; j += 32) {
                            __m512 vc0 = _mm512_loadu_ps(c_row + j);
                            __m512 vb0 = _mm512_loadu_ps(b_row + j);
                            __m512 vc1 = _mm512_loadu_ps(c_row + j + 16);
                            __m512 vb1 = _mm512_loadu_ps(b_row + j + 16);
                            _mm512_storeu_ps(c_row + j,      _mm512_fmadd_ps(va, vb0, vc0));
                            _mm512_storeu_ps(c_row + j + 16, _mm512_fmadd_ps(va, vb1, vc1));
                        }
                        for (; j + 16 <= j_end; j += 16) {
                            __m512 vc = _mm512_loadu_ps(c_row + j);
                            __m512 vb = _mm512_loadu_ps(b_row + j);
                            _mm512_storeu_ps(c_row + j, _mm512_fmadd_ps(va, vb, vc));
                        }
#elif defined(AXON_HAS_AVX2)
                        const __m256 va = _mm256_set1_ps(a_val);
                        for (; j + 32 <= j_end; j += 32) {
                            __m256 vc0 = _mm256_loadu_ps(c_row + j);
                            __m256 vb0 = _mm256_loadu_ps(b_row + j);
                            __m256 vc1 = _mm256_loadu_ps(c_row + j + 8);
                            __m256 vb1 = _mm256_loadu_ps(b_row + j + 8);
                            __m256 vc2 = _mm256_loadu_ps(c_row + j + 16);
                            __m256 vb2 = _mm256_loadu_ps(b_row + j + 16);
                            __m256 vc3 = _mm256_loadu_ps(c_row + j + 24);
                            __m256 vb3 = _mm256_loadu_ps(b_row + j + 24);
                            _mm256_storeu_ps(c_row + j,      fmadd256_ps(va, vb0, vc0));
                            _mm256_storeu_ps(c_row + j + 8,  fmadd256_ps(va, vb1, vc1));
                            _mm256_storeu_ps(c_row + j + 16, fmadd256_ps(va, vb2, vc2));
                            _mm256_storeu_ps(c_row + j + 24, fmadd256_ps(va, vb3, vc3));
                        }
                        for (; j + 8 <= j_end; j += 8) {
                            __m256 vc = _mm256_loadu_ps(c_row + j);
                            __m256 vb = _mm256_loadu_ps(b_row + j);
                            _mm256_storeu_ps(c_row + j, fmadd256_ps(va, vb, vc));
                        }
#elif defined(AXON_HAS_NEON)
                        const float32x4_t va = vdupq_n_f32(a_val);
                        for (; j + 16 <= j_end; j += 16) {
                            float32x4_t vc0 = vld1q_f32(c_row + j);
                            float32x4_t vb0 = vld1q_f32(b_row + j);
                            float32x4_t vc1 = vld1q_f32(c_row + j + 4);
                            float32x4_t vb1 = vld1q_f32(b_row + j + 4);
                            float32x4_t vc2 = vld1q_f32(c_row + j + 8);
                            float32x4_t vb2 = vld1q_f32(b_row + j + 8);
                            float32x4_t vc3 = vld1q_f32(c_row + j + 12);
                            float32x4_t vb3 = vld1q_f32(b_row + j + 12);
                            vst1q_f32(c_row + j,      vfmaq_f32(vc0, va, vb0));
                            vst1q_f32(c_row + j + 4,  vfmaq_f32(vc1, va, vb1));
                            vst1q_f32(c_row + j + 8,  vfmaq_f32(vc2, va, vb2));
                            vst1q_f32(c_row + j + 12, vfmaq_f32(vc3, va, vb3));
                        }
                        for (; j + 4 <= j_end; j += 4) {
                            float32x4_t vc = vld1q_f32(c_row + j);
                            float32x4_t vb = vld1q_f32(b_row + j);
                            vst1q_f32(c_row + j, vfmaq_f32(vc, va, vb));
                        }
#endif
                        for (; j < j_end; ++j) {
                            c_row[j] += a_val * b_row[j];
                        }
                    }
                }
            }
        }
    }
}

} // namespace kernels

// ============================================================================
// 3. HIGH-LEVEL TENSOR MATRIX MULTIPLICATION & LINEAR ALGEBRA API
// ============================================================================

float dot(
    const axon::tensor::Tensor& a,
    const axon::tensor::Tensor& b
) noexcept {
    detail::validate_dot_op(a, b);
    AXON_ASSERT(a.dtype() == axon::core::DType::Float32);
    return kernels::dot_kernel_f32(
        a.data<float>(),
        b.data<float>(),
        static_cast<size_t>(a.dim(0))
    );
}

void matmul_out(
    const axon::tensor::Tensor& a,
    const axon::tensor::Tensor& b,
    axon::tensor::Tensor& out
) noexcept {
    if (a.ndim() == 1 && b.ndim() == 1) {
        detail::validate_dot_op(a, b);
        AXON_ASSERT(!out.empty());
        AXON_ASSERT(out.dtype() == a.dtype() && out.device() == a.device());
        AXON_ASSERT(out.ndim() == 0 || (out.ndim() == 1 && out.dim(0) == 1));

        const size_t count = static_cast<size_t>(a.dim(0));
        switch (out.dtype()) {
            case axon::core::DType::Float32:
                out.data<float>()[0] = kernels::dot_kernel_f32(a.data<float>(), b.data<float>(), count);
                break;
            case axon::core::DType::Float64:
                out.data<double>()[0] = kernels::dot_kernel_f64(a.data<double>(), b.data<double>(), count);
                break;
            case axon::core::DType::Int32:
                out.data<int32_t>()[0] = kernels::dot_kernel_i32(a.data<int32_t>(), b.data<int32_t>(), count);
                break;
            default:
                AXON_DEBUG_TRAP();
                break;
        }
        return;
    }

    if (a.ndim() == 2 && b.ndim() == 2) {
        detail::validate_gemm_2d_op(a, b, out);

        const size_t M = static_cast<size_t>(a.dim(0));
        const size_t K = static_cast<size_t>(a.dim(1));
        const size_t N = static_cast<size_t>(b.dim(1));
        const bool b_is_transposed = !b.is_contiguous() && detail::is_transposed_contiguous_2d(b);

        switch (out.dtype()) {
            case axon::core::DType::Float32:
                if (b_is_transposed) {
                    kernels::gemm_nt_kernel_f32(a.data<float>(), b.data<float>(), out.data<float>(), M, K, N);
                } else {
                    kernels::gemm_kernel_f32(a.data<float>(), b.data<float>(), out.data<float>(), M, K, N);
                }
                break;
            case axon::core::DType::Float64:
                if (b_is_transposed) {
                    kernels::gemm_nt_kernel_f64(a.data<double>(), b.data<double>(), out.data<double>(), M, K, N);
                } else {
                    kernels::gemm_kernel_f64(a.data<double>(), b.data<double>(), out.data<double>(), M, K, N);
                }
                break;
            case axon::core::DType::Int32:
                if (b_is_transposed) {
                    const int32_t* a_ptr = a.data<int32_t>();
                    const int32_t* bt_ptr = b.data<int32_t>();
                    int32_t* c_ptr = out.data<int32_t>();
                    for (size_t i = 0; i < M; ++i) {
                        for (size_t j = 0; j < N; ++j) {
                            c_ptr[i * N + j] = kernels::dot_kernel_i32(a_ptr + i * K, bt_ptr + j * K, K);
                        }
                    }
                } else {
                    kernels::gemm_kernel_i32(a.data<int32_t>(), b.data<int32_t>(), out.data<int32_t>(), M, K, N);
                }
                break;
            default:
                AXON_DEBUG_TRAP();
                break;
        }
        return;
    }

    if (a.ndim() == 3 && b.ndim() == 3) {
        detail::validate_bmm_3d_op(a, b, out);

        const size_t B = static_cast<size_t>(a.dim(0));
        const size_t M = static_cast<size_t>(a.dim(1));
        const size_t K = static_cast<size_t>(a.dim(2));
        const size_t N = static_cast<size_t>(b.dim(2));

        const size_t stride_a = M * K;
        const size_t stride_b = K * N;
        const size_t stride_c = M * N;

        switch (out.dtype()) {
            case axon::core::DType::Float32: {
                const float* a_ptr = a.data<float>();
                const float* b_ptr = b.data<float>();
                float* c_ptr = out.data<float>();
                for (size_t batch_idx = 0; batch_idx < B; ++batch_idx) {
                    kernels::gemm_kernel_f32(
                        a_ptr + batch_idx * stride_a,
                        b_ptr + batch_idx * stride_b,
                        c_ptr + batch_idx * stride_c,
                        M, K, N
                    );
                }
                break;
            }
            case axon::core::DType::Float64: {
                const double* a_ptr = a.data<double>();
                const double* b_ptr = b.data<double>();
                double* c_ptr = out.data<double>();
                for (size_t batch_idx = 0; batch_idx < B; ++batch_idx) {
                    kernels::gemm_kernel_f64(
                        a_ptr + batch_idx * stride_a,
                        b_ptr + batch_idx * stride_b,
                        c_ptr + batch_idx * stride_c,
                        M, K, N
                    );
                }
                break;
            }
            case axon::core::DType::Int32: {
                const int32_t* a_ptr = a.data<int32_t>();
                const int32_t* b_ptr = b.data<int32_t>();
                int32_t* c_ptr = out.data<int32_t>();
                for (size_t batch_idx = 0; batch_idx < B; ++batch_idx) {
                    kernels::gemm_kernel_i32(
                        a_ptr + batch_idx * stride_a,
                        b_ptr + batch_idx * stride_b,
                        c_ptr + batch_idx * stride_c,
                        M, K, N
                    );
                }
                break;
            }
            default:
                AXON_DEBUG_TRAP();
                break;
        }
        return;
    }

    AXON_DEBUG_TRAP();
}

axon::tensor::Tensor matmul(
    const axon::tensor::Tensor& a,
    const axon::tensor::Tensor& b
) noexcept {
    if (a.ndim() == 1 && b.ndim() == 1) {
        const int64_t out_dims[1] = {1};
        axon::tensor::Tensor out(out_dims, 1, a.dtype(), a.device());
        matmul_out(a, b, out);
        return out;
    }

    if (a.ndim() == 2 && b.ndim() == 2) {
        const int64_t out_dims[2] = {a.dim(0), b.dim(1)};
        axon::tensor::Tensor out(out_dims, 2, a.dtype(), a.device());
        matmul_out(a, b, out);
        return out;
    }

    if (a.ndim() == 3 && b.ndim() == 3) {
        const int64_t out_dims[3] = {a.dim(0), a.dim(1), b.dim(2)};
        axon::tensor::Tensor out(out_dims, 3, a.dtype(), a.device());
        matmul_out(a, b, out);
        return out;
    }

    AXON_DEBUG_TRAP();
    return axon::tensor::Tensor();
}

void linear_out(
    const axon::tensor::Tensor& input,
    const axon::tensor::Tensor& weight,
    const axon::tensor::Tensor* bias,
    axon::tensor::Tensor& out
) noexcept {
    AXON_ASSERT(!input.empty() && !weight.empty() && !out.empty());
    AXON_ASSERT(weight.ndim() == 2);
    AXON_ASSERT(input.dtype() == weight.dtype() && input.dtype() == out.dtype());
    AXON_ASSERT(input.device() == weight.device() && input.device() == out.device());
    AXON_ASSERT(input.is_contiguous() && out.is_contiguous());
    AXON_ASSERT(weight.is_contiguous() || detail::is_transposed_contiguous_2d(weight));

    size_t total_M = 0;
    size_t K = 0;
    const size_t N = static_cast<size_t>(weight.dim(1));

    if (input.ndim() == 2) {
        AXON_ASSERT(out.ndim() == 2);
        AXON_ASSERT(input.dim(1) == weight.dim(0));
        AXON_ASSERT(out.dim(0) == input.dim(0) && out.dim(1) == weight.dim(1));
        total_M = static_cast<size_t>(input.dim(0));
        K = static_cast<size_t>(input.dim(1));
    } else if (input.ndim() == 3) {
        AXON_ASSERT(out.ndim() == 3);
        AXON_ASSERT(input.dim(2) == weight.dim(0));
        AXON_ASSERT(out.dim(0) == input.dim(0) && out.dim(1) == input.dim(1) && out.dim(2) == weight.dim(1));
        total_M = static_cast<size_t>(input.dim(0)) * static_cast<size_t>(input.dim(1));
        K = static_cast<size_t>(input.dim(2));
    } else {
        AXON_DEBUG_TRAP();
        return;
    }

    if (bias != nullptr) {
        AXON_ASSERT(!bias->empty());
        AXON_ASSERT(bias->ndim() == 1 && bias->dim(0) == weight.dim(1));
        AXON_ASSERT(bias->dtype() == out.dtype() && bias->device() == out.device());
        AXON_ASSERT(bias->is_contiguous());
    }

    const bool w_is_transposed = !weight.is_contiguous() && detail::is_transposed_contiguous_2d(weight);

    switch (out.dtype()) {
        case axon::core::DType::Float32: {
            const float* bias_ptr = (bias != nullptr) ? bias->data<float>() : nullptr;
            if (w_is_transposed) {
                kernels::gemm_nt_kernel_f32(input.data<float>(), weight.data<float>(), out.data<float>(), total_M, K, N);
                if (bias_ptr != nullptr) {
                    add_bias_rows_f32(out.data<float>(), bias_ptr, total_M, N);
                }
            } else {
                kernels::gemm_bias_kernel_f32(input.data<float>(), weight.data<float>(), bias_ptr, out.data<float>(), total_M, K, N);
            }
            break;
        }
        case axon::core::DType::Float64: {
            const double* bias_ptr = (bias != nullptr) ? bias->data<double>() : nullptr;
            if (w_is_transposed) {
                kernels::gemm_nt_kernel_f64(input.data<double>(), weight.data<double>(), out.data<double>(), total_M, K, N);
            } else {
                kernels::gemm_kernel_f64(input.data<double>(), weight.data<double>(), out.data<double>(), total_M, K, N);
            }
            if (bias_ptr != nullptr) {
                add_bias_rows_f64(out.data<double>(), bias_ptr, total_M, N);
            }
            break;
        }
        case axon::core::DType::Int32: {
            const int32_t* bias_ptr = (bias != nullptr) ? bias->data<int32_t>() : nullptr;
            if (w_is_transposed) {
                const int32_t* in_ptr = input.data<int32_t>();
                const int32_t* wt_ptr = weight.data<int32_t>();
                int32_t* out_ptr = out.data<int32_t>();
                for (size_t i = 0; i < total_M; ++i) {
                    for (size_t j = 0; j < N; ++j) {
                        out_ptr[i * N + j] = kernels::dot_kernel_i32(in_ptr + i * K, wt_ptr + j * K, K);
                    }
                }
            } else {
                kernels::gemm_kernel_i32(input.data<int32_t>(), weight.data<int32_t>(), out.data<int32_t>(), total_M, K, N);
            }
            if (bias_ptr != nullptr) {
                add_bias_rows_i32(out.data<int32_t>(), bias_ptr, total_M, N);
            }
            break;
        }
        default:
            AXON_DEBUG_TRAP();
            break;
    }
}

axon::tensor::Tensor linear(
    const axon::tensor::Tensor& input,
    const axon::tensor::Tensor& weight,
    const axon::tensor::Tensor* bias
) noexcept {
    AXON_ASSERT(weight.ndim() == 2);

    if (input.ndim() == 2) {
        const int64_t out_dims[2] = {input.dim(0), weight.dim(1)};
        axon::tensor::Tensor out(out_dims, 2, input.dtype(), input.device());
        linear_out(input, weight, bias, out);
        return out;
    }

    if (input.ndim() == 3) {
        const int64_t out_dims[3] = {input.dim(0), input.dim(1), weight.dim(1)};
        axon::tensor::Tensor out(out_dims, 3, input.dtype(), input.device());
        linear_out(input, weight, bias, out);
        return out;
    }

    AXON_DEBUG_TRAP();
    return axon::tensor::Tensor();
}

} // namespace ops
} // namespace axon
