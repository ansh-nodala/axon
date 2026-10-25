#ifndef AXON_CORE_MACROS_HPP
#define AXON_CORE_MACROS_HPP

// ============================================================================
// 1. OS & COMPILER DETECTION
// ============================================================================

// Compiler Detection (Note: Clang on Windows also defines _MSC_VER, so check __clang__ first)
#if defined(__clang__)
    #define AXON_COMPILER_CLANG 1
    #define AXON_COMPILER_NAME "Clang"
    #define AXON_COMPILER_VERSION (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#elif defined(_MSC_VER)
    #define AXON_COMPILER_MSVC 1
    #define AXON_COMPILER_NAME "MSVC"
    #define AXON_COMPILER_VERSION _MSC_FULL_VER
#elif defined(__GNUC__) || defined(__GNUG__)
    #define AXON_COMPILER_GCC 1
    #define AXON_COMPILER_NAME "GCC"
    #define AXON_COMPILER_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#else
    #define AXON_COMPILER_UNKNOWN 1
    #define AXON_COMPILER_NAME "Unknown"
    #define AXON_COMPILER_VERSION 0
#endif

// Platform / OS Detection (Note: Android defines __linux__, so check __ANDROID__ first)
#if defined(_WIN32) || defined(_WIN64)
    #define AXON_OS_WINDOWS 1
    #define AXON_OS_NAME "Windows"
#elif defined(__ANDROID__)
    #define AXON_OS_ANDROID 1
    #define AXON_OS_POSIX 1
    #define AXON_OS_NAME "Android"
#elif defined(__linux__)
    #define AXON_OS_LINUX 1
    #define AXON_OS_POSIX 1
    #define AXON_OS_NAME "Linux"
#elif defined(__APPLE__) && defined(__MACH__)
    #define AXON_OS_MACOS 1
    #define AXON_OS_POSIX 1
    #define AXON_OS_NAME "macOS"
#else
    #define AXON_OS_UNKNOWN 1
    #define AXON_OS_NAME "Unknown"
#endif

// ============================================================================
// 2. ARCHITECTURE & SIMD PRE-PROCESSOR FLAGS
// ============================================================================

// Architecture Detection
#if defined(__x86_64__) || defined(_M_X64)
    #define AXON_ARCH_X86_64 1
    #define AXON_ARCH_NAME "x86_64"
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define AXON_ARCH_ARM64 1
    #define AXON_ARCH_NAME "ARM64"
#else
    #define AXON_ARCH_UNKNOWN 1
    #define AXON_ARCH_NAME "Unknown"
#endif

// x86_64 SIMD Capability Flags
#if defined(AXON_ARCH_X86_64)
    #if defined(__SSE2__) || defined(AXON_COMPILER_MSVC)
        #define AXON_HAS_SSE2 1
    #endif
    #if defined(__SSE4_1__) || defined(__AVX__) || defined(__AVX2__)
        #define AXON_HAS_SSE4_1 1
    #endif
    #if defined(__SSE4_2__) || defined(__AVX__) || defined(__AVX2__)
        #define AXON_HAS_SSE4_2 1
    #endif
    #if defined(__AVX__) || defined(__AVX2__)
        #define AXON_HAS_AVX 1
    #endif
    #if defined(__AVX2__)
        #define AXON_HAS_AVX2 1
    #endif
    #if defined(__FMA__) || (defined(AXON_COMPILER_MSVC) && defined(__AVX2__))
        #define AXON_HAS_FMA 1
    #endif
    #if defined(__F16C__) || (defined(AXON_COMPILER_MSVC) && defined(__AVX2__))
        #define AXON_HAS_F16C 1
    #endif
    #if defined(__AVX512F__)
        #define AXON_HAS_AVX512F 1
        #define AXON_HAS_AVX512 1
    #endif
    #if defined(__AVX512BW__)
        #define AXON_HAS_AVX512BW 1
    #endif
    #if defined(__AVX512DQ__)
        #define AXON_HAS_AVX512DQ 1
    #endif
    #if defined(__AVX512VL__)
        #define AXON_HAS_AVX512VL 1
    #endif
    #if defined(__AVX512VNNI__)
        #define AXON_HAS_AVX512_VNNI 1
    #endif
    #if defined(__AVX512BF16__)
        #define AXON_HAS_AVX512_BF16 1
    #endif
#endif

// ARM64 SIMD Capability Flags
#if defined(AXON_ARCH_ARM64)
    #if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(AXON_COMPILER_MSVC)
        #define AXON_HAS_NEON 1
    #endif
    #if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        #define AXON_HAS_NEON_FP16 1
    #endif
    #if defined(__ARM_FEATURE_DOTPROD)
        #define AXON_HAS_NEON_DOTPROD 1
    #endif
    #if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
        #define AXON_HAS_NEON_BF16 1
    #endif
    #if defined(__ARM_FEATURE_MATMUL_INT8)
        #define AXON_HAS_NEON_I8MM 1
    #endif
    #if defined(__ARM_FEATURE_SVE)
        #define AXON_HAS_SVE 1
    #endif
    #if defined(__ARM_FEATURE_SVE2)
        #define AXON_HAS_SVE2 1
    #endif
#endif

// ============================================================================
// 3. ULTRA-OPTIMIZATION COMPILER HINTS
// ============================================================================

// Forced Inlining and No-Inlining Directives
#if defined(AXON_COMPILER_MSVC)
    #define AXON_ALWAYS_INLINE __forceinline
    #define AXON_NOINLINE __declspec(noinline)
    #define AXON_FLATTEN
    #define AXON_HOT
    #define AXON_COLD
#elif defined(AXON_COMPILER_GCC) || defined(AXON_COMPILER_CLANG)
    #define AXON_ALWAYS_INLINE inline __attribute__((always_inline))
    #define AXON_NOINLINE __attribute__((noinline))
    #define AXON_FLATTEN __attribute__((flatten))
    #define AXON_HOT __attribute__((hot))
    #define AXON_COLD __attribute__((cold))
#else
    #define AXON_ALWAYS_INLINE inline
    #define AXON_NOINLINE
    #define AXON_FLATTEN
    #define AXON_HOT
    #define AXON_COLD
#endif

// Pointer Non-Aliasing Hint (Restrict Semantics)
#if defined(AXON_COMPILER_MSVC)
    #define AXON_RESTRICT __restrict
#elif defined(AXON_COMPILER_GCC) || defined(AXON_COMPILER_CLANG)
    #define AXON_RESTRICT __restrict__
#else
    #define AXON_RESTRICT
#endif

// Pointer Alignment Assumption for Auto-Vectorizer & SIMD Load/Store Generation
#if defined(AXON_COMPILER_GCC) || defined(AXON_COMPILER_CLANG)
    #define AXON_ASSUME_ALIGNED(ptr, alignment) __builtin_assume_aligned((ptr), (alignment))
#else
    #define AXON_ASSUME_ALIGNED(ptr, alignment) (ptr)
#endif

// Optimizer Value Assumption Hints
#if defined(AXON_COMPILER_MSVC)
    #define AXON_ASSUME(cond) __assume(cond)
#elif defined(AXON_COMPILER_CLANG)
    #define AXON_ASSUME(cond) __builtin_assume(cond)
#elif defined(AXON_COMPILER_GCC)
    #define AXON_ASSUME(cond) do { if (!(cond)) __builtin_unreachable(); } while (0)
#else
    #define AXON_ASSUME(cond) do { } while (0)
#endif

// Static Branch Prediction Hints
#if defined(AXON_COMPILER_GCC) || defined(AXON_COMPILER_CLANG)
    #define AXON_LIKELY(x)   __builtin_expect(!!(x), 1)
    #define AXON_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define AXON_LIKELY(x)   (x)
    #define AXON_UNLIKELY(x) (x)
#endif

// Hardware Cache Prefetch Hints (L1 Temporal Locality by Default)
#if defined(AXON_COMPILER_GCC) || defined(AXON_COMPILER_CLANG)
    #define AXON_PREFETCH_READ(ptr)  __builtin_prefetch((ptr), 0, 3)
    #define AXON_PREFETCH_WRITE(ptr) __builtin_prefetch((ptr), 1, 3)
#else
    #define AXON_PREFETCH_READ(ptr)  ((void)(ptr))
    #define AXON_PREFETCH_WRITE(ptr) ((void)(ptr))
#endif

// ============================================================================
// 4. CACHE & MEMORY ALIGNMENT
// ============================================================================

// Standard 64-byte hardware cache line size to eliminate false sharing
#define AXON_CACHE_LINE_SIZE 64

// Default alignment for SIMD tensor buffers (compatible with AVX-512, AVX2, and NEON)
#define AXON_SIMD_ALIGNMENT 64

// Struct and Variable Alignment Directive
#if defined(__cplusplus)
    #define AXON_ALIGN(n) alignas(n)
#elif defined(AXON_COMPILER_MSVC)
    #define AXON_ALIGN(n) __declspec(align(n))
#elif defined(AXON_COMPILER_GCC) || defined(AXON_COMPILER_CLANG)
    #define AXON_ALIGN(n) __attribute__((aligned(n)))
#else
    #define AXON_ALIGN(n)
#endif

// Convenience Alignment Shorthands
#define AXON_CACHE_ALIGNED AXON_ALIGN(AXON_CACHE_LINE_SIZE)
#define AXON_SIMD_ALIGNED  AXON_ALIGN(AXON_SIMD_ALIGNMENT)

// ============================================================================
// 5. VISIBILITY & API EXPORTS
// ============================================================================

#if defined(AXON_STATIC)
    #define AXON_API
    #define AXON_LOCAL
#elif defined(AXON_OS_WINDOWS)
    #if defined(AXON_EXPORTS) || defined(AXON_BUILD_SHARED)
        #define AXON_API __declspec(dllexport)
    #else
        #define AXON_API __declspec(dllimport)
    #endif
    #define AXON_LOCAL
#elif defined(AXON_COMPILER_GCC) || defined(AXON_COMPILER_CLANG)
    #define AXON_API   __attribute__((visibility("default")))
    #define AXON_LOCAL __attribute__((visibility("hidden")))
#else
    #define AXON_API
    #define AXON_LOCAL
#endif

// ============================================================================
// 6. BARE-METAL DEBUG TRAP & DIAGNOSTICS (ZERO STDLIB DEPENDENCY)
// ============================================================================

// Immediate Hardware Trap Instruction
#if defined(AXON_COMPILER_MSVC)
    #define AXON_DEBUG_TRAP() __debugbreak()
#elif defined(AXON_COMPILER_GCC) || defined(AXON_COMPILER_CLANG)
    #define AXON_DEBUG_TRAP() __builtin_trap()
#else
    #define AXON_DEBUG_TRAP() (*(volatile int*)0 = 0)
#endif

// Unreachable Code Path Hint
#if defined(AXON_COMPILER_MSVC)
    #define AXON_UNREACHABLE() __assume(0)
#elif defined(AXON_COMPILER_GCC) || defined(AXON_COMPILER_CLANG)
    #define AXON_UNREACHABLE() __builtin_unreachable()
#else
    #define AXON_UNREACHABLE() AXON_DEBUG_TRAP()
#endif

// Zero-Overhead Bare-Metal Assertions
#if defined(NDEBUG) && !defined(AXON_ENABLE_ASSERTS)
    #define AXON_ASSERT(cond) ((void)0)
#else
    #define AXON_ASSERT(cond) \
        do { \
            if (AXON_UNLIKELY(!(cond))) { \
                AXON_DEBUG_TRAP(); \
            } \
        } while (0)
#endif

// Always-Active Runtime Verification (Executes in both Debug and Release builds)
#define AXON_VERIFY(cond) \
    do { \
        if (AXON_UNLIKELY(!(cond))) { \
            AXON_DEBUG_TRAP(); \
        } \
    } while (0)

// Utility Attributes & Macros
#define AXON_NODISCARD [[nodiscard]]
#define AXON_MAYBE_UNUSED [[maybe_unused]]
#define AXON_UNUSED(x) ((void)(x))

#define AXON_STRINGIFY_IMPL(x) #x
#define AXON_STRINGIFY(x) AXON_STRINGIFY_IMPL(x)

#define AXON_CONCAT_IMPL(a, b) a##b
#define AXON_CONCAT(a, b) AXON_CONCAT_IMPL(a, b)

#endif // AXON_CORE_MACROS_HPP
