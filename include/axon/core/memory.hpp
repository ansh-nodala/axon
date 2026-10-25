#ifndef AXON_CORE_MEMORY_HPP
#define AXON_CORE_MEMORY_HPP

#include "axon/core/axon_macros.hpp"
#include <stddef.h>
#include <stdint.h>
#include <string.h>

namespace axon {
namespace memory {

// ============================================================================
// 1. ALIGNMENT ARITHMETIC & VALIDATION UTILITIES
// ============================================================================

// Compile-time check verifying that an alignment value is a valid power of two
AXON_NODISCARD constexpr AXON_ALWAYS_INLINE bool is_power_of_two(size_t value) noexcept {
    return (value != 0) && ((value & (value - 1)) == 0);
}

// Rounds a byte size up to the nearest multiple of the specified power-of-two alignment
AXON_NODISCARD constexpr AXON_ALWAYS_INLINE size_t align_up(size_t size, size_t alignment = AXON_SIMD_ALIGNMENT) noexcept {
    AXON_ASSERT(is_power_of_two(alignment));
    return (size + (alignment - 1)) & ~(alignment - 1);
}

// Checks whether a raw memory address satisfies the specified byte alignment
AXON_NODISCARD AXON_ALWAYS_INLINE bool is_aligned(const void* ptr, size_t alignment = AXON_SIMD_ALIGNMENT) noexcept {
    AXON_ASSERT(is_power_of_two(alignment));
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

// ============================================================================
// 2. CORE SIMD-ALIGNED MEMORY ALLOCATION API
// ============================================================================

// Allocates a contiguous block of uninitialized memory aligned to `alignment` bytes.
// Guaranteed to be compatible with AVX-512, AVX2, and ARM NEON vector load/store instructions.
AXON_NODISCARD AXON_API void* allocate_aligned(size_t size, size_t alignment = AXON_SIMD_ALIGNMENT);

// Releases a memory block previously allocated via `allocate_aligned` or `allocate_elements`.
// Passing nullptr is a safe no-op.
AXON_API void free_aligned(void* ptr);

// ============================================================================
// 3. TYPED ALLOCATION HELPERS
// ============================================================================

// Allocates an aligned contiguous buffer for `count` elements of type `T` with overflow protection.
template <typename T>
AXON_NODISCARD AXON_ALWAYS_INLINE T* allocate_elements(size_t count, size_t alignment = AXON_SIMD_ALIGNMENT) {
    if (AXON_UNLIKELY(count == 0)) {
        return nullptr;
    }

    // Guard against multiplicative size_t overflow prior to allocation
    if (AXON_UNLIKELY(count > (SIZE_MAX / sizeof(T)))) {
        AXON_DEBUG_TRAP();
        return nullptr;
    }

    return static_cast<T*>(allocate_aligned(count * sizeof(T), alignment));
}

// Releases a typed buffer previously allocated via `allocate_elements<T>`
template <typename T>
AXON_ALWAYS_INLINE void free_elements(T* ptr) {
    free_aligned(static_cast<void*>(ptr));
}

// ============================================================================
// 4. LOW-LEVEL MEMORY OPERATIONS
// ============================================================================

// Copies `bytes` from non-overlapping `src` to `dst` with restrict pointer semantics
AXON_ALWAYS_INLINE void copy_memory(void* AXON_RESTRICT dst, const void* AXON_RESTRICT src, size_t bytes) {
    if (AXON_UNLIKELY(bytes == 0 || dst == src)) {
        return;
    }
    AXON_ASSERT(dst != nullptr);
    AXON_ASSERT(src != nullptr);
    memcpy(dst, src, bytes);
}

// Copies `bytes` from SIMD-aligned `src` to SIMD-aligned `dst`
AXON_ALWAYS_INLINE void copy_memory_aligned(void* AXON_RESTRICT dst, const void* AXON_RESTRICT src, size_t bytes) {
    if (AXON_UNLIKELY(bytes == 0 || dst == src)) {
        return;
    }
    AXON_ASSERT(dst != nullptr);
    AXON_ASSERT(src != nullptr);
    AXON_ASSERT(is_aligned(dst, AXON_SIMD_ALIGNMENT));
    AXON_ASSERT(is_aligned(src, AXON_SIMD_ALIGNMENT));

    void* aligned_dst = AXON_ASSUME_ALIGNED(dst, AXON_SIMD_ALIGNMENT);
    const void* aligned_src = AXON_ASSUME_ALIGNED(src, AXON_SIMD_ALIGNMENT);
    memcpy(aligned_dst, aligned_src, bytes);
}

// Zero-initializes `bytes` of memory starting at `dst`
AXON_ALWAYS_INLINE void zero_memory(void* AXON_RESTRICT dst, size_t bytes) {
    if (AXON_UNLIKELY(bytes == 0)) {
        return;
    }
    AXON_ASSERT(dst != nullptr);
    memset(dst, 0, bytes);
}

// Zero-initializes `bytes` of SIMD-aligned memory starting at `dst`
AXON_ALWAYS_INLINE void zero_memory_aligned(void* AXON_RESTRICT dst, size_t bytes) {
    if (AXON_UNLIKELY(bytes == 0)) {
        return;
    }
    AXON_ASSERT(dst != nullptr);
    AXON_ASSERT(is_aligned(dst, AXON_SIMD_ALIGNMENT));

    void* aligned_dst = AXON_ASSUME_ALIGNED(dst, AXON_SIMD_ALIGNMENT);
    memset(aligned_dst, 0, bytes);
}

// Typed wrapper to copy `count` elements of type `T` from `src` to `dst`
template <typename T>
AXON_ALWAYS_INLINE void copy_elements(T* AXON_RESTRICT dst, const T* AXON_RESTRICT src, size_t count) {
    copy_memory(static_cast<void*>(dst), static_cast<const void*>(src), count * sizeof(T));
}

// Typed wrapper to zero-initialize `count` elements of type `T` at `dst`
template <typename T>
AXON_ALWAYS_INLINE void zero_elements(T* AXON_RESTRICT dst, size_t count) {
    zero_memory(static_cast<void*>(dst), count * sizeof(T));
}

} // namespace memory
} // namespace axon

#endif // AXON_CORE_MEMORY_HPP
