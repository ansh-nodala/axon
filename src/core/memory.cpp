#include "axon/core/memory.hpp"
#include <stdlib.h>

#if defined(AXON_OS_WINDOWS)
    #include <malloc.h>
#endif

namespace axon {
namespace memory {

// ============================================================================
// OS-LEVEL SIMD-ALIGNED MEMORY ALLOCATOR IMPLEMENTATION
// ============================================================================

void* allocate_aligned(size_t size, size_t alignment) {
    if (AXON_UNLIKELY(size == 0)) {
        return nullptr;
    }

    // Ensure alignment is at least sizeof(void*) as required by POSIX posix_memalign
    const size_t min_alignment = sizeof(void*);
    const size_t effective_alignment = (alignment < min_alignment) ? min_alignment : alignment;

    // Alignment must strictly be a power of two
    if (AXON_UNLIKELY(!is_power_of_two(effective_alignment))) {
        AXON_DEBUG_TRAP();
        return nullptr;
    }

    void* ptr = nullptr;

#if defined(AXON_OS_WINDOWS)
    ptr = _aligned_malloc(size, effective_alignment);
#elif defined(AXON_OS_POSIX)
    const int result = ::posix_memalign(&ptr, effective_alignment, size);
    if (AXON_UNLIKELY(result != 0)) {
        ptr = nullptr;
    }
#else
    const size_t aligned_size = align_up(size, effective_alignment);
    ptr = ::aligned_alloc(effective_alignment, aligned_size);
#endif

    // Fail-fast architecture: trap immediately on out-of-memory or invalid allocation state
    if (AXON_UNLIKELY(ptr == nullptr)) {
        AXON_DEBUG_TRAP();
        return nullptr;
    }

    AXON_ASSERT(is_aligned(ptr, effective_alignment));
    return ptr;
}

void free_aligned(void* ptr) {
    if (AXON_UNLIKELY(ptr == nullptr)) {
        return;
    }

#if defined(AXON_OS_WINDOWS)
    _aligned_free(ptr);
#else
    ::free(ptr);
#endif
}

} // namespace memory
} // namespace axon
