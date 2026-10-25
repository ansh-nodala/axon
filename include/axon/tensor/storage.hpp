#ifndef AXON_TENSOR_STORAGE_HPP
#define AXON_TENSOR_STORAGE_HPP

#include "axon/core/axon_macros.hpp"
#include "axon/core/memory.hpp"
#include "axon/core/types.hpp"

namespace axon {
namespace tensor {

// ============================================================================
// CONTIGUOUS SIMD-ALIGNED TENSOR STORAGE ENGINE
// ============================================================================

class AXON_API Storage {
private:
    void* data_ptr_;
    size_t num_elements_;
    axon::core::DType dtype_;
    axon::core::DeviceType device_;
    bool owns_memory_;

public:
    // Default constructor: creates an empty, unallocated storage handle
    constexpr Storage() noexcept
        : data_ptr_(nullptr),
          num_elements_(0),
          dtype_(axon::core::DType::Float32),
          device_(axon::core::DeviceType::CPU),
          owns_memory_(false) {}

    // Owning allocation constructor: allocates a 64-byte SIMD-aligned contiguous memory block
    explicit Storage(
        size_t num_elements,
        axon::core::DType dtype,
        axon::core::DeviceType device = axon::core::DeviceType::CPU
    ) noexcept
        : data_ptr_(nullptr),
          num_elements_(num_elements),
          dtype_(dtype),
          device_(device),
          owns_memory_(true) {
        if (AXON_LIKELY(num_elements_ > 0)) {
            const size_t element_size = axon::core::get_dtype_size(dtype_);
            if (AXON_UNLIKELY(element_size == 0 || num_elements_ > (SIZE_MAX / element_size))) {
                AXON_DEBUG_TRAP();
            }
            const size_t total_bytes = num_elements_ * element_size;
            data_ptr_ = axon::memory::allocate_aligned(total_bytes, AXON_SIMD_ALIGNMENT);
        }
    }

    // Zero-copy view constructor: wraps an external memory buffer without taking ownership
    Storage(
        void* external_data,
        size_t num_elements,
        axon::core::DType dtype,
        axon::core::DeviceType device = axon::core::DeviceType::CPU
    ) noexcept
        : data_ptr_(external_data),
          num_elements_(num_elements),
          dtype_(dtype),
          device_(device),
          owns_memory_(false) {
        if (AXON_UNLIKELY(num_elements_ > 0 && data_ptr_ == nullptr)) {
            AXON_DEBUG_TRAP();
        }
    }

    // Destructor: deterministically releases aligned memory if owned by this instance
    ~Storage() noexcept {
        if (owns_memory_ && data_ptr_ != nullptr) {
            axon::memory::free_aligned(data_ptr_);
            data_ptr_ = nullptr;
        }
    }

    // Strictly disable implicit deep copies
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    // Move constructor: transfers buffer ownership in O(1) with zero allocations
    Storage(Storage&& other) noexcept
        : data_ptr_(other.data_ptr_),
          num_elements_(other.num_elements_),
          dtype_(other.dtype_),
          device_(other.device_),
          owns_memory_(other.owns_memory_) {
        other.data_ptr_ = nullptr;
        other.num_elements_ = 0;
        other.owns_memory_ = false;
    }

    // Move assignment operator: releases current owned buffer and claims source buffer
    Storage& operator=(Storage&& other) noexcept {
        if (AXON_LIKELY(this != &other)) {
            if (owns_memory_ && data_ptr_ != nullptr) {
                axon::memory::free_aligned(data_ptr_);
            }

            data_ptr_ = other.data_ptr_;
            num_elements_ = other.num_elements_;
            dtype_ = other.dtype_;
            device_ = other.device_;
            owns_memory_ = other.owns_memory_;

            other.data_ptr_ = nullptr;
            other.num_elements_ = 0;
            other.owns_memory_ = false;
        }
        return *this;
    }

    // ========================================================================
    // ZERO-OVERHEAD INTROSPECTION & ACCESSORS
    // ========================================================================

    AXON_NODISCARD AXON_ALWAYS_INLINE void* data() noexcept {
        return data_ptr_;
    }

    AXON_NODISCARD AXON_ALWAYS_INLINE const void* data() const noexcept {
        return data_ptr_;
    }

    // Type-checked pointer accessor with compile-time DType verification
    template <typename T>
    AXON_NODISCARD AXON_ALWAYS_INLINE T* data_as() noexcept {
        constexpr auto expected_dtype = axon::core::dtype_of<T>();
        AXON_ASSERT(dtype_ == expected_dtype);
        return static_cast<T*>(data_ptr_);
    }

    template <typename T>
    AXON_NODISCARD AXON_ALWAYS_INLINE const T* data_as() const noexcept {
        constexpr auto expected_dtype = axon::core::dtype_of<T>();
        AXON_ASSERT(dtype_ == expected_dtype);
        return static_cast<const T*>(data_ptr_);
    }

    AXON_NODISCARD AXON_ALWAYS_INLINE size_t num_elements() const noexcept {
        return num_elements_;
    }

    AXON_NODISCARD AXON_ALWAYS_INLINE axon::core::DType dtype() const noexcept {
        return dtype_;
    }

    AXON_NODISCARD AXON_ALWAYS_INLINE axon::core::DeviceType device() const noexcept {
        return device_;
    }

    AXON_NODISCARD AXON_ALWAYS_INLINE bool owns_memory() const noexcept {
        return owns_memory_;
    }

    AXON_NODISCARD AXON_ALWAYS_INLINE size_t itemsize() const noexcept {
        return axon::core::get_dtype_size(dtype_);
    }

    AXON_NODISCARD AXON_ALWAYS_INLINE size_t nbytes() const noexcept {
        return num_elements_ * itemsize();
    }

    AXON_NODISCARD AXON_ALWAYS_INLINE bool empty() const noexcept {
        return num_elements_ == 0 || data_ptr_ == nullptr;
    }

    AXON_NODISCARD AXON_ALWAYS_INLINE bool is_simd_aligned() const noexcept {
        return axon::memory::is_aligned(data_ptr_, AXON_SIMD_ALIGNMENT);
    }

    // ========================================================================
    // LOW-LEVEL MEMORY MANIPULATION
    // ========================================================================

    // Zero-initializes the entire underlying memory buffer
    AXON_ALWAYS_INLINE void zero() noexcept {
        if (AXON_LIKELY(data_ptr_ != nullptr && num_elements_ > 0)) {
            if (is_simd_aligned()) {
                axon::memory::zero_memory_aligned(data_ptr_, nbytes());
            } else {
                axon::memory::zero_memory(data_ptr_, nbytes());
            }
        }
    }

    // Performs an explicit deep copy of this Storage into a newly allocated owning Storage
    AXON_NODISCARD AXON_ALWAYS_INLINE Storage clone() const noexcept {
        Storage copy_storage(num_elements_, dtype_, device_);
        if (AXON_LIKELY(data_ptr_ != nullptr && num_elements_ > 0)) {
            if (is_simd_aligned() && copy_storage.is_simd_aligned()) {
                axon::memory::copy_memory_aligned(copy_storage.data_ptr_, data_ptr_, nbytes());
            } else {
                axon::memory::copy_memory(copy_storage.data_ptr_, data_ptr_, nbytes());
            }
        }
        return copy_storage;
    }
};

} // namespace tensor
} // namespace axon

#endif // AXON_TENSOR_STORAGE_HPP
