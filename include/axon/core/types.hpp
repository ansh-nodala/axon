#ifndef AXON_CORE_TYPES_HPP
#define AXON_CORE_TYPES_HPP

#include "axon/core/axon_macros.hpp"
#include <stddef.h>
#include <stdint.h>

namespace axon {
namespace core {

// ============================================================================
// 1. HARDWARE & STORAGE DATA TYPES
// ============================================================================

// 16-bit IEEE 754 half-precision floating-point storage type
struct AXON_ALIGN(2) float16_t {
    uint16_t bits;

    constexpr float16_t() noexcept : bits(0) {}
    constexpr explicit float16_t(uint16_t raw_bits) noexcept : bits(raw_bits) {}
};

// 16-bit Brain Floating-Point (bfloat16) storage type
struct AXON_ALIGN(2) bfloat16_t {
    uint16_t bits;

    constexpr bfloat16_t() noexcept : bits(0) {}
    constexpr explicit bfloat16_t(uint16_t raw_bits) noexcept : bits(raw_bits) {}
};

// ============================================================================
// 2. TENSOR DATA TYPE & DEVICE ENUMERATIONS
// ============================================================================

// Fundamental Tensor Scalar Data Types
enum class DType : uint8_t {
    Float32  = 0,
    Float64  = 1,
    Float16  = 2,
    BFloat16 = 3,
    Int32    = 4,
    Int64    = 5,
    Int16    = 6,
    Int8     = 7,
    Uint8    = 8,
    Bool     = 9
};

// Compute Execution Target Device Types
enum class DeviceType : uint8_t {
    CPU = 0
};

// ============================================================================
// 3. DATA TYPE METADATA & INTROSPECTION QUERIES
// ============================================================================

// Returns the size in bytes of a single scalar element of the given DType
AXON_NODISCARD constexpr AXON_ALWAYS_INLINE size_t get_dtype_size(DType dtype) noexcept {
    switch (dtype) {
        case DType::Float32:  return 4;
        case DType::Float64:  return 8;
        case DType::Float16:  return 2;
        case DType::BFloat16: return 2;
        case DType::Int32:    return 4;
        case DType::Int64:    return 8;
        case DType::Int16:    return 2;
        case DType::Int8:     return 1;
        case DType::Uint8:    return 1;
        case DType::Bool:     return 1;
    }
    AXON_DEBUG_TRAP();
    return 0;
}

// Returns the canonical human-readable string identifier for the given DType
AXON_NODISCARD constexpr AXON_ALWAYS_INLINE const char* get_dtype_name(DType dtype) noexcept {
    switch (dtype) {
        case DType::Float32:  return "float32";
        case DType::Float64:  return "float64";
        case DType::Float16:  return "float16";
        case DType::BFloat16: return "bfloat16";
        case DType::Int32:    return "int32";
        case DType::Int64:    return "int64";
        case DType::Int16:    return "int16";
        case DType::Int8:     return "int8";
        case DType::Uint8:    return "uint8";
        case DType::Bool:     return "bool";
    }
    return "unknown";
}

// Returns the canonical string identifier for the given DeviceType
AXON_NODISCARD constexpr AXON_ALWAYS_INLINE const char* get_device_name(DeviceType device) noexcept {
    switch (device) {
        case DeviceType::CPU: return "cpu";
    }
    return "unknown";
}

// Evaluates to true if the given DType represents an IEEE/Brain floating-point format
AXON_NODISCARD constexpr AXON_ALWAYS_INLINE bool is_floating_point(DType dtype) noexcept {
    switch (dtype) {
        case DType::Float32:
        case DType::Float64:
        case DType::Float16:
        case DType::BFloat16:
            return true;
        case DType::Int32:
        case DType::Int64:
        case DType::Int16:
        case DType::Int8:
        case DType::Uint8:
        case DType::Bool:
            return false;
    }
    return false;
}

// Evaluates to true if the given DType represents a signed or unsigned integer format
AXON_NODISCARD constexpr AXON_ALWAYS_INLINE bool is_integral(DType dtype) noexcept {
    switch (dtype) {
        case DType::Int32:
        case DType::Int64:
        case DType::Int16:
        case DType::Int8:
        case DType::Uint8:
            return true;
        case DType::Float32:
        case DType::Float64:
        case DType::Float16:
        case DType::BFloat16:
        case DType::Bool:
            return false;
    }
    return false;
}

// Evaluates to true if the given DType supports negative values
AXON_NODISCARD constexpr AXON_ALWAYS_INLINE bool is_signed(DType dtype) noexcept {
    switch (dtype) {
        case DType::Float32:
        case DType::Float64:
        case DType::Float16:
        case DType::BFloat16:
        case DType::Int32:
        case DType::Int64:
        case DType::Int16:
        case DType::Int8:
            return true;
        case DType::Uint8:
        case DType::Bool:
            return false;
    }
    return false;
}

// ============================================================================
// 4. COMPILE-TIME C++ TYPE TO AXON DTYPE MAPPING
// ============================================================================

template <typename T>
struct TypeToDType {
    static constexpr bool is_valid = false;
};

template <>
struct TypeToDType<float> {
    static constexpr bool is_valid = true;
    static constexpr DType value = DType::Float32;
};

template <>
struct TypeToDType<double> {
    static constexpr bool is_valid = true;
    static constexpr DType value = DType::Float64;
};

template <>
struct TypeToDType<float16_t> {
    static constexpr bool is_valid = true;
    static constexpr DType value = DType::Float16;
};

template <>
struct TypeToDType<bfloat16_t> {
    static constexpr bool is_valid = true;
    static constexpr DType value = DType::BFloat16;
};

template <>
struct TypeToDType<int32_t> {
    static constexpr bool is_valid = true;
    static constexpr DType value = DType::Int32;
};

template <>
struct TypeToDType<int64_t> {
    static constexpr bool is_valid = true;
    static constexpr DType value = DType::Int64;
};

template <>
struct TypeToDType<int16_t> {
    static constexpr bool is_valid = true;
    static constexpr DType value = DType::Int16;
};

template <>
struct TypeToDType<int8_t> {
    static constexpr bool is_valid = true;
    static constexpr DType value = DType::Int8;
};

template <>
struct TypeToDType<uint8_t> {
    static constexpr bool is_valid = true;
    static constexpr DType value = DType::Uint8;
};

template <>
struct TypeToDType<bool> {
    static constexpr bool is_valid = true;
    static constexpr DType value = DType::Bool;
};

template <typename T>
AXON_NODISCARD constexpr DType dtype_of() noexcept {
    static_assert(TypeToDType<T>::is_valid, "Unsupported C++ scalar type for Axon DType mapping.");
    return TypeToDType<T>::value;
}

} // namespace core
} // namespace axon

#endif // AXON_CORE_TYPES_HPP
