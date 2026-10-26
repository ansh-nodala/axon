#ifndef AXON_NN_MODULE_HPP
#define AXON_NN_MODULE_HPP

#include "axon/core/axon_macros.hpp"
#include "axon/core/types.hpp"
#include "axon/tensor/tensor.hpp"
#include "axon/autograd/node.hpp"
#include <stddef.h>
#include <stdint.h>

namespace axon {
namespace nn {

// ============================================================================
// 1. CONSTANTS & TYPE ALIASES
// ============================================================================

using Variable = axon::autograd::Variable;

constexpr size_t AXON_MAX_MODULE_PARAMS = 64;
constexpr size_t AXON_MAX_SEQUENTIAL_LAYERS = 32;

// ============================================================================
// 2. BASE MODULE CLASS
// ============================================================================

class AXON_API Module {
public:
    Module() noexcept;
    virtual ~Module() noexcept;

    // Move-only semantics
    Module(Module&& other) noexcept;
    Module& operator=(Module&& other) noexcept;

    AXON_DISABLE_COPY(Module);

    // Parameter & submodule registration
    AXON_API void register_parameter(const Variable& param) noexcept;
    AXON_API void register_module(const Module& submodule) noexcept;

    // Parameter access
    [[nodiscard]] AXON_ALWAYS_INLINE size_t num_parameters() const noexcept {
        return num_params_;
    }

    [[nodiscard]] AXON_ALWAYS_INLINE Variable parameter(size_t idx) const noexcept {
        AXON_ASSERT(idx < num_params_);
        return params_[idx];
    }

    // Gradient management
    AXON_API void zero_grad() noexcept;

    // Pure virtual forward interface
    [[nodiscard]] virtual Variable forward(const Variable& input) noexcept = 0;

    // PyTorch-style callable syntax: layer(x)
    [[nodiscard]] AXON_ALWAYS_INLINE Variable operator()(const Variable& input) noexcept {
        return forward(input);
    }

protected:
    Variable params_[AXON_MAX_MODULE_PARAMS];
    size_t num_params_ = 0;
};

// ============================================================================
// 3. CORE NEURAL NETWORK LAYERS
// ============================================================================

class AXON_API Linear : public Module {
public:
    Linear(size_t in_features,
           size_t out_features,
           bool bias = true,
           axon::core::DType dtype = axon::core::DType::Float32,
           uint32_t seed = 0x9E3779B9u) noexcept;

    [[nodiscard]] Variable forward(const Variable& input) noexcept override;

    [[nodiscard]] AXON_ALWAYS_INLINE Variable weight() const noexcept {
        return weight_;
    }

    [[nodiscard]] AXON_ALWAYS_INLINE Variable bias() const noexcept {
        return bias_;
    }

    [[nodiscard]] AXON_ALWAYS_INLINE size_t in_features() const noexcept {
        return in_features_;
    }

    [[nodiscard]] AXON_ALWAYS_INLINE size_t out_features() const noexcept {
        return out_features_;
    }

    [[nodiscard]] AXON_ALWAYS_INLINE bool has_bias() const noexcept {
        return has_bias_;
    }

private:
    Variable weight_;
    Variable bias_;
    size_t in_features_;
    size_t out_features_;
    bool has_bias_;
};

class AXON_API RMSNorm : public Module {
public:
    explicit RMSNorm(size_t normalized_shape,
                     float eps = 1e-5f,
                     axon::core::DType dtype = axon::core::DType::Float32) noexcept;

    [[nodiscard]] Variable forward(const Variable& input) noexcept override;

    [[nodiscard]] AXON_ALWAYS_INLINE Variable weight() const noexcept {
        return weight_;
    }

    [[nodiscard]] AXON_ALWAYS_INLINE size_t normalized_shape() const noexcept {
        return normalized_shape_;
    }

    [[nodiscard]] AXON_ALWAYS_INLINE float eps() const noexcept {
        return eps_;
    }

private:
    Variable weight_;
    size_t normalized_shape_;
    float eps_;
};

class AXON_API ReLU : public Module {
public:
    ReLU() noexcept = default;
    [[nodiscard]] Variable forward(const Variable& input) noexcept override;
};

class AXON_API SiLU : public Module {
public:
    SiLU() noexcept = default;
    [[nodiscard]] Variable forward(const Variable& input) noexcept override;
};

class AXON_API GELU : public Module {
public:
    GELU() noexcept = default;
    [[nodiscard]] Variable forward(const Variable& input) noexcept override;
};

class AXON_API Sequential : public Module {
public:
    Sequential() noexcept = default;

    AXON_API void add(Module* layer) noexcept;

    [[nodiscard]] Variable forward(const Variable& input) noexcept override;

private:
    Module* layers_[AXON_MAX_SEQUENTIAL_LAYERS];
    size_t num_layers_ = 0;
};

// ============================================================================
// 4. RESEARCH-GRADE OPTIMIZERS
// ============================================================================

namespace optim {

using Variable = axon::autograd::Variable;

class AXON_API SGD {
public:
    SGD(float lr = 0.01f, float momentum = 0.0f, float weight_decay = 0.0f) noexcept;
    explicit SGD(const axon::nn::Module& module,
                 float lr = 0.01f,
                 float momentum = 0.0f,
                 float weight_decay = 0.0f) noexcept;

    AXON_API void add_parameters(const axon::nn::Module& module) noexcept;
    AXON_API void zero_grad() noexcept;
    AXON_API void step() noexcept;

    [[nodiscard]] AXON_ALWAYS_INLINE float learning_rate() const noexcept {
        return lr_;
    }

    AXON_ALWAYS_INLINE void set_learning_rate(float lr) noexcept {
        lr_ = lr;
    }

private:
    Variable params_[AXON_MAX_MODULE_PARAMS];
    axon::tensor::Tensor velocity_[AXON_MAX_MODULE_PARAMS];
    size_t num_params_ = 0;
    float lr_;
    float momentum_;
    float weight_decay_;
};

class AXON_API AdamW {
public:
    explicit AdamW(const axon::nn::Module& module,
                   float lr = 1e-3f,
                   float beta1 = 0.9f,
                   float beta2 = 0.999f,
                   float eps = 1e-8f,
                   float weight_decay = 0.01f) noexcept;

    AXON_API void add_parameters(const axon::nn::Module& module) noexcept;
    AXON_API void zero_grad() noexcept;
    AXON_API void step() noexcept;

    [[nodiscard]] AXON_ALWAYS_INLINE float learning_rate() const noexcept {
        return lr_;
    }

    AXON_ALWAYS_INLINE void set_learning_rate(float lr) noexcept {
        lr_ = lr;
    }

private:
    Variable params_[AXON_MAX_MODULE_PARAMS];
    axon::tensor::Tensor m_[AXON_MAX_MODULE_PARAMS];
    axon::tensor::Tensor v_[AXON_MAX_MODULE_PARAMS];
    size_t num_params_ = 0;
    size_t step_count_ = 0;
    float lr_;
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;
};

} // namespace optim

} // namespace nn
} // namespace axon

#endif // AXON_NN_MODULE_HPP

