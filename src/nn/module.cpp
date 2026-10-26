// ============================================================================
// axon/nn/module.cpp
//
// Implementation of concrete Module subclasses (Linear, RMSNorm) and the
// research-grade optimizers (SGD, AdamW) declared in <axon/nn/module.hpp>.
//
// Design notes:
//   - Zero dynamic allocation: optimizer parameter slots are stored as plain
//     fixed-size arrays of `Variable` (a cheap, ref-counted handle), never
//     heap-allocated via `new`. Copying a Variable retains the same
//     underlying autograd::Node, so the optimizer's copy shares gradients
//     with the live parameter inside the Module.
//   - No libm dependency: this is a freestanding-friendly translation unit.
//     Square roots are lowered to the native hardware instruction via the
//     compiler builtin `__builtin_sqrtf` (no libm call, no <math.h>).
//     Integer-exponent powers (bias-correction terms beta^t) are computed
//     with exact fast exponentiation-by-squaring instead of a generic
//     floating-point `powf`, which is both exact and branch-cheap.
// ============================================================================

#include "axon/nn/module.hpp"
#include "axon/autograd/node.hpp"
#include "axon/ops/elementwise.hpp"
#include "axon/ops/matmul.hpp"
#include "axon/ops/reductions.hpp"
#include "axon/core/memory.hpp"

namespace axon {
namespace nn {

// ============================================================================
// INTERNAL NUMERIC HELPERS (translation-unit-local, no libm)
// ============================================================================

namespace {

// Hardware square root with zero libm linkage: GCC/Clang lower this directly
// to SQRTSS (x86) / FSQRT (ARM64) regardless of -ffreestanding or -nostdlib.
AXON_NODISCARD AXON_ALWAYS_INLINE float hw_sqrt(float x) noexcept {
    return __builtin_sqrtf(x);
}

// Exact integer-exponent power via exponentiation-by-squaring: O(log2 exp)
// multiplies, no transcendental/libm call. Used for Adam bias-correction
// terms (beta1^t, beta2^t) where `t` is always a non-negative step count.
AXON_NODISCARD AXON_ALWAYS_INLINE float hw_powi(float base, size_t exp) noexcept {
    float result = 1.0f;
    float b = base;
    while (exp != 0) {
        if (exp & 1u) {
            result *= b;
        }
        b *= b;
        exp >>= 1u;
    }
    return result;
}

} // anonymous namespace

// ============================================================================
// MODULE BASE CLASS IMPLEMENTATION
//
// Declared in module.hpp but not defined anywhere else in src/nn/, so the
// move-only lifecycle and parameter-registration machinery lives here.
// ============================================================================

Module::Module() noexcept : num_params_(0) {
}

Module::~Module() noexcept {
    // Variable destructors handle node release individually; nothing else
    // for the base class to tear down.
}

Module::Module(Module&& other) noexcept : num_params_(other.num_params_) {
    for (size_t i = 0; i < num_params_; ++i) {
        params_[i] = static_cast<Variable&&>(other.params_[i]);
    }
    other.num_params_ = 0;
}

Module& Module::operator=(Module&& other) noexcept {
    if (AXON_LIKELY(this != &other)) {
        num_params_ = other.num_params_;
        for (size_t i = 0; i < num_params_; ++i) {
            params_[i] = static_cast<Variable&&>(other.params_[i]);
        }
        other.num_params_ = 0;
    }
    return *this;
}

void Module::register_parameter(const Variable& param) noexcept {
    AXON_ASSERT(num_params_ < AXON_MAX_MODULE_PARAMS);
    params_[num_params_++] = param;
}

void Module::register_module(const Module& submodule) noexcept {
    const size_t count = submodule.num_parameters();
    AXON_ASSERT(num_params_ + count <= AXON_MAX_MODULE_PARAMS);
    for (size_t i = 0; i < count; ++i) {
        register_parameter(submodule.parameter(i));
    }
}

void Module::zero_grad() noexcept {
    for (size_t i = 0; i < num_params_; ++i) {
        if (params_[i].valid()) {
            params_[i].zero_grad();
        }
    }
}

// ============================================================================
// LINEAR LAYER IMPLEMENTATION
// ============================================================================

Linear::Linear(size_t in_features, size_t out_features, bool bias,
               axon::core::DType dtype, uint32_t seed) noexcept
    : in_features_(in_features), out_features_(out_features), has_bias_(bias) {

    int64_t w_dims[2] = {static_cast<int64_t>(in_features), static_cast<int64_t>(out_features)};
    weight_ = Variable(axon::tensor::Tensor(w_dims, 2, dtype, axon::core::DeviceType::CPU), true);

    // Xavier/Glorot uniform initialization in [-bound, +bound].
    const float xavier_bound = hw_sqrt(6.0f / static_cast<float>(in_features + out_features));
    const float range = 2.0f * xavier_bound;

    float* AXON_RESTRICT w_data = weight_.data().data<float>();
    const size_t w_count = in_features * out_features;

    // Lightweight xorshift-style LCG: deterministic, branchless, no global
    // RNG state, cheap enough to inline at the call site.
    uint32_t rng_state = (seed == 0) ? 0x9E3779B9u : seed;
    for (size_t i = 0; i < w_count; ++i) {
        rng_state = rng_state * 1664525u + 1013904223u;
        const float u = static_cast<float>(rng_state & 0xFFFFFFu) / 16777216.0f;
        w_data[i] = (u - 0.5f) * range;
    }

    if (has_bias_) {
        int64_t b_dims[1] = {static_cast<int64_t>(out_features)};
        bias_ = Variable(axon::tensor::Tensor(b_dims, 1, dtype, axon::core::DeviceType::CPU), true);
        bias_.data().zero();
    } else {
        bias_ = Variable();
    }

    // Register parameters with the base Module so that generic consumers
    // (optimizers, checkpointing, introspection) can enumerate them via
    // Module::num_parameters() / Module::parameter(idx) without Linear
    // needing to expose any bespoke accessor surface.
    register_parameter(weight_);
    if (has_bias_) {
        register_parameter(bias_);
    }
}

axon::autograd::Variable Linear::forward(const axon::autograd::Variable& input) noexcept {
    return axon::autograd::linear(input, weight_, has_bias_ ? &bias_ : nullptr);
}

// ============================================================================
// RMSNORM LAYER IMPLEMENTATION
// ============================================================================

RMSNorm::RMSNorm(size_t normalized_shape, float eps, axon::core::DType dtype) noexcept
    : normalized_shape_(normalized_shape), eps_(eps) {

    int64_t norm_dims[1] = {static_cast<int64_t>(normalized_shape)};
    weight_ = Variable(axon::tensor::Tensor(norm_dims, 1, dtype, axon::core::DeviceType::CPU), true);

    float* AXON_RESTRICT w_data = weight_.data().data<float>();
    const size_t count = weight_.data().num_elements();
    for (size_t i = 0; i < count; ++i) {
        w_data[i] = 1.0f;
    }

    register_parameter(weight_);
}

axon::autograd::Variable RMSNorm::forward(const axon::autograd::Variable& input) noexcept {
    return axon::autograd::rmsnorm(input, weight_, eps_);
}

// ============================================================================
// OPTIMIZERS IMPLEMENTATION
// ============================================================================

namespace optim {

// ----------------------------------------------------------------------------
// SGD (with classical momentum)
// ----------------------------------------------------------------------------

SGD::SGD(float lr, float momentum, float weight_decay) noexcept
    : lr_(lr), momentum_(momentum), weight_decay_(weight_decay) {
}

SGD::SGD(const axon::nn::Module& module, float lr, float momentum, float weight_decay) noexcept
    : SGD(lr, momentum, weight_decay) {
    add_parameters(module);
}

void SGD::add_parameters(const axon::nn::Module& module) noexcept {
    const size_t count = module.num_parameters();
    AXON_ASSERT(num_params_ + count <= AXON_MAX_MODULE_PARAMS);

    for (size_t i = 0; i < count; ++i) {
        const size_t idx = num_params_++;

        // Shares the underlying autograd::Node with the Module's own
        // parameter (ref-counted copy) -- no allocation, gradients stay in sync.
        params_[idx] = module.parameter(i);

        const auto& shape = params_[idx].data().shape();
        velocity_[idx] = axon::tensor::Tensor(
            shape.dims(), shape.ndim(),
            params_[idx].data().dtype(), params_[idx].data().device());
        velocity_[idx].zero();
    }
}

void SGD::step() noexcept {
    for (size_t i = 0; i < num_params_; ++i) {
        Variable& p = params_[i];
        if (AXON_UNLIKELY(!p.valid())) {
            continue;
        }
        if (p.grad().empty()) {
            continue;
        }

        float* AXON_RESTRICT data = p.data().data<float>();
        const float* AXON_RESTRICT grad = p.grad().data<float>();
        float* AXON_RESTRICT vel = velocity_[i].data<float>();
        const size_t count = p.data().num_elements();

        AXON_PREFETCH_READ(grad);
        AXON_PREFETCH_WRITE(data);

        for (size_t j = 0; j < count; ++j) {
            float g = grad[j];
            if (weight_decay_ != 0.0f) {
                g += weight_decay_ * data[j];
            }
            if (momentum_ != 0.0f) {
                vel[j] = momentum_ * vel[j] + g;
                g = vel[j];
            }
            data[j] -= lr_ * g;
        }
    }
}

void SGD::zero_grad() noexcept {
    for (size_t i = 0; i < num_params_; ++i) {
        if (params_[i].valid()) {
            params_[i].zero_grad();
        }
    }
}

// ----------------------------------------------------------------------------
// AdamW (decoupled weight decay)
// ----------------------------------------------------------------------------

AdamW::AdamW(const axon::nn::Module& module, float lr, float beta1, float beta2,
             float eps, float weight_decay) noexcept
    : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay) {
    add_parameters(module);
}

void AdamW::add_parameters(const axon::nn::Module& module) noexcept {
    const size_t count = module.num_parameters();
    AXON_ASSERT(num_params_ + count <= AXON_MAX_MODULE_PARAMS);

    for (size_t i = 0; i < count; ++i) {
        const size_t idx = num_params_++;

        params_[idx] = module.parameter(i);

        const auto& shape = params_[idx].data().shape();
        m_[idx] = axon::tensor::Tensor(
            shape.dims(), shape.ndim(),
            params_[idx].data().dtype(), params_[idx].data().device());
        v_[idx] = axon::tensor::Tensor(
            shape.dims(), shape.ndim(),
            params_[idx].data().dtype(), params_[idx].data().device());
        m_[idx].zero();
        v_[idx].zero();
    }
}

void AdamW::step() noexcept {
    ++step_count_;
    const size_t t = step_count_;

    // Bias-correction terms are identical for every parameter at this step,
    // so they are hoisted out of the per-parameter loop entirely.
    const float bias_correction1 = 1.0f - hw_powi(beta1_, t);
    const float bias_correction2 = 1.0f - hw_powi(beta2_, t);

    for (size_t i = 0; i < num_params_; ++i) {
        Variable& p = params_[i];
        if (AXON_UNLIKELY(!p.valid())) {
            continue;
        }
        if (p.grad().empty()) {
            continue;
        }

        float* AXON_RESTRICT data = p.data().data<float>();
        const float* AXON_RESTRICT grad = p.grad().data<float>();
        float* AXON_RESTRICT m_data = m_[i].data<float>();
        float* AXON_RESTRICT v_data = v_[i].data<float>();
        const size_t count = p.data().num_elements();

        AXON_PREFETCH_READ(grad);
        AXON_PREFETCH_WRITE(data);

        for (size_t j = 0; j < count; ++j) {
            const float g = grad[j];

            // Decoupled weight decay (applied directly to the weights,
            // not folded into the gradient, per the AdamW formulation).
            if (weight_decay_ != 0.0f) {
                data[j] -= lr_ * weight_decay_ * data[j];
            }

            m_data[j] = beta1_ * m_data[j] + (1.0f - beta1_) * g;
            v_data[j] = beta2_ * v_data[j] + (1.0f - beta2_) * g * g;

            const float m_hat = m_data[j] / bias_correction1;
            const float v_hat = v_data[j] / bias_correction2;

            data[j] -= lr_ * m_hat / (hw_sqrt(v_hat) + eps_);
        }
    }
}

void AdamW::zero_grad() noexcept {
    for (size_t i = 0; i < num_params_; ++i) {
        if (params_[i].valid()) {
            params_[i].zero_grad();
        }
    }
}

} // namespace optim
} // namespace nn
} // namespace axon
