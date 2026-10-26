#ifndef AXON_AUTOGRAD_NODE_HPP
#define AXON_AUTOGRAD_NODE_HPP

#include "axon/core/axon_macros.hpp"
#include "axon/core/memory.hpp"
#include "axon/core/types.hpp"
#include "axon/tensor/tensor.hpp"
#include <stddef.h>
#include <stdint.h>

namespace axon {
namespace autograd {

// ============================================================================
// 1. GRAPH TOPOLOGY CONSTANTS & FORWARD DECLARATIONS
// ============================================================================

constexpr size_t AXON_MAX_GRAD_INPUTS = 4;
constexpr size_t AXON_MAX_GRAPH_NODES = 4096;

class Node;
struct NodeHandle;

// ============================================================================
// 2. EXTENSIBLE BACKWARD OPERATION INTERFACE
// ============================================================================

enum class OpType : uint8_t {
    None = 0,
    Add,
    Sub,
    Mul,
    Div,
    Scale,
    ReLU,
    SiLU,
    GELU,
    Sigmoid,
    Tanh,
    Exp,
    MatMul,
    Linear,
    SumAll,
    MeanAll,
    Softmax,
    RMSNorm,
    Custom
};

using CustomBackwardFn = void (*)(Node* self) noexcept;

// ============================================================================
// 3. INTRUSIVE REFERENCE-COUNTED GRAPH NODE
// ============================================================================

class AXON_API Node {
public:
    // ------------------------------------------------------------------------
    // Custom Allocation (64-byte SIMD Aligned)
    // ------------------------------------------------------------------------
    [[nodiscard]] static AXON_ALWAYS_INLINE void* operator new(size_t size) noexcept {
        AXON_ASSERT(size == sizeof(Node));
        return axon::memory::allocate_aligned(size, AXON_SIMD_ALIGNMENT);
    }

    static AXON_ALWAYS_INLINE void operator delete(void* ptr) noexcept {
        axon::memory::free_aligned(ptr);
    }

    // ------------------------------------------------------------------------
    // Construction & Initialization
    // ------------------------------------------------------------------------
    Node() noexcept
        : data_(),
          grad_(),
          inputs_{nullptr, nullptr, nullptr, nullptr},
          num_inputs_(0),
          ref_count_(1),
          op_type_(OpType::None),
          scalar_param_(0.0f),
          custom_backward_fn_(nullptr),
          requires_grad_(false),
          is_leaf_(true) {}

    // Initialize as leaf tensor (moves rvalue tensor data)
    explicit Node(axon::tensor::Tensor&& data, bool requires_grad = false) noexcept
        : data_(static_cast<axon::tensor::Tensor&&>(data)),
          grad_(),
          inputs_{nullptr, nullptr, nullptr, nullptr},
          num_inputs_(0),
          ref_count_(1),
          op_type_(OpType::None),
          scalar_param_(0.0f),
          custom_backward_fn_(nullptr),
          requires_grad_(requires_grad),
          is_leaf_(true) {}

    // Initialize as leaf tensor from non-const lvalue (moves data)
    explicit Node(axon::tensor::Tensor& data, bool requires_grad = false) noexcept
        : data_(static_cast<axon::tensor::Tensor&&>(data)),
          grad_(),
          inputs_{nullptr, nullptr, nullptr, nullptr},
          num_inputs_(0),
          ref_count_(1),
          op_type_(OpType::None),
          scalar_param_(0.0f),
          custom_backward_fn_(nullptr),
          requires_grad_(requires_grad),
          is_leaf_(true) {}

    // Initialize as leaf tensor from const lvalue (deep copies via clone)
    explicit Node(const axon::tensor::Tensor& data, bool requires_grad = false) noexcept
        : data_(data.clone()),
          grad_(),
          inputs_{nullptr, nullptr, nullptr, nullptr},
          num_inputs_(0),
          ref_count_(1),
          op_type_(OpType::None),
          scalar_param_(0.0f),
          custom_backward_fn_(nullptr),
          requires_grad_(requires_grad),
          is_leaf_(true) {}

    // Initialize as computation node (moves rvalue tensor data)
    Node(axon::tensor::Tensor&& data,
         OpType op,
         Node* const* inputs,
         size_t num_inputs,
         float scalar_param = 0.0f,
         CustomBackwardFn custom_fn = nullptr) noexcept
        : data_(static_cast<axon::tensor::Tensor&&>(data)),
          grad_(),
          inputs_{nullptr, nullptr, nullptr, nullptr},
          num_inputs_(0),
          ref_count_(1),
          op_type_(op),
          scalar_param_(scalar_param),
          custom_backward_fn_(custom_fn),
          requires_grad_(false),
          is_leaf_(false) {
        AXON_ASSERT(num_inputs <= AXON_MAX_GRAD_INPUTS);
        for (size_t i = 0; i < num_inputs; ++i) {
            inputs_[i] = inputs[i];
            if (inputs_[i]) {
                inputs_[i]->retain();
            }
        }
        num_inputs_ = num_inputs;
        for (size_t i = 0; i < num_inputs; ++i) {
            if (inputs_[i] && inputs_[i]->requires_grad_) {
                requires_grad_ = true;
                break;
            }
        }
    }

    // Initialize as computation node from non-const lvalue (moves data)
    Node(axon::tensor::Tensor& data,
         OpType op,
         Node* const* inputs,
         size_t num_inputs,
         float scalar_param = 0.0f,
         CustomBackwardFn custom_fn = nullptr) noexcept
        : data_(static_cast<axon::tensor::Tensor&&>(data)),
          grad_(),
          inputs_{nullptr, nullptr, nullptr, nullptr},
          num_inputs_(0),
          ref_count_(1),
          op_type_(op),
          scalar_param_(scalar_param),
          custom_backward_fn_(custom_fn),
          requires_grad_(false),
          is_leaf_(false) {
        AXON_ASSERT(num_inputs <= AXON_MAX_GRAD_INPUTS);
        for (size_t i = 0; i < num_inputs; ++i) {
            inputs_[i] = inputs[i];
            if (inputs_[i]) {
                inputs_[i]->retain();
            }
        }
        num_inputs_ = num_inputs;
        for (size_t i = 0; i < num_inputs; ++i) {
            if (inputs_[i] && inputs_[i]->requires_grad_) {
                requires_grad_ = true;
                break;
            }
        }
    }

    // Initialize as computation node from const lvalue (deep copies via clone)
    Node(const axon::tensor::Tensor& data,
         OpType op,
         Node* const* inputs,
         size_t num_inputs,
         float scalar_param = 0.0f,
         CustomBackwardFn custom_fn = nullptr) noexcept
        : data_(data.clone()),
          grad_(),
          inputs_{nullptr, nullptr, nullptr, nullptr},
          num_inputs_(0),
          ref_count_(1),
          op_type_(op),
          scalar_param_(scalar_param),
          custom_backward_fn_(custom_fn),
          requires_grad_(false),
          is_leaf_(false) {
        AXON_ASSERT(num_inputs <= AXON_MAX_GRAD_INPUTS);
        for (size_t i = 0; i < num_inputs; ++i) {
            inputs_[i] = inputs[i];
            if (inputs_[i]) {
                inputs_[i]->retain();
            }
        }
        num_inputs_ = num_inputs;
        for (size_t i = 0; i < num_inputs; ++i) {
            if (inputs_[i] && inputs_[i]->requires_grad_) {
                requires_grad_ = true;
                break;
            }
        }
    }

    // ------------------------------------------------------------------------
    // Intrusive Reference Counting
    // ------------------------------------------------------------------------
    AXON_ALWAYS_INLINE void retain() noexcept {
        ++ref_count_;
    }

    AXON_API void release() noexcept;

    // ------------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------------
    [[nodiscard]] AXON_ALWAYS_INLINE axon::tensor::Tensor& data() noexcept { return data_; }
    [[nodiscard]] AXON_ALWAYS_INLINE const axon::tensor::Tensor& data() const noexcept { return data_; }

    [[nodiscard]] AXON_ALWAYS_INLINE axon::tensor::Tensor& grad() noexcept { return grad_; }
    [[nodiscard]] AXON_ALWAYS_INLINE const axon::tensor::Tensor& grad() const noexcept { return grad_; }

    [[nodiscard]] AXON_ALWAYS_INLINE bool requires_grad() const noexcept { return requires_grad_; }
    [[nodiscard]] AXON_ALWAYS_INLINE bool is_leaf() const noexcept { return is_leaf_; }
    [[nodiscard]] AXON_ALWAYS_INLINE OpType op_type() const noexcept { return op_type_; }
    [[nodiscard]] AXON_ALWAYS_INLINE size_t num_inputs() const noexcept { return num_inputs_; }
    [[nodiscard]] AXON_ALWAYS_INLINE Node* input(size_t idx) const noexcept {
        AXON_ASSERT(idx < num_inputs_);
        return inputs_[idx];
    }
    [[nodiscard]] AXON_ALWAYS_INLINE float scalar_param() const noexcept { return scalar_param_; }
    [[nodiscard]] AXON_ALWAYS_INLINE CustomBackwardFn custom_backward_fn() const noexcept { return custom_backward_fn_; }
    [[nodiscard]] AXON_ALWAYS_INLINE uint32_t ref_count() const noexcept { return ref_count_; }

    // ------------------------------------------------------------------------
    // Gradient Management
    // ------------------------------------------------------------------------
    AXON_API void ensure_grad() noexcept;
    AXON_API void zero_grad() noexcept;
    AXON_API void accumulate_grad(const axon::tensor::Tensor& incoming_grad) noexcept;

    // ------------------------------------------------------------------------
    // Reverse-Mode Automatic Differentiation Engine
    // ------------------------------------------------------------------------
    AXON_API void backward() noexcept;

private:
    axon::tensor::Tensor data_;
    axon::tensor::Tensor grad_;
    Node* inputs_[AXON_MAX_GRAD_INPUTS];
    size_t num_inputs_;
    uint32_t ref_count_;
    OpType op_type_;
    float scalar_param_;
    CustomBackwardFn custom_backward_fn_;
    bool requires_grad_;
    bool is_leaf_;

    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;
    Node(Node&&) = delete;
    Node& operator=(Node&&) = delete;
};

// ============================================================================
// 4. RAII SMART HANDLE FOR RESEARCHERS (PYTORCH-LIKE VARIABLE)
// ============================================================================

class AXON_API Variable {
public:
    constexpr Variable() noexcept : node_(nullptr) {}

    explicit Variable(Node* node, bool add_ref = false) noexcept : node_(node) {
        if (node_ && add_ref) {
            node_->retain();
        }
    }

    explicit Variable(axon::tensor::Tensor&& data, bool requires_grad = false) noexcept
        : node_(new Node(static_cast<axon::tensor::Tensor&&>(data), requires_grad)) {}

    explicit Variable(axon::tensor::Tensor& data, bool requires_grad = false) noexcept
        : node_(new Node(static_cast<axon::tensor::Tensor&&>(data), requires_grad)) {}

    explicit Variable(const axon::tensor::Tensor& data, bool requires_grad = false) noexcept
        : node_(new Node(data.clone(), requires_grad)) {}

    Variable(const Variable& other) noexcept : node_(other.node_) {
        if (node_) {
            node_->retain();
        }
    }

    Variable(Variable&& other) noexcept : node_(other.node_) {
        other.node_ = nullptr;
    }

    ~Variable() noexcept {
        if (node_) {
            node_->release();
        }
    }

    Variable& operator=(const Variable& other) noexcept {
        if (AXON_LIKELY(this != &other)) {
            if (node_) {
                node_->release();
            }
            node_ = other.node_;
            if (node_) {
                node_->retain();
            }
        }
        return *this;
    }

    Variable& operator=(Variable&& other) noexcept {
        if (AXON_LIKELY(this != &other)) {
            if (node_) {
                node_->release();
            }
            node_ = other.node_;
            other.node_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] AXON_ALWAYS_INLINE axon::tensor::Tensor& data() noexcept {
        AXON_ASSERT(node_ != nullptr);
        return node_->data();
    }

    [[nodiscard]] AXON_ALWAYS_INLINE const axon::tensor::Tensor& data() const noexcept {
        AXON_ASSERT(node_ != nullptr);
        return node_->data();
    }

    [[nodiscard]] AXON_ALWAYS_INLINE axon::tensor::Tensor& grad() noexcept {
        AXON_ASSERT(node_ != nullptr);
        return node_->grad();
    }

    [[nodiscard]] AXON_ALWAYS_INLINE const axon::tensor::Tensor& grad() const noexcept {
        AXON_ASSERT(node_ != nullptr);
        return node_->grad();
    }

    [[nodiscard]] AXON_ALWAYS_INLINE bool requires_grad() const noexcept {
        return node_ ? node_->requires_grad() : false;
    }

    [[nodiscard]] AXON_ALWAYS_INLINE bool is_leaf() const noexcept {
        return node_ ? node_->is_leaf() : false;
    }

    [[nodiscard]] AXON_ALWAYS_INLINE bool valid() const noexcept {
        return node_ != nullptr;
    }

    [[nodiscard]] AXON_ALWAYS_INLINE Node* node() const noexcept {
        return node_;
    }

    AXON_ALWAYS_INLINE void zero_grad() noexcept {
        if (node_) {
            node_->zero_grad();
        }
    }

    AXON_ALWAYS_INLINE void backward() noexcept {
        if (node_) {
            node_->backward();
        }
    }

private:
    Node* node_;
};

// ============================================================================
// 5. DIFFERENTIABLE FUNCTIONAL OPERATIONS (namespace axon::autograd)
// ============================================================================

[[nodiscard]] AXON_API Variable add(const Variable& a, const Variable& b) noexcept;
[[nodiscard]] AXON_API Variable sub(const Variable& a, const Variable& b) noexcept;
[[nodiscard]] AXON_API Variable mul(const Variable& a, const Variable& b) noexcept;

[[nodiscard]] AXON_API Variable scale(const Variable& in, float scalar) noexcept;
[[nodiscard]] AXON_API Variable relu(const Variable& in) noexcept;
[[nodiscard]] AXON_API Variable silu(const Variable& in) noexcept;
[[nodiscard]] AXON_API Variable gelu(const Variable& in) noexcept;
[[nodiscard]] AXON_API Variable sigmoid(const Variable& in) noexcept;
[[nodiscard]] AXON_API Variable tanh_op(const Variable& in) noexcept;

[[nodiscard]] AXON_API Variable matmul(const Variable& a, const Variable& b) noexcept;
[[nodiscard]] AXON_API Variable linear(const Variable& input, const Variable& weight, const Variable* bias = nullptr) noexcept;

[[nodiscard]] AXON_API Variable rmsnorm(const Variable& in, const Variable& weight, float eps = 1e-5f) noexcept;

[[nodiscard]] AXON_API Variable sum_all(const Variable& in) noexcept;
[[nodiscard]] AXON_API Variable mean_all(const Variable& in) noexcept;

[[nodiscard]] AXON_API Variable mse_loss(const Variable& pred, const Variable& target) noexcept;

[[nodiscard]] AXON_API Variable custom_op(
    axon::tensor::Tensor&& forward_out,
    Node* const* inputs,
    size_t num_inputs,
    CustomBackwardFn backward_fn,
    float scalar_param = 0.0f
) noexcept;

} // namespace autograd
} // namespace axon

#endif // AXON_AUTOGRAD_NODE_HPP
