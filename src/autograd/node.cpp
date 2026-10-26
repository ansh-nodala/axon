#include "axon/autograd/node.hpp"
#include "axon/ops/elementwise.hpp"
#include "axon/ops/matmul.hpp"
#include "axon/ops/reductions.hpp"
#include <math.h>

namespace axon {
namespace autograd {

// ============================================================================
// 1. NODE LIFECYCLE & GRADIENT MANAGEMENT
// ============================================================================

void Node::release() noexcept {
    --ref_count_;
    if (ref_count_ == 0) {
        for (size_t i = 0; i < num_inputs_; ++i) {
            if (inputs_[i] != nullptr) {
                inputs_[i]->release();
            }
        }
        delete this;
    }
}

void Node::ensure_grad() noexcept {
    if (grad_.empty()) {
        grad_ = axon::tensor::Tensor(data_.shape().dims(), data_.ndim(), data_.dtype(), data_.device());
        grad_.zero();
    }
}

void Node::zero_grad() noexcept {
    if (!grad_.empty()) {
        grad_.zero();
    }
}

void Node::accumulate_grad(const axon::tensor::Tensor& incoming_grad) noexcept {
    ensure_grad();
    axon::ops::add_(grad_, incoming_grad);
}

// ============================================================================
// 2. REVERSE-MODE AUTOMATIC DIFFERENTIATION ENGINE
// ============================================================================

namespace {

// Fixed-capacity topological sort containers (zero stdlib)
struct TopoState {
    Node* nodes[AXON_MAX_GRAPH_NODES];
    size_t count;
    bool visited[AXON_MAX_GRAPH_NODES];
    Node* node_map[AXON_MAX_GRAPH_NODES];
    size_t map_count;
};

// O(n) lookup for visited set using fixed array
bool is_visited(TopoState& ts, Node* n) noexcept {
    for (size_t i = 0; i < ts.map_count; ++i) {
        if (ts.node_map[i] == n) {
            return ts.visited[i];
        }
    }
    return false;
}

void mark_visited(TopoState& ts, Node* n) noexcept {
    if (ts.map_count < AXON_MAX_GRAPH_NODES) {
        ts.node_map[ts.map_count] = n;
        ts.visited[ts.map_count] = true;
        ++ts.map_count;
    }
}

void build_topo(TopoState& ts, Node* n) noexcept {
    if (n == nullptr || !n->requires_grad() || is_visited(ts, n)) {
        return;
    }
    mark_visited(ts, n);
    for (size_t i = 0; i < n->num_inputs(); ++i) {
        build_topo(ts, n->input(i));
    }
    if (ts.count < AXON_MAX_GRAPH_NODES) {
        ts.nodes[ts.count] = n;
        ++ts.count;
    }
}

} // anonymous namespace

void Node::backward() noexcept {
    TopoState ts{};
    ts.count = 0;
    ts.map_count = 0;

    build_topo(ts, this);

    // Seed gradient: this->grad_ = 1.0
    ensure_grad();
    axon::ops::fill_(grad_, 1.0f);

    // Iterate in reverse (post-order from root down to leaves)
    for (size_t i = ts.count; i > 0; --i) {
        Node* node = ts.nodes[i - 1];
        if (!node->requires_grad()) {
            continue;
        }

        node->ensure_grad();

        switch (node->op_type_) {
            case OpType::Add: {
                if (node->input(0) && node->input(0)->requires_grad()) {
                    node->input(0)->accumulate_grad(node->grad_);
                }
                if (node->input(1) && node->input(1)->requires_grad()) {
                    node->input(1)->accumulate_grad(node->grad_);
                }
                break;
            }
            case OpType::Sub: {
                if (node->input(0) && node->input(0)->requires_grad()) {
                    node->input(0)->accumulate_grad(node->grad_);
                }
                if (node->input(1) && node->input(1)->requires_grad()) {
                    axon::tensor::Tensor scaled_neg = axon::ops::scale(node->grad_, -1.0f);
                    node->input(1)->accumulate_grad(scaled_neg);
                }
                break;
            }
            case OpType::Mul: {
                if (node->input(0) && node->input(0)->requires_grad()) {
                    axon::tensor::Tensor da = axon::ops::mul(node->grad_, node->input(1)->data_);
                    node->input(0)->accumulate_grad(da);
                }
                if (node->input(1) && node->input(1)->requires_grad()) {
                    axon::tensor::Tensor db = axon::ops::mul(node->grad_, node->input(0)->data_);
                    node->input(1)->accumulate_grad(db);
                }
                break;
            }
            case OpType::Scale: {
                if (node->input(0) && node->input(0)->requires_grad()) {
                    axon::tensor::Tensor dg = axon::ops::scale(node->grad_, node->scalar_param_);
                    node->input(0)->accumulate_grad(dg);
                }
                break;
            }
            case OpType::ReLU: {
                if (node->input(0) && node->input(0)->requires_grad()) {
                    const axon::tensor::Tensor& x = node->input(0)->data_;
                    const axon::tensor::Tensor& dy = node->grad_;
                    axon::tensor::Tensor dx(x.shape().dims(), x.ndim(), x.dtype(), x.device());
                    dx.zero();
                    size_t count = x.num_elements();
                    float* dx_ptr = dx.data<float>();
                    const float* x_ptr = x.data<float>();
                    const float* dy_ptr = dy.data<float>();
                    for (size_t k = 0; k < count; ++k) {
                        dx_ptr[k] = (x_ptr[k] > 0.0f) ? dy_ptr[k] : 0.0f;
                    }
                    node->input(0)->accumulate_grad(dx);
                }
                break;
            }
            case OpType::Sigmoid: {
                if (node->input(0) && node->input(0)->requires_grad()) {
                    const axon::tensor::Tensor& y = node->data_;  // forward output
                    const axon::tensor::Tensor& dy = node->grad_;
                    axon::tensor::Tensor dx(y.shape().dims(), y.ndim(), y.dtype(), y.device());
                    dx.zero();
                    size_t count = y.num_elements();
                    float* dx_ptr = dx.data<float>();
                    const float* y_ptr = y.data<float>();
                    const float* dy_ptr = dy.data<float>();
                    for (size_t k = 0; k < count; ++k) {
                        dx_ptr[k] = dy_ptr[k] * y_ptr[k] * (1.0f - y_ptr[k]);
                    }
                    node->input(0)->accumulate_grad(dx);
                }
                break;
            }
            case OpType::Tanh: {
                if (node->input(0) && node->input(0)->requires_grad()) {
                    const axon::tensor::Tensor& y = node->data_;
                    const axon::tensor::Tensor& dy = node->grad_;
                    axon::tensor::Tensor dx(y.shape().dims(), y.ndim(), y.dtype(), y.device());
                    dx.zero();
                    size_t count = y.num_elements();
                    float* dx_ptr = dx.data<float>();
                    const float* y_ptr = y.data<float>();
                    const float* dy_ptr = dy.data<float>();
                    for (size_t k = 0; k < count; ++k) {
                        dx_ptr[k] = dy_ptr[k] * (1.0f - y_ptr[k] * y_ptr[k]);
                    }
                    node->input(0)->accumulate_grad(dx);
                }
                break;
            }
            case OpType::SiLU: {
                if (node->input(0) && node->input(0)->requires_grad()) {
                    const axon::tensor::Tensor& x = node->input(0)->data_;
                    const axon::tensor::Tensor& dy = node->grad_;
                    axon::tensor::Tensor dx(x.shape().dims(), x.ndim(), x.dtype(), x.device());
                    dx.zero();
                    size_t count = x.num_elements();
                    float* dx_ptr = dx.data<float>();
                    const float* x_ptr = x.data<float>();
                    const float* dy_ptr = dy.data<float>();
                    for (size_t k = 0; k < count; ++k) {
                        float xi = x_ptr[k];
                        float sig = 1.0f / (1.0f + expf(-xi));
                        dx_ptr[k] = dy_ptr[k] * sig * (1.0f + xi * (1.0f - sig));
                    }
                    node->input(0)->accumulate_grad(dx);
                }
                break;
            }
            case OpType::GELU: {
                if (node->input(0) && node->input(0)->requires_grad()) {
                    const axon::tensor::Tensor& x = node->input(0)->data_;
                    const axon::tensor::Tensor& dy = node->grad_;
                    axon::tensor::Tensor dx(x.shape().dims(), x.ndim(), x.dtype(), x.device());
                    dx.zero();
                    size_t count = x.num_elements();
                    float* dx_ptr = dx.data<float>();
                    const float* x_ptr = x.data<float>();
                    const float* dy_ptr = dy.data<float>();
                    for (size_t k = 0; k < count; ++k) {
                        float xi = x_ptr[k];
                        float inner = 0.044715f * xi * xi * xi;
                        float sqrt_2pi = 0.7978845608028654f;
                        float t = sqrt_2pi * (xi + inner);
                        float tanh_t = tanf(t);
                        float sech_t_sq = 1.0f - tanh_t * tanh_t;
                        float coef = sqrt_2pi * (1.0f + 3.0f * 0.044715f * xi * xi);
                        dx_ptr[k] = dy_ptr[k] * (0.5f * (1.0f + tanh_t) + 0.5f * xi * sech_t_sq * coef);
                    }
                    node->input(0)->accumulate_grad(dx);
                }
                break;
            }
            case OpType::MatMul: {
                if (node->input(0) && node->input(0)->requires_grad()) {
                    const axon::tensor::Tensor& b = node->input(1)->data_;
                    const axon::tensor::Tensor& dy = node->grad_;
                    if (b.is_contiguous()) {
                        axon::tensor::Tensor b_t = b.transpose(0, 1);
                        axon::tensor::Tensor da = axon::ops::matmul(dy, b_t);
                        node->input(0)->accumulate_grad(da);
                    }
                }
                if (node->input(1) && node->input(1)->requires_grad()) {
                    const axon::tensor::Tensor& a = node->input(0)->data_;
                    const axon::tensor::Tensor& dy = node->grad_;
                    if (a.is_contiguous()) {
                        axon::tensor::Tensor at = a.transpose(0, 1);
                        axon::tensor::Tensor db = axon::ops::matmul(at, dy);
                        node->input(1)->accumulate_grad(db);
                    }
                }
                break;
            }
            case OpType::Linear: {
                if (node->input(0) && node->input(0)->requires_grad()) {
                    const axon::tensor::Tensor& weight = node->input(1)->data_;
                    const axon::tensor::Tensor& dy = node->grad_;
                    axon::tensor::Tensor weight_t = weight.transpose(0, 1);
                    axon::tensor::Tensor dinput = axon::ops::matmul(dy, weight_t);
                    node->input(0)->accumulate_grad(dinput);
                }
                if (node->input(1) && node->input(1)->requires_grad()) {
                    const axon::tensor::Tensor& input = node->input(0)->data_;
                    const axon::tensor::Tensor& dy = node->grad_;
                    if (input.is_contiguous()) {
                        size_t M = input.dim(0);
                        size_t K = input.dim(1); // in_features
                        size_t N = dy.dim(1);    // out_features

                        int64_t dw_dims[2] = {static_cast<int64_t>(K), static_cast<int64_t>(N)};
                        axon::tensor::Tensor dweight(dw_dims, 2, dy.dtype(), dy.device());
                        dweight.zero();

                        const float* in_ptr = input.data<float>();
                        const float* dy_ptr = dy.data<float>();
                        float* dw_ptr = dweight.data<float>();

                        for (size_t k = 0; k < K; ++k) {
                            for (size_t n = 0; n < N; ++n) {
                                float sum = 0.0f;
                                for (size_t m = 0; m < M; ++m) {
                                    sum += in_ptr[m * K + k] * dy_ptr[m * N + n];
                                }
                                dw_ptr[k * N + n] = sum;
                            }
                        }
                        node->input(1)->accumulate_grad(dweight);
                    }
                }
                if (node->num_inputs() >= 3 && node->input(2) && node->input(2)->requires_grad()) {
                    const axon::tensor::Tensor& dy = node->grad_;
                    size_t M = dy.dim(0);
                    size_t N = dy.dim(1);
                    int64_t shape_1d[1] = {static_cast<int64_t>(N)};
                    axon::tensor::Tensor dbias(shape_1d, 1, dy.dtype(), dy.device());
                    dbias.zero();
                    const float* dy_ptr = dy.data<float>();
                    float* bias_ptr = dbias.data<float>();
                    for (size_t row = 0; row < M; ++row) {
                        for (size_t col = 0; col < N; ++col) {
                            bias_ptr[col] += dy_ptr[row * N + col];
                        }
                    }
                    node->input(2)->accumulate_grad(dbias);
                }
                break;
            }
            case OpType::RMSNorm: {
                if (node->input(0) && node->input(0)->requires_grad()) {
                    const axon::tensor::Tensor& in = node->input(0)->data_;
                    const axon::tensor::Tensor& weight = node->input(1)->data_;
                    const axon::tensor::Tensor& dy = node->grad_;
                    float eps = node->scalar_param_;
                    size_t B = in.dim(0);
                    size_t D = in.dim(1);
                    axon::tensor::Tensor dx(in.shape().dims(), in.ndim(), in.dtype(), in.device());
                    dx.zero();
                    const float* in_ptr = in.data<float>();
                    const float* w_ptr = weight.data<float>();
                    const float* dy_ptr = dy.data<float>();
                    float* dx_ptr = dx.data<float>();
                    for (size_t b = 0; b < B; ++b) {
                        float sum_sq = 0.0f;
                        for (size_t d = 0; d < D; ++d) {
                            float val = in_ptr[b * D + d];
                            sum_sq += val * val;
                        }
                        float mean_sq = sum_sq / static_cast<float>(D);
                        float denom = sqrtf(mean_sq + eps);
                        float inv_denom = 1.0f / denom;
                        float norm_const = 1.0f / (denom * denom * denom) / static_cast<float>(D);
                        float sum_dy_w = 0.0f;
                        for (size_t d = 0; d < D; ++d) {
                            sum_dy_w += dy_ptr[b * D + d] * w_ptr[d];
                        }
                        for (size_t d = 0; d < D; ++d) {
                            float val = in_ptr[b * D + d];
                            float partial = dy_ptr[b * D + d] * w_ptr[d] * inv_denom;
                            float correction = val * val * sum_dy_w * norm_const;
                            dx_ptr[b * D + d] = partial - correction;
                        }
                    }
                    node->input(0)->accumulate_grad(dx);
                }
                if (node->input(1) && node->input(1)->requires_grad()) {
                    const axon::tensor::Tensor& in = node->input(0)->data_;
                    const axon::tensor::Tensor& dy = node->grad_;
                    float eps = node->scalar_param_;
                    size_t B = in.dim(0);
                    size_t D = in.dim(1);
                    int64_t shape_1d[1] = {static_cast<int64_t>(D)};
                    axon::tensor::Tensor dw(shape_1d, 1, in.dtype(), in.device());
                    dw.zero();
                    const float* in_ptr = in.data<float>();
                    const float* dy_ptr = dy.data<float>();
                    float* dw_ptr = dw.data<float>();
                    for (size_t b = 0; b < B; ++b) {
                        float sum_sq = 0.0f;
                        for (size_t d = 0; d < D; ++d) {
                            float val = in_ptr[b * D + d];
                            sum_sq += val * val;
                        }
                        float inv_denom = 1.0f / sqrtf(sum_sq / static_cast<float>(D) + eps);
                        for (size_t d = 0; d < D; ++d) {
                            dw_ptr[d] += dy_ptr[b * D + d] * in_ptr[b * D + d] * inv_denom;
                        }
                    }
                    node->input(1)->accumulate_grad(dw);
                }
                break;
            }
            case OpType::SumAll: {
                if (node->input(0) && node->input(0)->requires_grad()) {
                    const axon::tensor::Tensor& dy = node->grad_;
                    const axon::tensor::Tensor& in = node->input(0)->data_;
                    axon::tensor::Tensor dx(in.shape().dims(), in.ndim(), in.dtype(), in.device());
                    float grad_val = dy.num_elements() > 0 ? dy.data<float>()[0] : 1.0f;
                    axon::ops::fill_(dx, grad_val);
                    node->input(0)->accumulate_grad(dx);
                }
                break;
            }
            case OpType::MeanAll: {
                if (node->input(0) && node->input(0)->requires_grad()) {
                    const axon::tensor::Tensor& dy = node->grad_;
                    const axon::tensor::Tensor& in = node->input(0)->data_;
                    size_t count = in.num_elements();
                    axon::tensor::Tensor dx(in.shape().dims(), in.ndim(), in.dtype(), in.device());
                    float grad_val = (dy.num_elements() > 0)
                        ? dy.data<float>()[0] / static_cast<float>(count)
                        : 1.0f / static_cast<float>(count);
                    axon::ops::fill_(dx, grad_val);
                    node->input(0)->accumulate_grad(dx);
                }
                break;
            }
            case OpType::Custom: {
                if (node->custom_backward_fn_ != nullptr) {
                    node->custom_backward_fn_(node);
                }
                break;
            }
            default:
                break;
        }
    }
}

// ============================================================================
// 3. DIFFERENTIABLE FUNCTIONAL OPERATIONS
// ============================================================================

static Node* extract_node(const Variable& v) noexcept {
    return v.node();
}

Variable add(const Variable& a, const Variable& b) noexcept {
    axon::tensor::Tensor out = axon::ops::add(a.data(), b.data());
    Node* parents[2] = {extract_node(a), extract_node(b)};
    Node* node = new Node(out, OpType::Add, parents, 2);
    return Variable(node, false);
}

Variable sub(const Variable& a, const Variable& b) noexcept {
    axon::tensor::Tensor out = axon::ops::sub(a.data(), b.data());
    Node* parents[2] = {extract_node(a), extract_node(b)};
    Node* node = new Node(out, OpType::Sub, parents, 2);
    return Variable(node, false);
}

Variable mul(const Variable& a, const Variable& b) noexcept {
    axon::tensor::Tensor out = axon::ops::mul(a.data(), b.data());
    Node* parents[2] = {extract_node(a), extract_node(b)};
    Node* node = new Node(out, OpType::Mul, parents, 2);
    return Variable(node, false);
}

Variable scale(const Variable& in, float scalar) noexcept {
    axon::tensor::Tensor out = axon::ops::scale(in.data(), scalar);
    Node* parents[1] = {extract_node(in)};
    Node* node = new Node(out, OpType::Scale, parents, 1, scalar);
    return Variable(node, false);
}

Variable relu(const Variable& in) noexcept {
    axon::tensor::Tensor out = axon::ops::relu(in.data());
    Node* parents[1] = {extract_node(in)};
    Node* node = new Node(out, OpType::ReLU, parents, 1);
    return Variable(node, false);
}

Variable silu(const Variable& in) noexcept {
    axon::tensor::Tensor out = axon::ops::silu(in.data());
    Node* parents[1] = {extract_node(in)};
    Node* node = new Node(out, OpType::SiLU, parents, 1);
    return Variable(node, false);
}

Variable gelu(const Variable& in) noexcept {
    axon::tensor::Tensor out = axon::ops::gelu(in.data());
    Node* parents[1] = {extract_node(in)};
    Node* node = new Node(out, OpType::GELU, parents, 1);
    return Variable(node, false);
}

Variable sigmoid(const Variable& in) noexcept {
    axon::tensor::Tensor out = axon::ops::sigmoid(in.data());
    Node* parents[1] = {extract_node(in)};
    Node* node = new Node(out, OpType::Sigmoid, parents, 1);
    return Variable(node, false);
}

Variable tanh_op(const Variable& in) noexcept {
    axon::tensor::Tensor out = axon::ops::tanh(in.data());
    Node* parents[1] = {extract_node(in)};
    Node* node = new Node(out, OpType::Tanh, parents, 1);
    return Variable(node, false);
}

Variable matmul(const Variable& a, const Variable& b) noexcept {
    axon::tensor::Tensor out = axon::ops::matmul(a.data(), b.data());
    Node* parents[2] = {extract_node(a), extract_node(b)};
    Node* node = new Node(out, OpType::MatMul, parents, 2);
    return Variable(node, false);
}

Variable linear(const Variable& input, const Variable& weight, const Variable* bias) noexcept {
    const axon::tensor::Tensor* bias_ptr = bias ? &bias->data() : nullptr;
    axon::tensor::Tensor out = axon::ops::linear(input.data(), weight.data(), bias_ptr);
    if (bias_ptr) {
        Node* parents[3] = {extract_node(input), extract_node(weight), extract_node(*bias)};
        Node* node = new Node(out, OpType::Linear, parents, 3);
        return Variable(node, false);
    } else {
        Node* parents[2] = {extract_node(input), extract_node(weight)};
        Node* node = new Node(out, OpType::Linear, parents, 2);
        return Variable(node, false);
    }
}

Variable rmsnorm(const Variable& in, const Variable& weight, float eps) noexcept {
    axon::tensor::Tensor out = axon::ops::rmsnorm(in.data(), weight.data(), eps);
    Node* parents[2] = {extract_node(in), extract_node(weight)};
    Node* node = new Node(out, OpType::RMSNorm, parents, 2, eps);
    return Variable(node, false);
}

Variable sum_all(const Variable& in) noexcept {
    float val = axon::ops::sum_all(in.data());
    int64_t shape_1d[1] = {1};
    axon::tensor::Tensor out(shape_1d, 1, in.data().dtype(), in.data().device());
    out.data<float>()[0] = val;
    Node* parents[1] = {extract_node(in)};
    Node* node = new Node(out, OpType::SumAll, parents, 1);
    return Variable(node, false);
}

Variable mean_all(const Variable& in) noexcept {
    float val = axon::ops::mean_all(in.data());
    int64_t shape_1d[1] = {1};
    axon::tensor::Tensor out(shape_1d, 1, in.data().dtype(), in.data().device());
    out.data<float>()[0] = val;
    Node* parents[1] = {extract_node(in)};
    Node* node = new Node(out, OpType::MeanAll, parents, 1);
    return Variable(node, false);
}

Variable mse_loss(const Variable& pred, const Variable& target) noexcept {
    Variable diff = sub(pred, target);
    Variable sq = mul(diff, diff);
    return mean_all(sq);
}

Variable custom_op(
    const axon::tensor::Tensor& forward_out,
    Node* const* inputs,
    size_t num_inputs,
    CustomBackwardFn backward_fn,
    float scalar_param
) noexcept {
    Node* node = new Node(forward_out, OpType::Custom, inputs, num_inputs, scalar_param, backward_fn);
    return Variable(node, false);
}

} // namespace autograd
} // namespace axon
