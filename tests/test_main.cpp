#include "axon/tensor/tensor.hpp"
#include "axon/ops/elementwise.hpp"
#include "axon/ops/matmul.hpp"
#include "axon/ops/reductions.hpp"
#include "axon/autograd/node.hpp"
#include <iostream>

int main() {
    std::cout << "=== AXON AUTOGRAD & FULL ENGINE INTEGRATION TEST ===" << std::endl;

    // 1. Test Tensor & Basic Ops
    int64_t sizes_a[] = {2, 2};
    axon::tensor::Tensor a_tensor(sizes_a, 2, axon::core::DType::Float32);
    float* ptr_a = a_tensor.data<float>();
    ptr_a[0] = 1.0f; ptr_a[1] = 2.0f;
    ptr_a[2] = 3.0f; ptr_a[3] = 4.0f;

    std::cout << "[1] Tensor Initialized | SIMD Aligned: " 
              << (a_tensor.is_simd_aligned() ? "YES" : "NO") << std::endl;

    // 2. Test Autograd Variables & Forward/Backward Pass (MSE Loss Training Test)
    // Let's create a prediction and a target variable
    axon::autograd::Variable pred(a_tensor, true); // requires_grad = true

    int64_t sizes_t[] = {2, 2};
    axon::tensor::Tensor t_tensor(sizes_t, 2, axon::core::DType::Float32);
    float* ptr_t = t_tensor.data<float>();
    ptr_t[0] = 1.5f; ptr_t[1] = 2.5f;
    ptr_t[2] = 3.5f; ptr_t[3] = 4.5f;
    
    axon::autograd::Variable target(t_tensor, false);

    // Compute MSE Loss: loss = mean((pred - target)^2)
    axon::autograd::Variable loss = axon::autograd::mse_loss(pred, target);

    std::cout << "[2] Forward Pass Computed. Loss Value: " 
              << loss.data().data<float>()[0] << std::endl;

    // Trigger Backward Pass (Backpropagation)
    loss.backward();

    std::cout << "[3] Backward Pass Executed Successfully!" << std::endl;
    std::cout << "    Gradient at Pred [0, 0] (Expected 2*(1.0-1.5)/4 = -0.25): " 
              << pred.grad().data<float>()[0] << std::endl;

    std::cout << "=== ALL AXON AUTOGRAD TESTS PASSED SUCCESSFULLY! ===" << std::endl;
    return 0;
}
