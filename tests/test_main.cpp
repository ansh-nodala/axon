#include "axon/core/memory.hpp"
#include "axon/tensor/tensor.hpp"
#include "axon/ops/elementwise.hpp"
#include "axon/ops/matmul.hpp"
#include "axon/ops/reductions.hpp"
#include "axon/autograd/node.hpp"
#include "axon/nn/module.hpp"
#include <iostream>

// Simple assertion helper for testing
#define AXON_TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[FAIL] " << msg << " (Line: " << __LINE__ << ")" << std::endl; \
            std::exit(1); \
        } \
    } while (0)

void test_tensor_and_memory() {
    std::cout << "[*] Running Tensor & Memory Allocation Tests..." << std::endl;
    
    int64_t dims[2] = {2, 3};
    axon::tensor::Tensor t(dims, 2, axon::core::DType::Float32, axon::core::DeviceType::CPU);
    
    AXON_TEST_ASSERT(t.ndim() == 2, "Tensor ndim should be 2");
    AXON_TEST_ASSERT(t.dim(0) == 2, "Tensor dim(0) should be 2");
    AXON_TEST_ASSERT(t.dim(1) == 3, "Tensor dim(1) should be 3");
    AXON_TEST_ASSERT(t.num_elements() == 6, "Total elements should be 6");

    t.zero();
    float* ptr = t.data<float>();
    for (size_t i = 0; i < 6; ++i) {
        AXON_TEST_ASSERT(ptr[i] == 0.0f, "Tensor zero initialization failed");
    }

    std::cout << "[+] Tensor & Memory Tests Passed Successfully!" << std::endl;
}

void test_elementwise_and_matmul() {
    std::cout << "[*] Running Elementwise & MatMul Tests..." << std::endl;

    int64_t dims_a[2] = {2, 2};
    int64_t dims_b[2] = {2, 2};

    axon::tensor::Tensor a(dims_a, 2, axon::core::DType::Float32, axon::core::DeviceType::CPU);
    axon::tensor::Tensor b(dims_b, 2, axon::core::DType::Float32, axon::core::DeviceType::CPU);

    a.data<float>()[0] = 1.0f; a.data<float>()[1] = 2.0f;
    a.data<float>()[2] = 3.0f; a.data<float>()[3] = 4.0f;

    b.data<float>()[0] = 5.0f; b.data<float>()[1] = 6.0f;
    b.data<float>()[2] = 7.0f; b.data<float>()[3] = 8.0f;

    // Test addition
    axon::tensor::Tensor c = axon::ops::add(a, b);
    AXON_TEST_ASSERT(c.data<float>()[0] == 6.0f, "Elementwise add failed at index 0");
    AXON_TEST_ASSERT(c.data<float>()[3] == 12.0f, "Elementwise add failed at index 3");

    // Test matrix multiplication
    axon::tensor::Tensor mm = axon::ops::matmul(a, b);
    // [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
    // [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
    AXON_TEST_ASSERT(mm.data<float>()[0] == 19.0f, "MatMul failed at [0,0]");
    AXON_TEST_ASSERT(mm.data<float>()[1] == 22.0f, "MatMul failed at [0,1]");
    AXON_TEST_ASSERT(mm.data<float>()[2] == 43.0f, "MatMul failed at [1,0]");
    AXON_TEST_ASSERT(mm.data<float>()[3] == 50.0f, "MatMul failed at [1,1]");

    std::cout << "[+] Elementwise & MatMul Tests Passed Successfully!" << std::endl;
}

void test_neural_network_and_optimizers() {
    std::cout << "[*] Running Neural Network & Optimizer Integration Test..." << std::endl;

    // 1. Initialize Linear Layer (in=2, out=1, bias=true)
    axon::nn::Linear linear(2, 1, true);

    // 2. Setup AdamW Optimizer
    axon::nn::optim::AdamW optimizer(linear, 0.05f);

    // 3. Create input [1, 2] -> [1.0, 2.0]
    int64_t in_dims[2] = {1, 2};
    axon::tensor::Tensor input_tensor(in_dims, 2, axon::core::DType::Float32, axon::core::DeviceType::CPU);
    input_tensor.data<float>()[0] = 1.0f;
    input_tensor.data<float>()[1] = 2.0f;
    axon::autograd::Variable x(input_tensor, false);

    // 4. Create target [1, 1] -> [5.0]
    int64_t tgt_dims[2] = {1, 1};
    axon::tensor::Tensor tgt_tensor(tgt_dims, 2, axon::core::DType::Float32, axon::core::DeviceType::CPU);
    tgt_tensor.data<float>()[0] = 5.0f;
    axon::autograd::Variable target(tgt_tensor, false);

    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    // Training loop for 5 steps
    for (int step = 0; step < 5; ++step) {
        optimizer.zero_grad();

        axon::autograd::Variable pred = linear.forward(x);
        axon::autograd::Variable loss = axon::autograd::mse_loss(pred, target);

        float current_loss = loss.data().data<float>()[0];
        if (step == 0) initial_loss = current_loss;
        if (step == 4) final_loss = current_loss;

        std::cout << "Step " << step << " | Loss: " << current_loss << std::endl;

        loss.backward();
        optimizer.step();
    }

    AXON_TEST_ASSERT(final_loss < initial_loss, "Optimizer failed to reduce loss over steps");

    std::cout << "[+] Neural Network & Optimizer Tests Passed Successfully!" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "   AXON FRAMEWORK COMPREHENSIVE TEST SUITE" << std::endl;
    std::cout << "========================================" << std::endl;

    test_tensor_and_memory();
    test_elementwise_and_matmul();
    test_neural_network_and_optimizers();

    std::cout << "========================================" << std::endl;
    std::cout << "   ALL AXON TESTS PASSED SUCCESSFULLY!  " << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
