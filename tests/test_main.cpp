#include "axon/tensor/tensor.hpp"
#include <iostream>

int main() {
    // Test 1: Shape creation and tensor allocation (e.g., shape [2, 3])
    int64_t sizes[] = {2, 3};
    axon::tensor::Tensor t(sizes, 2, axon::core::DType::Float32);

    std::cout << "Axon Tensor Initialized Successfully!" << std::endl;
    std::cout << "Number of dimensions: " << t.ndim() << std::endl;
    std::cout << "Total elements: " << t.num_elements() << std::endl;
    std::cout << "Is contiguous: " << (t.is_contiguous() ? "true" : "false") << std::endl;
    std::cout << "Is SIMD aligned (64-byte): " << (t.is_simd_aligned() ? "yes" : "no") << std::endl;

    return 0;
}
