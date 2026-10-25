#include "axon/tensor/tensor.hpp"

namespace axon {
namespace tensor {

// ============================================================================
// OUT-OF-LINE TENSOR OPERATIONS IMPLEMENTATION
// ============================================================================

Tensor Tensor::clone() const {
    // Cloning requires a contiguous source tensor layout
    AXON_ASSERT(is_contiguous());

    Tensor cloned_tensor(dims(), ndim(), dtype(), device());

    const size_t total_bytes = nbytes();
    if (AXON_LIKELY(total_bytes > 0)) {
        const void* src_ptr = raw_data();
        void* dst_ptr = cloned_tensor.raw_data();

        AXON_ASSERT(src_ptr != nullptr);
        AXON_ASSERT(dst_ptr != nullptr);

        if (is_simd_aligned() && cloned_tensor.is_simd_aligned()) {
            axon::memory::copy_memory_aligned(dst_ptr, src_ptr, total_bytes);
        } else {
            axon::memory::copy_memory(dst_ptr, src_ptr, total_bytes);
        }
    }

    return cloned_tensor;
}

Tensor Tensor::transpose(size_t dim0, size_t dim1) const {
    AXON_ASSERT(dim0 < ndim());
    AXON_ASSERT(dim1 < ndim());

    TensorShape swapped_shape = shape().transpose(dim0, dim1);

    // Create a zero-copy shared storage view referencing the underlying storage buffer.
    // We construct a non-owning Storage view wrapping storage_.data().
    Storage view_storage(
        const_cast<void*>(storage_.data()),
        storage_.num_elements(),
        storage_.dtype(),
        storage_.device()
    );

    return Tensor(static_cast<Storage&&>(view_storage), swapped_shape, storage_offset_);
}

Tensor Tensor::view(const int64_t* sizes, size_t new_ndim) const {
    // Reshaping/viewing requires contiguous memory layout
    AXON_ASSERT(is_contiguous());

    TensorShape new_shape(sizes, new_ndim);
    AXON_ASSERT(new_shape.total_elements() == num_elements());

    // Create a zero-copy shared storage view referencing the underlying storage buffer.
    Storage view_storage(
        const_cast<void*>(storage_.data()),
        storage_.num_elements(),
        storage_.dtype(),
        storage_.device()
    );

    return Tensor(static_cast<Storage&&>(view_storage), new_shape, storage_offset_);
}

} // namespace tensor
} // namespace axon
