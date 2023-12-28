#include <cuda.h>

#include "type_shim.h"

#define A(i, j, k, r) A[i * b * np * hn + j * np * hn + k * hn + r]
#define B(i, j, r) B[i * b * hn + j * hn + r]
#define C(i, j, r) C[i * b * hn + j * hn + r]
#define res(i, j, k, r) res[i * b * np * hn + j * np * hn + k * hn + r]
#define res_grad(i, j, k, r) \
    res_grad[i * b * np * hn + j * np * hn + k * hn + r]
#define grad_out(i, j, k, r) \
    grad_out[i * b * np * hn + j * np * hn + k * hn + r]

const int block_np = 16, block_hn = 16;

template <typename U>
__global__ __global__ void tensorMult_gpu(int sq, int b, int np, int hn,
                                          const U* A, const U* B, const U* C,
                                          U* res) {
    int i = blockIdx.x, j = blockIdx.y;
    int k0 = threadIdx.y, r0 = threadIdx.x;
    for (int k = k0; k < np; k += 16) {
        for (int r = r0; r < hn; r += 16) {
            U temp = (r + hn / 2 < hn) ? -A(i, j, k, r + hn / 2)
                                       : A(i, j, k, r + hn / 2 - hn);
            res(i, j, k, r) = A(i, j, k, r) * B(i, j, r) + temp * C(i, j, r);
        }
    }
}

template <typename U>
void host_apply_tensorMult_gpu(int sq, int b, int np, int hn, const U* A,
                               const U* B, const U* C, U* res) {
    tensorMult_gpu<<<dim3(sq, b), dim3(block_hn, block_np)>>>(sq, b, np, hn, A,
                                                              B, C, res);
}

void tensorMult_gpu_launch(int sq, int b, int np, int hn, at::Tensor* A,
                           at::Tensor* B, at::Tensor* C, at::Tensor* res) {
    DISPATCH_FLOAT_HALF_AND_BFLOAT_TYPES(
        B->scalar_type(), "forward_gpu",
        host_apply_tensorMult_gpu(
            sq, b, np, hn, A->data_ptr<scalar_t>(), B->data_ptr<scalar_t>(),
            C->data_ptr<scalar_t>(), res->data_ptr<scalar_t>());)
}

template <typename U>
__global__ __global__ void tensorMult_backward_gpu(int sq, int b, int np,
                                                   int hn, const U* grad_out,
                                                   const U* B, const U* C,
                                                   U* res_grad) {
    int i = blockIdx.x, j = blockIdx.y;
    int k0 = threadIdx.y, r0 = threadIdx.x;
    for (int k = k0; k < np; k += 16) {
        for (int r = r0; r < hn; r += 16) {
            U temp1 = (r + hn / 2 < hn) ? grad_out(i, j, k, r + hn / 2)
                                        : grad_out(i, j, k, r + hn / 2 - hn);
            U temp2 = (r + hn / 2 < hn) ? C(i, j, r + hn / 2)
                                        : -C(i, j, r + hn / 2 - hn);
            res_grad(i, j, k, r) =
                grad_out(i, j, k, r) * B(i, j, r) + temp1 * temp2;
        }
    }
}

template <typename U>
void host_apply_tensorMult_backward_gpu(int sq, int b, int np, int hn,
                                        const U* grad_out, const U* B,
                                        const U* C, U* res_grad) {
    tensorMult_backward_gpu<<<dim3(sq, b), dim3(block_hn, block_np)>>>(
        sq, b, np, hn, grad_out, B, C, res_grad);
}

void tensorMult_backward_gpu_launch(int sq, int b, int np, int hn,
                                    at::Tensor* grad_out, at::Tensor* B,
                                    at::Tensor* C, at::Tensor* res_grad) {
    DISPATCH_FLOAT_HALF_AND_BFLOAT_TYPES(
        B->scalar_type(), "backward_gpu",
        host_apply_tensorMult_backward_gpu(
            sq, b, np, hn, grad_out->data_ptr<scalar_t>(),
            B->data_ptr<scalar_t>(), C->data_ptr<scalar_t>(),
            res_grad->data_ptr<scalar_t>());)
}
