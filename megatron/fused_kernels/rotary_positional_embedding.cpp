#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "type_shim.h"

void tensorMult_gpu_launch(int sq, int b, int np, int hn, at::Tensor* A,
                           at::Tensor* B, at::Tensor* C, at::Tensor* res);
void tensorMult_backward_gpu_launch(int sq, int b, int np, int hn,
                                    at::Tensor* grad_out, at::Tensor* B,
                                    at::Tensor* C, at::Tensor* res_grad);

at::Tensor forward_gpu(int sq, int b, int np, int hn, at::Tensor tensor_A,
                       at::Tensor tensor_B, at::Tensor tensor_C) {
    at::Tensor tensor_res =
        at::empty({sq, b, np, hn},
                  at::device(torch::kCUDA).dtype(tensor_B.scalar_type()));
    tensorMult_gpu_launch(sq, b, np, hn, &tensor_A, &tensor_B, &tensor_C,
                          &tensor_res);
    return tensor_res;
}

at::Tensor backward_gpu(int sq, int b, int np, int hn,
                        at::Tensor tensor_grad_out, at::Tensor tensor_B,
                        at::Tensor tensor_C) {
    at::Tensor tensor_res =
        at::empty({sq, b, np, hn},
                  at::device(torch::kCUDA).dtype(tensor_B.scalar_type()));
    tensorMult_backward_gpu_launch(sq, b, np, hn, &tensor_grad_out, &tensor_B,
                                   &tensor_C, &tensor_res);
    return tensor_res;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_gpu, "forward for gpu");
    m.def("backward", &backward_gpu, "backward for gpu");
}
