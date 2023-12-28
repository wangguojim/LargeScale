import torch
import torch.nn.functional as F
import time
import os
import pathlib
import subprocess

from torch.utils import cpp_extension

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def load():
    arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', None)
    if arch_list is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = ""

    # Build path
    srcpath = pathlib.Path('../megatron/fused_kernels').absolute()
    buildpath = srcpath / 'build'
    buildpath.mkdir(parents=True, exist_ok=True)

    # Helper function to build the kernels.
    def _cpp_extention_load_helper(name, sources, extra_cuda_flags):
        return cpp_extension.load(
            name=name,
            sources=sources,
            build_directory=buildpath,
            extra_cflags=['-O3',],
            extra_cuda_cflags=['-O3',
                               '--use_fast_math'] + extra_cuda_flags,
            verbose=True
        )
                               # '-gencode', 'arch=compute_70,code=sm_70',

    extra_cuda_flags = []
    sources=[srcpath / 'rotary_positional_embedding.cpp',
             srcpath / 'rotary_positional_embedding.cu']
    rotary_positional_embedding_cuda = _cpp_extention_load_helper(
        "rotary_positional_embedding_cuda", sources, extra_cuda_flags)


class RotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, base=10000, precision=torch.half, learnable=False):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.learnable = learnable
        if learnable:
            self.inv_freq = torch.nn.Parameter(inv_freq)
            self.max_seq_len_cached = None
        else:
            self.register_buffer('inv_freq', inv_freq)
            self.max_seq_len_cached = None
            self.cos_cached = None
            self.sin_cached = None
        self.precision = precision

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()
            # [sx, 1 (b * np), hn]
            cos_cached = emb.cos()[:, None, :]
            sin_cached = emb.sin()[:, None, :]
            if self.precision == torch.bfloat16:
                cos_cached = cos_cached.bfloat16()
                sin_cached = sin_cached.bfloat16()
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


class RotaryPositionalEmbeddingFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, cos, sin):
        import rotary_positional_embedding_cuda

        q_ = q.contiguous()
        cos_ = cos.contiguous()
        sin_ = sin.contiguous()
        output = rotary_positional_embedding_cuda.forward(*q.shape, q_, cos_, sin_)
        ctx.save_for_backward(cos, sin)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        import rotary_positional_embedding_cuda

        cos_, sin_ = ctx.saved_tensors
        grad_q = rotary_positional_embedding_cuda.backward(*grad_output.shape, grad_output, cos_, sin_)

        return grad_q, None, None

# rotary pos emb helpers:

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos, sin = cos[offset:q.shape[0] + offset, ...], sin[offset:q.shape[0] + offset, ...]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def apply_rotary_pos_emb_fused(q, k, cos, sin, offset: int = 0):

    cos, sin = cos[offset:q.shape[0] + offset, ...], sin[offset:q.shape[0] + offset, ...]
    q = RotaryPositionalEmbeddingFunction.apply(q, cos, sin)
    k = RotaryPositionalEmbeddingFunction.apply(k, cos, sin)
    return q, k


@torch.jit.script
def apply_rotary_pos_emb_index(q, k, cos, sin, position_id):
    # position_id: [sq, b], q: [sq, b * np, hn] -> [sq, b, np, hn], cos: [sq, 1, hn] -> [sq, b, 1, hn]
    sq, b, np = position_id.size(0), position_id.size(1), q.size(1) // position_id.size(1)
    q, k = q.view(sq, b, np, -1), k.view(sq, b, np, -1)
    cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(2), \
               F.embedding(position_id, sin.squeeze(1)).unsqueeze(2)
    q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    return q.view(sq, b * np, -1), k.view(sq, b * np, -1)


def apply_rotary_pos_emb_index_fused(q, k, cos, sin, position_id):
    # position_id: [sq, b], q: [sq, b * np, hn] -> [sq, b, np, hn], cos: [sq, 1, hn] -> [sq, b, 1, hn]
    sq, b, np = position_id.size(0), position_id.size(1), q.size(1) // position_id.size(1)
    q, k = q.view(sq, b, np, -1), k.view(sq, b, np, -1)
    cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(2), \
               F.embedding(position_id, sin.squeeze(1)).unsqueeze(2)
    q = RotaryPositionalEmbeddingFunction.apply(q, cos, sin)
    k = RotaryPositionalEmbeddingFunction.apply(k, cos, sin)
    return q.view(sq, b * np, -1), k.view(sq, b * np, -1)


if __name__ == '__main__':
    load()

    RUNS = 100

    SEQ_LEN = 2048
    HIDDEN = 12288
    NHEADS = 96
    BATCH_SIZE = 1
    HP = HIDDEN // NHEADS
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    q = torch.randn((SEQ_LEN, BATCH_SIZE * NHEADS, HP)).cuda().float()
    k = torch.randn((SEQ_LEN, BATCH_SIZE * NHEADS, HP)).cuda().float()

    q.requires_grad = True
    k.requires_grad = True

    position_id = torch.randint(0, SEQ_LEN, (SEQ_LEN, BATCH_SIZE)).cuda()
    rotary_embedding = RotaryEmbedding(HP).cuda().float()
    cos, sin = rotary_embedding.forward(k, seq_len=SEQ_LEN)

    for _ in range(10):
        apply_rotary_pos_emb_index_fused(q, k, cos, sin, position_id)
        apply_rotary_pos_emb_index(q, k, cos, sin, position_id)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(RUNS):
        q.grad = None
        k.grad = None
        q2, k2 = apply_rotary_pos_emb_index_fused(q, k, cos, sin, position_id)
        loss = 12 * (q2.mean() + k2.mean()) ** 2
        loss.backward()
    gradq2, gradk2 = q.grad, k.grad
    end.record()
    torch.cuda.synchronize()

    print(f"custom_kernel time {start.elapsed_time(end)} ms")

    start.record()
    for _ in range(RUNS):
        q.grad = None
        k.grad = None
        q1, k1 = apply_rotary_pos_emb_index(q, k, cos, sin, position_id)
        loss = 12 * (q1.mean() + k1.mean()) ** 2
        loss.backward()
    gradq1, gradk1 = q.grad, k.grad
    end.record()
    torch.cuda.synchronize()

    print(f"jit time {start.elapsed_time(end)} ms")

    # q.grad = None
    # k.grad = None
    # q1, k1 = apply_rotary_pos_emb_index(q.double(), k.double(), cos.double(), sin.double(), position_id)
    # loss = 12 * (q1.mean() + k1.mean()) ** 2
    # loss.backward()
    # gradq1, gradk1 = q.grad, k.grad

    def rerr(x1, x2):
        return ((x1 - x2) / (x1 + x2 + 1e-6)).abs().max()

    print(((q1 - q2).abs().max()))
    # print(q2)

    print(
        f"element-wise relative error max: q: {rerr(q1, q2)}, k: {rerr(k1, k2)}")
    print(
        f"element-wise relative error max: q.grad: {rerr(gradq1, gradq2)}, k.grad: {rerr(gradk1, gradk2)}")
