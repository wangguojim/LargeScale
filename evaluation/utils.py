import torch

from megatron import get_args, get_tokenizer
from megatron import mpu
from megatron.utils import unwrap_model
from megatron.p2p_communication import recv_forward, send_forward

from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model.distributed import DistributedDataParallel as LocalDDP
from megatron.model.module import Float16Module

from glm.generation.utils import sample_sequence_batch


def process_data(batch):
    return (
        batch["tokens"].to(device=torch.cuda.current_device()).long(),
        batch["targets"].to(device=torch.cuda.current_device()).long(),
        batch["position_ids"].to(device=torch.cuda.current_device()).long(),
        batch["attention_mask"].to(device=torch.cuda.current_device()).bool().unsqueeze(1),
    )


def build_data_loader(dataset, micro_batch_size, num_workers, drop_last):
    # Sampler.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    # Data loader. Note that batch size is the per GPU batch size.
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,
    )

    return data_loader


def cond_log_prob(batch, model):
    # Get the batch.
    tokens, targets, position_ids, attention_mask = process_data(batch)

    # Tell the model what our actual batch size will be
    args = get_args()
    args.micro_batch_size, args.seq_length = tokens.shape[:2]

    input_tensor = recv_forward()

    # Forward pass through the model.
    unwrapped_model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module))
    unwrapped_model.set_input_tensor(input_tensor)
    output = model(tokens, position_ids, attention_mask)

    send_forward(output)

    if mpu.is_pipeline_last_stage():
        output = mpu.gather_from_tensor_model_parallel_region(output)
        # output: [b, sq, vocab]
        output = torch.nn.functional.log_softmax(output, dim=-1)
        batch_ids = torch.arange(tokens.size(0), dtype=tokens.dtype, device=tokens.device).unsqueeze(1)

        choice_logits = []

        # Single token
        if batch["is_single_token"].any():
            target_ids = batch["choice_target_ids"][0]
            logits = output[batch_ids, target_ids, batch["choices"]]
            choice_logits = logits.squeeze(0).tolist()
        # Multi token
        else:
            for target_ids in batch["choice_target_ids"]:
                logits = output[batch_ids, target_ids, targets[batch_ids, target_ids]]
                choice_logits.append(logits.squeeze(0).sum(dim=-1).tolist())

        return choice_logits

    return None


def generate_text(model, batch, max_length=2048):
    args = get_args()
    tokenizer = get_tokenizer()

    args.recompute = False
    args.benchmark = False
    args.eos_id = [tokenizer.get_special_token("eop"), tokenizer.get_special_token("eos")]

    return sample_sequence_batch(
        model,
        batch["tokens"].to(device=torch.cuda.current_device()).long(),
        batch["context_length"].to(device=torch.cuda.current_device()).long(),
        batch["attention_mask"].to(device=torch.cuda.current_device()).bool().unsqueeze(1),
        batch["position_ids"].to(device=torch.cuda.current_device()).long(),
        maxlen=max_length - 1,
        no_eos=args.no_eos_generation,
        no_punctuation=args.no_punctuation_generation
    )
