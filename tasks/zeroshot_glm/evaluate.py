"""GLM zero-shot evaluation."""

import os
import glob
import json
import time
import torch
import numpy as np

from megatron import get_args, get_tokenizer
from megatron import print_rank_0
from megatron import mpu
from megatron.checkpointing import load_checkpoint
from megatron.training import get_model
from megatron.utils import unwrap_model, report_memory
from megatron.p2p_communication import recv_forward, send_forward

from .datasets import build_dataset

# These are needed to unwrap the model, would be nice to put these in megatron.utils if possible?
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model.distributed import DistributedDataParallel as LocalDDP
from megatron.model.module import Float16Module

from pretrain_glm import model_provider as glm_model_provider


def get_model_provider():
    """Based on evaluation metric set the parallel-output flag and
    return the model provider."""

    def model_provider(pre_process=True, post_process=True):
        """Build the model."""
        model = glm_model_provider(pre_process=pre_process, post_process=post_process)
        # model = GLMForMultiTokenCloze(model)
        return model

    return model_provider


def process_data(batch):
    return (
        batch['tokens'].to(device=torch.cuda.current_device()).long(),
        batch['targets'].to(device=torch.cuda.current_device()).long(),
        batch['position_ids'].to(device=torch.cuda.current_device()).long(),
        batch['attention_mask'].to(device=torch.cuda.current_device()).bool().unsqueeze(1)
    )


def forward_step(batch, model):
    """Forward step."""

    # Get the batch.
    tokens, targets, position_ids, attention_mask = process_data(batch)

    # Tell the model what our actual batch size will be
    args = get_args()
    args.micro_batch_size, args.seq_length = tokens.shape[:2]

    input_tensor = recv_forward()

    # Forward pass through the model.
    unwrapped_model = unwrap_model(
        model, (torchDDP, LocalDDP, Float16Module))
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
        if batch['is_single_token'].any():
            target_ids = batch['choice_target_ids'][0]
            logits = output[batch_ids, target_ids, batch['choices']]
            choice_logits = logits.squeeze(0).tolist()
            # if mpu.get_tensor_model_parallel_rank() == 0:
            #     for target_ids in batch['choice_target_ids']:
            #         print(output[batch_ids, target_ids, target_vocab].squeeze(0).tolist())
            #     print("-------")
        # Multi token
        else:
            for target_ids in batch['choice_target_ids']:
                logits = output[batch_ids, target_ids, targets[batch_ids, target_ids]]
                choice_logits.append(logits.squeeze(0).sum(dim=-1).tolist())

        # if torch.distributed.get_rank() == 0:
        #     tokenizer = get_tokenizer()
        #     import pdb
        #     pdb.set_trace()

        # if mpu.get_tensor_model_parallel_rank() == 0:
        #     print(choice_logits)
        # print(choice_logits)

        return choice_logits

    return None


def evaluate(data_loader, model):
    """Evaluation."""
    args = get_args()

    # Turn on evaluation mode which disables dropout.
    model.eval()

    outputs = []
    with torch.no_grad():
        # For all the batches in the dataset.
        for iteration, batch in enumerate(data_loader):
            # Forward evaluation.
            output = forward_step(batch, model)
            # Reduce across processes.
            if mpu.is_pipeline_last_stage():
                outputs.append(np.argmax(output))

    return outputs


def build_data_loader(dataset, micro_batch_size, num_workers, drop_last):
    # Sampler.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # Data loader. Note that batch size is the per GPU batch size.
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=micro_batch_size,
                                              sampler=sampler,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              drop_last=drop_last,
                                              pin_memory=True)

    return data_loader


def main():
    """Main program."""
    args = get_args()

    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    assert args.micro_batch_size == 1

    # # Set up model and load checkpoint.
    model = get_model(get_model_provider())
    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    datasets = []
    dataloaders = []
    filenames = []

    for data in args.train_data:
        for file_name in sorted(glob.glob(os.path.join(data, "./**/*.json*"), recursive=True)):
            if file_name.endswith("_predict.json"):
                continue
            dataset = build_dataset(file_name)
            dataloader = build_data_loader(dataset, args.micro_batch_size,
                                           args.num_workers, drop_last=False)
            datasets.append(dataset)
            dataloaders.append(dataloader)
            filenames.append(file_name)
            print_rank_0(f"Loaded {file_name}")

    report_memory("Before train")

    start = time.time()
    torch.distributed.barrier()

    accuracys = []
    correct_all, sample_all = 0, 0
    for i in range(len(dataloaders)):
        outputs = evaluate(dataloaders[i], model)
        if mpu.is_pipeline_last_stage() and mpu.get_tensor_model_parallel_rank() == 0:
            world_size = mpu.get_data_parallel_world_size()
            rank = mpu.get_data_parallel_rank()
            # print(f"rank: {rank}, {len(datasets[i])} {len(outputs)}")
            predicted_gathered = torch.tensor(np.zeros((len(datasets[i]) + world_size - 1) // world_size * world_size),
                                              dtype=torch.int64, device=torch.cuda.current_device())
            predicted_gathered[rank::world_size] = \
                torch.tensor(outputs, dtype=torch.int64, device=torch.cuda.current_device())
            torch.distributed.all_reduce(predicted_gathered, group=mpu.get_data_parallel_group())
            predicted_gathered = predicted_gathered[:len(datasets[i])].tolist()

            if mpu.get_data_parallel_rank() == 0:
                correct_num = 0
                with open(f"{filenames[i].replace('.json', '')}_predict.json", 'w') as file:
                    for item, output in zip(datasets[i].data, predicted_gathered):
                        file.write(json.dumps({'predict': output}) + '\n')
                        correct_num += item['label'] == output
                accuracy = correct_num / len(datasets[i])
                accuracys.append(accuracy)
                print(f"Finish {filenames[i]}, accuracy = {accuracy * 100:.3f}%")
                correct_all += correct_num
                sample_all += len(datasets[i])

    if mpu.is_pipeline_last_stage() and mpu.get_tensor_model_parallel_rank() == 0 and mpu.get_data_parallel_rank() == 0:
        print("Micro accuracy:", correct_all / sample_all)

    torch.distributed.barrier()
    print_rank_0(f'done :-), total time: {time.time() - start}')
