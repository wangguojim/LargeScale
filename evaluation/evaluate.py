"""GLM zero-shot evaluation."""

import os
import glob
import time
import itertools
import torch
import torch.distributed as dist
import numpy as np

from collections import defaultdict
from megatron import get_args
from megatron import print_rank_0, print_rank_last
from megatron import mpu
from megatron.checkpointing import load_checkpoint
from megatron.training import get_model

from datasets import build_dataset

from pretrain_glm import model_provider as glm_model_provider
from evaluation.utils import build_data_loader, cond_log_prob, generate_text


def evaluate(dataset, data_loader, model):
    args = get_args()

    model.eval()

    prediction = []
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            if dataset.task_type == "mul":
                output = cond_log_prob(batch, model)
                if mpu.is_pipeline_last_stage():
                    prediction.append(np.argmax(output))
            elif dataset.task_type == "gen":
                blank = []
                max_tgt_length = max([target.size(1) for target in batch['targets']])
                for length, (tokens, _) in enumerate(generate_text(model, batch)):
                    if mpu.is_pipeline_last_stage():
                        token = tokens[0, batch["context_length"] + len(blank)].item()
                        blank.append(token)
                    if max_tgt_length + args.generation_tolerance_length == length + 1:
                        break
                if mpu.is_pipeline_last_stage():
                    while len(blank) > 0 and blank[-1] in args.eos_id:
                        blank = blank[:-1]  # drop eos token
                    prediction.append(blank)

    result = None
    if mpu.is_pipeline_last_stage():
        world_size = mpu.get_data_parallel_world_size()
        prediction_gathered = [None for _ in range(world_size)]
        dist.all_gather_object(prediction_gathered, prediction, group=mpu.get_data_parallel_group())
        prediction = list(itertools.chain(*zip(*prediction_gathered)))[: len(dataset)]
        result = {key: metric(prediction, dataset.data) for key, metric in dataset.metrics}

    return result, prediction


def main():
    """Main program."""
    args = get_args()

    assert args.micro_batch_size == 1

    model = get_model(glm_model_provider)
    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    dist.barrier()
    start = time.time()

    for task in args.task:
        datasets, dataloaders, filenames = [], [], []

        print_rank_last(f"Evaluating task {task}")
        for root, dirs, files in os.walk(os.path.join(args.eval_data_path, task)):
            if not dirs:  # leaf dirs
                for filename in files:
                    if filename.endswith(".jsonl"):
                        filename = os.path.join(root, filename)
        # for file_name in sorted(glob.glob(os.path.join(args.eval_data_path, task, "**/*.jsonl"), recursive=True)):
            # if file_name.endswith("_predict.jsonl") or file_name.endswith("test.jsonl"):
            #     continue
            # print_rank_0(f"Loading {file_name}")
                        dataset = build_dataset(filename)
                        dataloader = build_data_loader(dataset, args.micro_batch_size, args.num_workers, drop_last=False)
                        datasets.append(dataset)
                        dataloaders.append(dataloader)
                        filenames.append(filename)

        if len(datasets) == 0:
            continue

        result_dict_all = defaultdict(lambda: [])
        weight = []
        for dataset, dataloader, filename in zip(datasets, dataloaders, filenames):
            result_dict, _ = evaluate(dataset, dataloader, model)
            if dist.get_rank() == mpu.get_pipeline_model_parallel_last_rank():
                output_str = f"    Finish {filename}"
                for key, value in result_dict.items():
                    result_dict_all[key].append(value)
                    output_str += f", {key} = {value:.3f}%"
                print_rank_last(output_str)
                weight.append(len(dataset))
        print_rank_last(f"Task {task}:")
        for key, value in result_dict_all.items():
            idx = np.argmax(value)
            print_rank_last(
                f"    Metric {key}: max = {np.max(value):.3f}"
                f" | median = {np.median(value):.3f}, average = {(np.array(value) * np.array(weight) / np.sum(weight)).sum():.3f}"
                f" | ({'/'.join(result_dict_all.keys())}) = "
                f"{'/'.join(map(lambda x: f'{x[idx]:.3f}', result_dict_all.values()))}"
            )

    dist.barrier()
    print_rank_0(f"done :-), total time: {time.time() - start}")
