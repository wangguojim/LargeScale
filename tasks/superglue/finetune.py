# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SuperGLUE finetune/evaluation."""
import torch
import os

from collections import OrderedDict
from megatron import get_args, get_timers, print_rank_0, get_tokenizer, mpu
from megatron.enums import PositionEmbeddingType
from megatron.utils import average_losses_across_data_parallel_group
from tasks.finetune_utils import finetune
from tasks.superglue.dataset import SuperGlueDataset, PROCESSORS, get_output_func
from tasks.superglue.dataset import MULTI_CHOICE_DATASETS
from tasks.superglue.evaluate import qa_exact_match, qa_f1, multirc_em
from tasks.superglue.pvp import PVPS
from tasks.superglue.eval_utils import accuracy_metric, f1_macro_metric, f1_metric, accuracy_func_provider
from tasks.superglue.data_utils import build_data_loader
from pretrain_glm import model_provider as glm_model_provider
from pretrain_glm import process_data as process_batch_lm
from glm.model import GLMForSingleTokenCloze, GLMForMultiTokenClozeFast, GLMForMultiTokenCloze
from tasks.finetune_utils import cross_entropy_loss_func
from functools import partial


DEFAULT_METRICS = {
    "record": [("EM", qa_exact_match), ("F1", qa_f1)],
    "copa": [("accuracy", accuracy_metric)],
    "rte": [("accuracy", accuracy_metric)],
    "boolq": [("accuracy", accuracy_metric)],
    "wic": [("accuracy", accuracy_metric)],
    "wsc": [("accuracy", accuracy_metric)],
    "cb": [("accuracy", accuracy_metric), ("f1-macro", f1_macro_metric)],
    "multirc": [("f1a", f1_metric), ("em", multirc_em), ("acc", accuracy_metric)],
    "mnli": [("accuracy", accuracy_metric)],
    "sst2": [("accuracy", accuracy_metric)],
    "qnli": [("accuracy", accuracy_metric)],
    "qqp": [("accuracy", accuracy_metric)],
    "mrpc": [("accuracy", accuracy_metric)],
    "cola": [("accuracy", accuracy_metric)],
    "squad": [("accuracy", accuracy_metric)],
    "afqmc": [("accuracy", accuracy_metric)],
    "tnews": [("accuracy", accuracy_metric)],
    "cluewsc": [("accuracy", accuracy_metric)],
    "cmrc": [("accuracy", accuracy_metric)],
}


def train_valid_datasets_provider(pattern_text=False):
    """Provide train and validation datasets."""
    args = get_args()
    tokenizer = get_tokenizer()

    assert len(args.train_data) == 1

    task_name = args.task.lower()
    train_dataset = SuperGlueDataset(args, task_name, args.train_data[0], args.seq_length, "train", tokenizer,
                                pattern_text=pattern_text)
    valid_dataset = SuperGlueDataset(args, task_name, args.train_data[0], args.seq_length, "dev", tokenizer, for_train=True,
                                pattern_text=pattern_text)

    return train_dataset, valid_dataset


def model_provider(pre_process=True, post_process=True, model_type='multiple_choice'):
    """Build the model."""
    args = get_args()

    print_rank_0('building GLM downstream model for {} ...'.format(
        args.task))
    model = glm_model_provider(pre_process=pre_process, post_process=post_process)
    if model_type == 'multiple_choice':
        if args.cloze_eval:
            if args.multi_token:
                if args.fast_decode:
                    model = GLMForMultiTokenClozeFast(model, length_penalty=args.length_penalty)
                else:
                    model = GLMForMultiTokenCloze(model, length_penalty=args.length_penalty)
            else:
                model = GLMForSingleTokenCloze(model, take_softmax=args.adapet)
        else:
           raise NotImplementedError
    elif model_type == 'generation':
        pass
    else:
        raise NotImplementedError
    if args.prefix_prompt_length is not None:
        model.requires_grad_(False)
        for layer in model.model.language_model.encoder.layers:
            if hasattr(layer.self_attention, "prefix_prompts"):
                layer.self_attention.prefix_prompts.requires_grad_(True)
    if args.freeze_prefix_layer_num:
        for idx, layer in enumerate(model.model.language_model.encoder.layers):
            if idx < args.freeze_prefix_layer_num:
                layer.requires_grad_(False)
    return model


def process_batch(batch, args):
    """Process batch and produce inputs for the model."""
    keys = ["text", "label", "attention_mask", "position"]
    if args.cloze_eval:
        keys += ["target", "loss_mask"]
        if args.continuous_prompt:
            keys += ["prompt_pos"]
    if args.variable_num_choices:
        keys.append("choice_mask")
    # Broadcast data.
    datatype = torch.int64
    data_b = mpu.broadcast_data(keys, batch, datatype)

    if "padding_mask" in data_b:
        attention_mask = data_b['padding_mask'].float().cuda().contiguous()
        if args.fp16:
            attention_mask = attention_mask.half()
        data_b["padding_mask"] = attention_mask

    return data_b


def finetune_forward_step(batch, model):
    """Simple forward step with cross-entropy loss."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch generator').start()
    try:
        batch_ = next(batch)
    except BaseException:
        batch_ = batch

    data = process_batch(batch_, args)
    timers('batch generator').stop()

    # Forward model.
    if args.cloze_eval:
        tokens, labels, position_ids = data['text'], data['label'], data['position']
        attention_mask = data['attention_mask']

        target_ids, logit_mask = data['target'], data['loss_mask']
        if args.continuous_prompt:
            prompt_pos = data["prompt_pos"]
            result = model(tokens, position_ids, attention_mask, target_ids, logit_mask, prompt_pos=prompt_pos)
        else:
            result = model(tokens, position_ids, attention_mask, target_ids, logit_mask)
        if not args.multi_token:
            logits, lm_logits = result
        else:
            logits = result
    else:
        tokens, labels, position_ids, attention_mask = data['text'], data['label'], data['position'], data['mask']
        logits = model(tokens, position_ids, attention_mask)

    if "choice_mask" in data:
        loss_mask = data["choice_mask"]
        logits = logits * loss_mask - 10000.0 * (1.0 - loss_mask)
    if args.loss_func == "cross_entropy":
        # assert mpu.get_tensor_model_parallel_world_size() == 1
        return logits.contiguous().float(), partial(cross_entropy_loss_func, labels)
    elif args.loss_func == "hinge":
        raise NotImplementedError
        correct_logits = logits[range(logits.size(0)), labels]
        hinge_loss = 1 + logits - correct_logits.unsqueeze(1)
        hinge_loss[hinge_loss < 0.0] = 0.0
        loss = hinge_loss.sum(dim=1).mean() - 1.0
    elif args.loss_func == "generative" or args.loss_func == "mix":
        batch_size = logits.size(0)
        loss = -logits[range(batch_size), labels].mean()
        if args.loss_func == "mix":
            def mixed_cross_entropy_loss_func(loss, labels, output_tensor):
                logits = output_tensor

                # Cross-entropy loss.
                loss_func = torch.nn.CrossEntropyLoss()
                loss += loss_func(logits.contiguous().float(), labels)

                # Reduce loss for logging.
                averaged_loss = average_losses_across_data_parallel_group([loss])

                return loss, {'lm loss': averaged_loss[0]}

            return logits.contiguous().float(), partial(mixed_cross_entropy_loss_func, loss, labels)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # Reduce loss for logging.

    return loss


def forward_step_lm(batch, model, eval_metric=None):
    """Forward step."""
    args = get_args()
    timers = get_timers()
    tokenizer = get_tokenizer()

    # Get the batch.
    timers('batch generator').start()
    try:
        data = next(batch)
    except BaseException:
        data = batch

    if 'mask' in data:
        data['attention_mask'] = data.pop('mask')
    tokens, labels, loss_mask, attention_mask, position_ids = process_batch_lm(data)
    timers('batch generator').stop()

    def print_masked_text(batch_id):
        block_position_ids = position_ids[:, 1]
        position_ids_ = position_ids[:, 0]
        output_tokens = []
        sep = attention_mask[batch_id].item()
        for i, token in enumerate(tokens[batch_id, :sep].tolist()):
            if tokenizer is not None:
                token = tokenizer.IdToToken(token)
                if token.startswith('[MASK'):
                    token = f"[{position_ids_[batch_id, i].item()}, {token}]"
                if token.startswith('##') and len(output_tokens) > 0 and not output_tokens[-1].endswith(']'):
                    output_tokens[-1] += token[2:]
                else:
                    output_tokens.append(token)
            else:
                output_tokens.append(str(token))
        print(" ".join(output_tokens))
        last_index = None
        for i in range(sep, tokens.size(1)):
            if tokenizer.IdToToken(tokens[batch_id, i].item()).startswith("[sop"):
                if last_index is not None:
                    print(tokenizer.DecodeIds(tokens[batch_id, last_index: i].tolist()), "|",
                          tokenizer.DecodeIds(labels[batch_id, last_index: i].tolist())),
                    print(position_ids_[batch_id, last_index: i].tolist(),
                          block_position_ids[batch_id, last_index:i].tolist())
                last_index = i
        if last_index is not None:
            print(tokenizer.DecodeIds(tokens[batch_id, last_index:].tolist()), "|",
                  tokenizer.DecodeIds(labels[batch_id, last_index:].tolist()))
            print(position_ids_[batch_id, last_index:].tolist(), block_position_ids[batch_id, last_index:].tolist())

    # Forward model.
    if args.continuous_prompt:
        prompt_pos = data["prompt_pos"].long().cuda()
        logits = model(tokens, position_ids, attention_mask, prompt_pos=prompt_pos)
    else:
        logits = model(tokens, position_ids, attention_mask)

    if eval_metric is None or eval_metric == 'loss':
        def cross_entropy_loss_func(labels, output_tensor, loss_mask):
            logits = output_tensor

            losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(),
                                                      labels)
            loss_mask = loss_mask.view(-1)
            # The loss is not normalized for fair comparison
            loss = torch.sum(losses.view(-1) * loss_mask)

            if eval_metric is None:
                loss = loss / loss_mask.sum()

            averaged_loss = average_losses_across_data_parallel_group([loss])

            return loss, {'lm loss': averaged_loss[0]}

        return logits, partial(cross_entropy_loss_func, labels, loss_mask=loss_mask)
    elif eval_metric == 'accuracy' or eval_metric == 'classify':
        logits = mpu.gather_from_tensor_model_parallel_region(logits)
        outputs = torch.argmax(logits, -1)
        correct = (outputs == labels).float()
        correct[(1 - loss_mask).bool()] = 1
        correct = correct.prod(-1)
        if eval_metric == 'accuracy':
            correct = correct.sum()
        return correct
    else:
        raise NotImplementedError("Metric {} not implemented".format(eval_metric))


def classify_evaluate_lm(model, dataloader, example_dict, args):
    """Evaluation."""
    # Turn on evaluation mode which disables dropout.
    assert len(model) == 1
    model = model[0]
    model.eval()
    predictions, labels, examples = [], [], []
    with torch.no_grad():
        # For all the batches in the dataset.
        for iteration, batch in enumerate(dataloader):
            # Forward evaluation.
            output = forward_step_lm(batch, model, eval_metric='classify')
            uid_list = batch['uid']
            example_batch = [example_dict[uid] for uid in uid_list]
            predictions.extend(output.long().tolist())
            label = batch['label'].tolist()
            labels.extend(label)
            examples.extend(example_batch)
    return predictions, labels, examples


def metrics_func_provider(is_test=False):
    """Privde metrics callback function."""
    args = get_args()
    tokenizer = get_tokenizer()

    def single_dataset_provider(split):
        return SuperGlueDataset(args, args.task.lower(), args.train_data[0], args.seq_length, split, tokenizer)

    output_func = get_output_func(args.task.lower(), args)
    eval_func = None
    if args.task.lower() in ['wsc', 'squad'] and args.cloze_eval and not args.wsc_negative:
        eval_func = classify_evaluate_lm
    metric_dict = OrderedDict(DEFAULT_METRICS[args.task.lower()])
    return accuracy_func_provider(single_dataset_provider, metric_dict, args, is_test=is_test, eval_func=eval_func,
                                  output_func=output_func, only_rank0=False, tokenizer=tokenizer)


def main():
    args = get_args()

    assert args.glm, "Only support GLM for SuperGLUE"
    assert args.tokenizer_type == "IceTokenizer", "Only support IceTokenizer for SuperGLUE"
    assert args.position_embedding_type != PositionEmbeddingType.alibi, "Don't support alibi for finetune"

    # For compability
    args.few_superglue = False
    args.cloze_eval = True
    args.pretrained_bert = False
    args.segment_length = 0
    args.continuous_prompt = False
    args.num_prompt_tokens = 0
    args.task_mask = False
    args.prefix_prompt = False
    args.sentinel_token = False
    args.adapet = False
    args.no_block_position = False
    args.eval_batch_size = args.micro_batch_size
    args.master_ip = os.environ.get('MASTER_ADDR')
    args.master_port = os.environ.get('MASTER_PORT')

    # multi_token
    processor = PROCESSORS[args.task.lower()](args)
    pvp = PVPS[args.task.lower()](args, None, processor.get_labels(), args.seq_length,
                                  pattern_id=args.pattern_id, is_multi_token=args.multi_token,
                                  num_prompt_tokens=args.num_prompt_tokens)

    if args.task.lower() in ['wsc', 'squad'] and args.cloze_eval and not args.wsc_negative:
        finetune(train_valid_datasets_provider, partial(model_provider, model_type='generation'),
                 forward_step=forward_step_lm,
                 end_of_epoch_callback_provider=metrics_func_provider,
                 build_data_loader_fn=build_data_loader)
    else:
        if args.cloze_eval:
            multi_token = pvp.is_multi_token
        else:
            multi_token = args.task.lower() in MULTI_CHOICE_DATASETS
        args.multi_token = multi_token

        finetune(train_valid_datasets_provider, model_provider,
                 forward_step=finetune_forward_step,
                 end_of_epoch_callback_provider=metrics_func_provider,
                 build_data_loader_fn=build_data_loader)
