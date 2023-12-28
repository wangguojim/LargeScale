"""Zero-shot datasets."""

import os
import json

import numpy as np
import torch

from megatron import get_tokenizer, get_args
from scipy.linalg import block_diag


def build_dataset(path):
    """Helper function to select and build dataset."""
    return ZeroShotDataset(path)


def pad_batch(tokens, targets, position_ids, attention_mask, max_seq_length=None):
    assert len(tokens) <= max_seq_length
    attention_mask.append(np.zeros((max_seq_length - len(tokens), max_seq_length - len(tokens)), dtype=np.long))
    tokens = np.concatenate(
        (
            tokens,
            np.zeros(max_seq_length - len(tokens), dtype=np.long),
        )
    )
    targets = np.concatenate(
        (
            targets,
            np.zeros(max_seq_length - len(targets), dtype=np.long),
        )
    )
    position_ids = np.concatenate(
        (
            position_ids,
            np.zeros(
                max_seq_length - len(position_ids), dtype=np.long
            ),
        )
    )
    return tokens, targets, position_ids, attention_mask


class ZeroShotDataset(torch.utils.data.Dataset):
    """
    Jsonlines of {
        "text": context
        "choices": [choice_id1,...], if not None, len(target) == 1
        "label": If generation task -1, else [0, len(choices))
    }
    If [MASK] not in context, will append [MASK] after text
    """
    def __init__(self, path, use_gmask=False, max_seq_length=2048):
        args = get_args()

        self.path = path
        self.use_gmask = use_gmask
        self.max_seq_length = max_seq_length
        self.data = []

        tokenizers = get_tokenizer()
        self.tmask_id = tokenizers.get_special_token('MASK')
        self.gmask_id = tokenizers.get_special_token('gMASK')
        self.mask_id = self.gmask_id if self.use_gmask else self.tmask_id
        self.sop_id = tokenizers.get_special_token('sop')
        self.eop_id = tokenizers.get_special_token('eop')
        self.dtype = np.long

        self.is_single_token = True
        self.unified_multitask_encoding = args.unified_multitask_encoding

        with open(os.path.join(path), 'r') as file:
            for line in file:
                item = json.loads(line)
                text, choices, label = item['inputs'], item['choices'], item['label']
                # choice_ids = item.get('choice_ids', [20167, 20333, 20251, 20416])

                tgt_seq_length = sum([len(choice) for choice in choices])
                if tgt_seq_length != len(choices):
                    self.is_single_token = False
                else:
                    # For single token, we only insert one [sop]
                    tgt_seq_length = 1

                assert tgt_seq_length < max_seq_length
                if len(text) + tgt_seq_length + 2 > max_seq_length:
                    text_length = max_seq_length - tgt_seq_length - 2
                    text = text[len(text) - text_length:len(text)]

                assert not (self.tmask_id in text and self.unified_multitask_encoding), \
                    "Unified multitask encoding don't support blank filling"

                self.data.append({
                    'text': text,
                    'choices': choices,
                    'label': label,
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text, choices, label = item['text'], item['choices'], item['label']
        token = np.array(text, dtype=self.dtype)
        target = np.array(text, dtype=self.dtype)
        position_id = np.arange(len(text), dtype=self.dtype)
        choice_target_id = []

        if self.tmask_id not in text:
            mask_position = len(token)
            token = np.concatenate((token, [self.mask_id]))
            target = np.concatenate((target, [self.mask_id]))
            position_id = np.concatenate((position_id, [mask_position]))
        else:
            mask_position = text.index(self.tmask_id)

        division = len(token)
        attention_mask = [np.ones((len(token), len(token)), dtype=self.dtype)]

        for choice in choices:
            position_id = np.concatenate((
                position_id,
                [mask_position] * len(choice) if not self.unified_multitask_encoding else
                np.arange(mask_position, mask_position + len(choice), dtype=self.dtype)
            ))
            choice_target_id.append(np.arange(len(token), len(token) + len(choice), dtype=self.dtype))
            attention_mask.append(np.tril(np.ones((len(choice), len(choice)), dtype=np.long)))
            token = np.concatenate((token, [self.sop_id], choice[:-1]))
            target = np.concatenate((target, choice))

            if self.is_single_token:
                break


        # pad batch
        seq_length = len(token)
        TILE = 32
        token, target, position_id, attention_mask = pad_batch(
            token, target, position_id, attention_mask, ((seq_length + TILE - 1) // TILE) * TILE)

        attention_mask = block_diag(*attention_mask)
        attention_mask[:seq_length, :division] = 1

        item = {
            'tokens': token,
            'targets': target,
            'position_ids': position_id,
            'attention_mask': attention_mask < 0.5,
            'choice_target_ids': choice_target_id,
            'is_single_token': self.is_single_token,
        }
        if self.is_single_token:
            item['choices'] = np.array(choices, dtype=self.dtype).squeeze()
        return item
