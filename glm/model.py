import torch
from torch.nn import Module
from glm import build_mask_matrix
from megatron.mpu import vocab_parallel_cross_entropy


class GLMForDownstream(torch.nn.Module):
    def __init__(self, model=None, **kwargs):
        super().__init__()
        self.model = model

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = self.model.state_dict(destination, prefix, keep_vars)
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        return self.model.named_parameters(prefix=prefix, recurse=recurse)

    def disable_untrainable_params(self):
        self.model.disable_untrainable_params()

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.model.set_input_tensor(input_tensor)


class GLMForMultiTokenCloze(GLMForDownstream):
    def __init__(self, model, take_softmax=False, length_penalty=0.0):
        super().__init__(model)
        self.take_softmax = take_softmax
        self.length_penalty = length_penalty

    def forward(self, input_ids, position_ids, attention_mask, target_ids=None, logit_mask=None, prompt_pos=None):
        if target_ids == None:
            return self.model(input_ids, position_ids, attention_mask)
        num_choices = None
        if len(input_ids.shape) == 3:
            batch_size, num_choices = input_ids.shape[:2]
            input_ids = input_ids.reshape(-1, input_ids.size(-1))
            attention_mask = attention_mask.reshape(-1, *attention_mask.size()[2:])
            position_ids = position_ids.reshape(-1, *position_ids.size()[2:])
            target_ids = target_ids.reshape(-1, target_ids.size(-1))
            logit_mask = logit_mask.reshape(-1, logit_mask.size(-1))
        outputs = self.model(input_ids, position_ids, build_mask_matrix(attention_mask,
                                input_ids.size(0), input_ids.size(1)).to(torch.bool))
        logits = -vocab_parallel_cross_entropy(outputs, target_ids)
        logits = (logits * logit_mask).sum(dim=1)
        if self.length_penalty > 0.0:
            logits = logits / logit_mask.sum(dim=1) ** self.length_penalty
        if num_choices is not None:
            logits = logits.view(-1, num_choices)
        return logits


class GLMForMultiTokenClozeFast(GLMForDownstream):
    def __init__(self, model=None, take_softmax=False, length_penalty=0.0):
        super().__init__(model)
        self.take_softmax = take_softmax
        self.length_penalty = length_penalty

    def forward(self, input_ids, position_ids, attention_mask,  target_ids, logit_mask):
        # encoder
        outputs = self.model(input_ids, position_ids, attention_mask)
        if self.take_softmax:
            outputs = torch.nn.functional.log_softmax(outputs, dim=-1)
        batch_size, seq_length, vocab_size = outputs.shape
        num_choices = target_ids.size(1)
        outputs = outputs.repeat_interleave(num_choices, dim=0)
        target_ids = target_ids.reshape(-1, target_ids.size(-1))
        logit_mask = logit_mask.reshape(-1, logit_mask.size(-1))
        logits = -vocab_parallel_cross_entropy(outputs, target_ids)
        logits = (logits * logit_mask).sum(dim=1)
        if self.length_penalty > 0.0:
            logits = logits / logit_mask.sum(dim=1) ** self.length_penalty
        if num_choices is not None:
            logits = logits.view(-1, num_choices)
        return logits


class GLMForSingleTokenCloze(GLMForDownstream):
    def __init__(self, model, take_softmax=False):
        super().__init__(model)
        self.take_softmax = take_softmax

    def forward(self, input_ids, position_ids, attention_mask, target_ids=None, logit_mask=None, prompt_pos=None):
        if target_ids is None:
            return self.model(input_ids, position_ids, attention_mask)
        assert len(input_ids.shape) == 2
        # print(f"input_ids: {input_ids.shape}, position_ids: {position_ids.shape}, attention_mask: {attention_mask.shape}")
        # from megatron import get_tokenizer
        # tokenizer = get_tokenizer()
        # print(f"inputs_ids: { ' '.join([tokenizer.IdToToken(t.item()) for t in input_ids[0]])}")
        # print(f"position_ids: {position_ids[0]}")
        # print(f"attention_mask: {attention_mask[0]}")
        # # print("target_ids:", ' ' target_ids[0])
        # exit(0)
        outputs = self.model(input_ids, position_ids, build_mask_matrix(attention_mask,
                                input_ids.size(0), input_ids.size(1)).to(torch.bool))
        batch_size, vocab_size, num_choices = outputs.size(0), outputs.size(-1), target_ids.size(1)
        batch_ids = torch.arange(batch_size, dtype=attention_mask.dtype, device=attention_mask.device)
        target_logits = outputs[batch_ids, attention_mask]
        target_logits = target_logits.repeat(1, target_ids.size(1)).reshape(batch_size, num_choices, vocab_size)
        output = -vocab_parallel_cross_entropy(target_logits, target_ids)
        return (output, target_logits)

