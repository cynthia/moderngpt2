# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch OpenAI ModernGPT2 model."""

import math
import os
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN, get_activation
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_attention_mask_for_sdpa
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    logging,
)
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from .configuration_moderngpt2 import ModernGPT2Config


logger = logging.get_logger(__name__)

# Gemma2: Added helper functions and classes
# Copied from transformers.models.gemma2.modeling_gemma2.Gemma2RMSNorm
class ModernGPT2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

# Copied from transformers.models.gemma2.modeling_gemma2.Gemma2RotaryEmbedding
class ModernGPT2RotaryEmbedding(nn.Module):
    def __init__(self, config: ModernGPT2Config, cache_dtype=torch.float32): # MODIFIED: Added config type
        super().__init__()
        self.dim = config.head_dim
        self.max_position_embeddings = config.n_positions
        self.rope_theta = config.rope_theta
        self.cache_dtype = cache_dtype
        self.scaling_factor = getattr(config, "rope_scaling_factor", None)

        self._set_cos_sin_cache(seq_len=self.max_position_embeddings, device=None, dtype=self.cache_dtype)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        inv_freq = torch.arange(0, self.dim, 2, device=device, dtype=self.cache_dtype)
        inv_freq = 1.0 / (self.rope_theta ** (inv_freq / self.dim))

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.cache_dtype)

        if self.scaling_factor is not None:
            t = t / self.scaling_factor

        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype=dtype, device=device), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype=dtype, device=device), persistent=False)

    def forward(self, x, position_ids):
        # Dynamically extend position embeddings if needed
        max_pos = position_ids.max()
        if max_pos >= self.max_seq_len_cached:
            # Extend the cached embeddings to accommodate longer sequences
            self._set_cos_sin_cache(seq_len=max_pos + 1, device=position_ids.device, dtype=self.cache_dtype)
            logger.warning(
                f"Extended position embeddings from {self.max_position_embeddings} to {max_pos + 1}. "
                f"Consider setting n_positions={max_pos + 1} in your config to avoid dynamic extension."
            )
        
        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]
        return cos, sin

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class ModernGPT2MLP(nn.Module):
    def __init__(self, config: ModernGPT2Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.n_embd
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.activation_function]

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = self.act_fn(gate)
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs

def load_tf_weights_in_moderngpt2(model, config, moderngpt2_checkpoint_path):
    # ... (tf weight loading code - keeping it as is for now)
    pass


class ModernGPT2Attention(nn.Module):
    def __init__(self, config: ModernGPT2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_bias = config.attention_bias
        self.query_pre_attn_scalar = config.query_pre_attn_scalar
        self.attn_logit_softcapping = config.attn_logit_softcapping

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=self.attention_bias)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if self.query_pre_attn_scalar is not None:
            query_states *= self.query_pre_attn_scalar
        elif self.config.query_pre_attn_scalar is None :
             query_states *= (self.head_dim**-0.5)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        if self.attn_logit_softcapping is not None:
            attn_weights = torch.tanh(attn_weights / self.attn_logit_softcapping) * self.attn_logit_softcapping

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_for_output = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights_for_output, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class ModernGPT2Block(nn.Module):
    def __init__(self, config: ModernGPT2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.n_embd
        self.config = config
        self.layer_idx = layer_idx

        self.input_layernorm = ModernGPT2RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.attn = ModernGPT2Attention(config, layer_idx)
        self.post_attention_layernorm = ModernGPT2RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        self.pre_feedforward_layernorm = ModernGPT2RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.mlp = ModernGPT2MLP(config)
        self.post_feedforward_layernorm = ModernGPT2RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[torch.Tensor], Optional[Cache]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_outputs = self.attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = attn_outputs[0]
        self_attn_weights = attn_outputs[1]
        present_key_value = attn_outputs[2] if use_cache else None

        hidden_states = self.resid_dropout(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.resid_dropout(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class ModernGPT2SequenceSummary(nn.Module):
    # ... (Sequence Summary code - keeping as is, with n_embd fix)
    def __init__(self, config: ModernGPT2Config):
        super().__init__()

        self.summary_type = getattr(config, "summary_type", "last")
        if self.summary_type == "attn":
            raise NotImplementedError

        self.summary = nn.Identity()
        if hasattr(config, "summary_use_proj") and config.summary_use_proj:
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.n_embd
            self.summary = nn.Linear(config.n_embd, num_classes)

        activation_string = getattr(config, "summary_activation", None)
        self.activation: Callable = get_activation(activation_string) if activation_string else nn.Identity()

        self.first_dropout = nn.Identity()
        if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)

        self.last_dropout = nn.Identity()
        if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

    def forward(
        self, hidden_states: torch.FloatTensor, cls_index: Optional[torch.LongTensor] = None
    ) -> torch.FloatTensor:
        if self.summary_type == "last":
            output = hidden_states[:, -1]
        elif self.summary_type == "first":
            output = hidden_states[:, 0]
        elif self.summary_type == "mean":
            output = hidden_states.mean(dim=1)
        elif self.summary_type == "cls_index":
            if cls_index is None:
                cls_index = torch.full_like(
                    hidden_states[..., :1, :],
                    hidden_states.shape[-2] - 1,
                    dtype=torch.long,
                )
            else:
                cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
                cls_index = cls_index.expand((-1,) * (cls_index.dim() - 1) + (hidden_states.size(-1),))
            output = hidden_states.gather(-2, cls_index).squeeze(-2)
        elif self.summary_type == "attn":
            raise NotImplementedError

        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)
        return output

class ModernGPT2PreTrainedModel(PreTrainedModel):
    config_class = ModernGPT2Config
    load_tf_weights = load_tf_weights_in_moderngpt2
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["ModernGPT2Block"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_attention_backend = True
    _supports_cache_class = True
    _supports_static_cache = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, ModernGPT2RMSNorm):
             module.weight.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        for name, p in module.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"): # Gemma2 MLP and Attention output layers
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))


@dataclass
class ModernGPT2DoubleHeadsModelOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.

    Args:
        loss (`Optional[torch.FloatTensor]` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        mc_loss (`Optional[torch.FloatTensor]` of shape `(1,)`, *optional*, returned when `mc_labels` is provided):
            Multiple choice classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_logits (`torch.FloatTensor` of shape `(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        past_key_values (`Optional[Cache]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used to speed up
            sequential decoding.
        hidden_states (`Optional[Tuple[torch.FloatTensor]]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`Optional[Tuple[torch.FloatTensor]]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[torch.FloatTensor] = None
    mc_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    mc_logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None # MODIFIED: Gemma2 returns Cache object
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

# PARALLELIZE_DOCSTRING and DEPARALLELIZE_DOCSTRING remain the same

PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.

    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.

    Args:
        device_map (`Dict[int, list]`, *optional*):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the gpt2 models have the
            following number of attention modules:

                - openai-community/gpt2: 12
                - openai-community/gpt2-medium: 24
                - openai-community/gpt2-large: 36
                - openai-community/gpt2-xl: 48

    Example:

    ```python
    # Here is an example of a device map on a machine with 4 GPUs using moderngpt2-xl, which has a total of 48 attention modules:
    model = ModernGPT2LMHeadModel.from_pretrained("openai-community/gpt2-xl") # Ensure this checkpoint exists or adapt
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8],
        1: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        2: [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
        3: [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
    }
    model.parallelize(device_map)
    ```
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.

    Example:

    ```python
    # On a 4 GPU machine with openai-community/moderngpt2-large:
    model = ModernGPT2LMHeadModel.from_pretrained("openai-community/gpt2-large") # Ensure this checkpoint exists or adapt
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7],
        1: [8, 9, 10, 11, 12, 13, 14, 15],
        2: [16, 17, 18, 19, 20, 21, 22, 23],
        3: [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    }
    model.parallelize(device_map)  # Splits the model across several devices
    model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
    ```
"""

class ModernGPT2Model(ModernGPT2PreTrainedModel):
    _supports_param_buffer_assignment = False

    def __init__(self, config: ModernGPT2Config):
        super().__init__(config)
        self.config = config

        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.rotary_emb = ModernGPT2RotaryEmbedding(config)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([ModernGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = ModernGPT2RMSNorm(self.embed_dim, eps=config.rms_norm_eps)

        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
        self._attn_implementation = getattr(config, "_attn_implementation", "eager")

        self.post_init()

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # ... (parallelize code - keeping as is, wpe already removed)
        warnings.warn(
            "`ModernGPT2Model.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your"
            " model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'h.0': 0, 'h.1': 1,"
            " ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        self.wte = self.wte.to(self.first_device)
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.h[block] = self.h[block].to(cuda_device)
        self.ln_f = self.ln_f.to(self.last_device)


    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        # ... (deparallelize code - keeping as is, wpe already removed)
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()


    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        # MODIFIED: Pruning logic might need update if attention structure changed significantly for pruning
        # For now, assuming ModernGPT2Attention has a prune_heads method or this needs to be adapted/removed.
        # Gemma2Attention does not have a prune_heads method by default.
        logger.warning("Head pruning is not yet fully supported for Gemma2-style attention in ModernGPT2.")
        # for layer, heads in heads_to_prune.items():
        #     if hasattr(self.h[layer], "attn"):
        #          self.h[layer].attn.prune_heads(heads)
        #     else:
        #         logger.warning(f"Layer {layer} does not have an attention module to prune.")


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Tuple[Tuple[torch.Tensor]], Cache]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        q_len = input_shape[-1]

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        if cache_position is None:
            if past_key_values is not None:
                past_seen_tokens = past_key_values.get_seq_length() if isinstance(past_key_values, Cache) else past_key_values[0][0].shape[2]
                cache_position = torch.arange(past_seen_tokens, past_seen_tokens + q_len, device=device)
            else:
                cache_position = torch.arange(q_len, device=device)

        if position_ids is None:
             position_ids = cache_position.unsqueeze(0)

        # Debug logging for position_ids
        if position_ids.max() >= self.rotary_emb.max_position_embeddings:
            logger.warning(
                f"Position ids exceed max_position_embeddings! "
                f"position_ids shape: {position_ids.shape}, "
                f"range: [{position_ids.min().item()}, {position_ids.max().item()}], "
                f"max_position_embeddings: {self.rotary_emb.max_position_embeddings}, "
                f"cache_position shape: {cache_position.shape}, "
                f"past_seen_tokens: {past_seen_tokens if 'past_seen_tokens' in locals() else 'N/A'}"
            )

        cos, sin = self.rotary_emb(inputs_embeds, position_ids)
        position_embeddings = (cos,sin)
        hidden_states = inputs_embeds * (self.config.n_embd**0.5)

        # Determine past_key_values_length
        past_key_values_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_key_values_length = past_key_values.get_seq_length()
            else: # Assuming tuple cache
                past_key_values_length = past_key_values[0][0].shape[2]

        # Standard Hugging Face logic for preparing 4D attention mask for causal decoder
        # The input `attention_mask` to this function is the 2D padding mask.
        # We are creating `model_attention_mask` for the transformer blocks.
        if input_shape[-1] > 1: # Sequence length > 1
            causal_mask = AttentionMaskConverter._make_causal_mask(
                input_shape,
                hidden_states.dtype,
                device=hidden_states.device, # Use hidden_states.device for target device
                past_key_values_length=past_key_values_length,
            )
        else:
            causal_mask = None

        if attention_mask is not None: # If a 2D padding mask is provided
            expanded_padding_mask = AttentionMaskConverter._expand_mask(
                attention_mask, dtype=hidden_states.dtype, tgt_len=input_shape[-1]
            ).to(hidden_states.device) # Use hidden_states.device

            if causal_mask is not None:
                model_attention_mask = causal_mask + expanded_padding_mask
            else:
                model_attention_mask = expanded_padding_mask
        else: # No 2D padding mask, just use the causal mask
            model_attention_mask = causal_mask

        attention_mask = model_attention_mask

        head_mask = self.get_head_mask(head_mask, self.config.n_layer)
        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)

        all_self_attentions = () if output_attentions else None
        all_cross_attentions = None
        all_hidden_states = () if output_hidden_states else None

        next_decoder_cache = None
        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_past = past_key_values

            if self.gradient_checkpointing and self.training:
                 outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    layer_past,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                outputs = block(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_value=layer_past,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = outputs[0]
            if use_cache:
                if isinstance(outputs[-1], Cache):
                    past_key_values = outputs[-1]
                elif isinstance(outputs, tuple) and len(outputs) > 2 and isinstance(outputs[2], tuple) :
                     if next_decoder_cache is None: next_decoder_cache = ()
                     next_decoder_cache = next_decoder_cache + (outputs[2],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if use_cache:
            if hasattr(past_key_values, "to_legacy_cache") and callable(getattr(past_key_values, "to_legacy_cache")):
                 past_key_values = past_key_values.to_legacy_cache()
            elif next_decoder_cache is not None:
                past_key_values = next_decoder_cache

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, past_key_values, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

    def _update_causal_mask( # Kept for reference, might be adapted or removed
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # Default to Llama's _update_causal_mask logic if not using specific attn implementations
        # that handle it internally (like FlashAttention with is_causal=True)
        # This is a placeholder and might need specific adjustments for Gemma2's needs if different.
        # The core idea is to create a 4D mask for attention.
        # See LlamaModel._update_causal_mask for a more complete example.

        # For simplicity, if a 2D mask is provided, assume it's for padding and combine with causal.
        # If no mask, generate a causal one. This part is highly dependent on expected input.
        # The current ModernGPT2Attention expects a 4D mask.

        input_ids_shape = input_tensor.shape[:-1] # bsz, seq_len
        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        # Create a 4D causal mask using AttentionMaskConverter
        # This will be of shape (bsz, 1, q_len, kv_len)
        causal_4d_mask = AttentionMaskConverter._make_causal_mask(
            input_ids_shape,
            input_tensor.dtype,
            device=input_tensor.device,
            past_key_values_length=past_key_values_length,
        )

        # If a 2D attention_mask (padding mask) is provided, expand and combine it
        if attention_mask is not None and attention_mask.ndim == 2:
            # Expand padding mask to [bsz, 1, 1, seq_len]
            padding_mask_4d = attention_mask[:, None, None, :]
            # Combine: where padding_mask_4d is 0 (masked), the result should be min_float
            # Causal mask is 0 for attend, min_float for masked.
            # Padding mask (converted to additive) is 0 for attend, min_float for masked.
            additive_padding_mask = (1.0 - padding_mask_4d) * torch.finfo(input_tensor.dtype).min

            # Ensure kv_len of causal_4d_mask matches padding_mask's seq_len for combination
            # This logic might need refinement based on how padding_mask is structured with past_key_values
            # Assuming padding_mask covers the full sequence length including past
            # For now, slice causal_4d_mask if padding_mask is shorter (e.g. only current tokens)
            # This part is tricky and depends on how attention_mask is meant to be used with past_key_values.
            # A common approach is for attention_mask to cover the *entire* sequence (past + current).

            # If causal_4d_mask's src_len (kv_len) is longer than padding_mask's src_len
            if causal_4d_mask.shape[-1] > additive_padding_mask.shape[-1]:
                 # This happens if padding_mask only covers current q_len
                 # We need to align them. Assuming padding mask refers to the full effective sequence.
                 # This part is complex and needs to match precisely how padding is handled.
                 # For now, let's assume if a 2D mask is given, it applies to the source_sequence_length
                 # which is kv_len.
                 # If attention_mask is for q_len only, it needs expansion for kv_len.
                 # If attention_mask is for kv_len already, direct combination is fine.
                 # Let's assume attention_mask is [bsz, kv_len]
                 # The following line is a simplification and might need adjustment
                 causal_mask = causal_4d_mask + additive_padding_mask[:,:,:causal_4d_mask.shape[-2],:]

            else: # If padding_mask_4d is already correctly shaped or longer (should not happen for causal)
                causal_mask = causal_4d_mask + additive_padding_mask

            causal_mask = torch.clamp(causal_mask, min=torch.finfo(input_tensor.dtype).min)

        elif attention_mask is not None and attention_mask.ndim == 4: # Already a 4D mask
            causal_mask = attention_mask
        else: # No padding mask provided, just use the causal mask
            causal_mask = causal_4d_mask

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(*args, **kwargs):
        # This method might be deprecated if _update_causal_mask is used and handles all cases.
        # Or it can be a more specific utility. For now, point to the main one.
        # return ModernGPT2Model._update_causal_mask(*args, **kwargs)
        # Keeping the original static method structure for now if it's called elsewhere.
        # This was originally copied from Llama.
        pass # Placeholder, will rely on _update_causal_mask or direct creation in forward.


class ModernGPT2LMHeadModel(ModernGPT2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = ModernGPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.model_parallel = False
        self.device_map = None
        self.post_init()

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # ... (unchanged)
        warnings.warn(
            "`ModernGPT2LMHeadModel.parallelize` is deprecated and will be removed in v5 of Transformers, you should load"
            " your model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'transformer.h.0':"
            " 0, 'transformer.h.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        # ... (unchanged)
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()


    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        if self.config.final_logit_softcapping is not None:
            lm_logits = torch.tanh(lm_logits / self.config.final_logit_softcapping) * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(lm_logits, labels, vocab_size=self.config.vocab_size) # MODIFIED: Removed **kwargs

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

# ... (Other head models: ModernGPT2DoubleHeadsModel, ModernGPT2ForSequenceClassification, etc.)
# These will also need their self.transformer calls updated similarly to ModernGPT2LMHeadModel's forward.
# For brevity, I'm showing the LMHeadModel changes. Assume similar argument passing for other head models.

class ModernGPT2DoubleHeadsModel(ModernGPT2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = ModernGPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.multiple_choice_head = ModernGPT2SequenceSummary(config)
        self.model_parallel = False
        self.device_map = None
        self.post_init()

    # ... (parallelize, deparallelize, get_output_embeddings, set_output_embeddings are similar)
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # ... (unchanged)
        warnings.warn(
            "`ModernGPT2DoubleHeadsModel.parallelize` is deprecated and will be removed in v5 of Transformers, you should"
            " load your model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your"
            " own `device_map` but it needs to be a dictionary module_name to device, so for instance"
            " {'transformer.h.0': 0, 'transformer.h.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.multiple_choice_head = self.multiple_choice_head.to(self.transformer.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        # ... (unchanged)
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.multiple_choice_head = self.multiple_choice_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        mc_token_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        mc_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ModernGPT2DoubleHeadsModelOutput]:
        """
        Args:
            input_ids (`Optional[torch.LongTensor]` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of input sequence tokens in the vocabulary.
                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.
                [What are input IDs?](../glossary#input-ids)
            past_key_values (`Optional[Cache]`, *optional*):
                Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
                `past_key_values` input) to speed up sequential decoding.
                [Standard past_key_values description]
            attention_mask (`Optional[torch.FloatTensor]` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            mc_token_ids (`Optional[torch.LongTensor]` of shape `(batch_size, num_choices)`, *optional*):
                Indices of the multiple choice classification tokens in the input sequence. This is used to
                gather the hidden states of chosen tokens for the multiple choice classification head. For example,
                if `batch_size` is 3 and `num_choices` is 4, and `mc_token_ids` is `[[0, 1, 0, 1], [2, 0, 1, 2], [1, 1, 0, 0]]`,
                this will gather the hidden states at these token indices for each example and choice.
            labels (`Optional[torch.LongTensor]` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
                `labels = input_ids`. Further note that `labels` is not used in the calculation of the loss if
                `lm_head` is not defined.
            mc_labels (`Optional[torch.LongTensor]` of shape `(batch_size,)`, *optional*):
                Labels for the multiple choice classification task. Provides the index of the correct choice for
                each example in the batch.
            use_cache (`Optional[bool]`, *optional*):
                Whether or not the model should return the last key/values attentions (not used by all models).
                [Standard use_cache description]
            output_attentions (`Optional[bool]`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`Optional[bool]`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`Optional[bool]`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)
        if self.config.final_logit_softcapping is not None:
            lm_logits = torch.tanh(lm_logits / self.config.final_logit_softcapping) * self.config.final_logit_softcapping
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

        mc_loss = None
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            mc_loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
        lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits, mc_logits) + transformer_outputs[1:]
            if mc_loss is not None:
                output = (mc_loss,) + output
            return ((lm_loss,) + output) if lm_loss is not None else output

        return ModernGPT2DoubleHeadsModelOutput(
            loss=lm_loss,
            mc_loss=mc_loss,
            logits=lm_logits,
            mc_logits=mc_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(past_key_values: Cache, beam_idx: torch.Tensor) -> Cache:
        if isinstance(past_key_values, Cache):
            return past_key_values.reorder_cache(beam_idx)
        else:
            # Fallback for tuple cache, though ideally Cache object should be used
            return tuple(
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
                for layer_past in past_key_values
            )

class ModernGPT2ForSequenceClassification(ModernGPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = ModernGPT2Model(config)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)
        self.model_parallel = False
        self.device_map = None
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # ... (rest of the method from original, with hidden_states = transformer_outputs[0])
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1: self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int): self.config.problem_type = "single_label_classification"
                else: self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1: loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else: loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutputWithPast(loss=loss, logits=pooled_logits, past_key_values=transformer_outputs.past_key_values, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

class ModernGPT2ForTokenClassification(ModernGPT2PreTrainedModel):
    def __init__(self, config):
        # ... (init as before, ensure self.transformer = ModernGPT2Model(config))
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = ModernGPT2Model(config)
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.n_embd, config.num_labels)
        self.model_parallel = False
        self.device_map = None
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + transformer_outputs[1:] # MODIFIED: transformer_outputs[1:] because past_key_values is now a single object
            return ((loss,) + output) if loss is not None else output
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

class ModernGPT2ForQuestionAnswering(ModernGPT2PreTrainedModel):
    def __init__(self, config):
        # ... (init as before, ensure self.transformer = ModernGPT2Model(config))
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = ModernGPT2Model(config)
        self.qa_outputs = nn.Linear(config.n_embd, 2)
        self.model_parallel = False
        self.device_map = None
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # MODIFIED: Removed past_key_values, cache_position, use_cache as QA doesn't typically use them for training
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer( # MODIFIED: Pass relevant args
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # past_key_values, use_cache, cache_position are not typically used in QA forward pass
        )
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1: start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1: end_positions = end_positions.squeeze(-1).to(end_logits.device)
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:] # MODIFIED: outputs[1:] as past_key_values might not be there
            return ((total_loss,) + output) if total_loss is not None else output
        return QuestionAnsweringModelOutput(loss=total_loss, start_logits=start_logits, end_logits=end_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

__all__ = [
    "ModernGPT2DoubleHeadsModel",
    "ModernGPT2ForQuestionAnswering",
    "ModernGPT2ForSequenceClassification",
    "ModernGPT2ForTokenClassification",
    "ModernGPT2LMHeadModel",
    "ModernGPT2Model",
    "ModernGPT2PreTrainedModel",
    "load_tf_weights_in_moderngpt2",
]
