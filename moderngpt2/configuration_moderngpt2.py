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
"""OpenAI ModernGPT2 configuration"""

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, List, Optional

from transformers import PreTrainedTokenizer, TensorType, is_torch_available
from transformers.configuration_utils import PretrainedConfig
from transformers.onnx import OnnxConfigWithPast, PatchingSpec
from transformers.utils import logging


logger = logging.get_logger(__name__)


class ModernGPT2Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`ModernGPT2Model`] or a [`TFModernGPT2Model`]. It is used to
    instantiate a ModernGPT2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the ModernGPT2
    [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the ModernGPT2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ModernGPT2Model`] or [`TFModernGPT2Model`].
        n_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*):
            Number of key and value heads for Grouped Query Attention (GQA). If None, defaults to `n_head`.
            Gemma2: Added for GQA.
        head_dim (`int`, *optional*):
            Dimensionality of the attention head. If None, defaults to `n_embd // n_head`.
            Gemma2: Added for explicit head dimension control.
        intermediate_size (`int`, *optional*):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd.
            Gemma2: Replaces `n_inner`.
        activation_function (`str`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the encoder and pooler.
            Gemma2: Changed default to "gelu_pytorch_tanh".
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        summary_type (`string`, *optional*, defaults to `"cls_index"`):
            Argument used when doing sequence summary, used in the models [`ModernGPT2DoubleHeadsModel`] and
            [`TFModernGPT2DoubleHeadsModel`].

            Has to be one of the following options:

                - `"last"`: Take the last token hidden state (like XLNet).
                - `"first"`: Take the first token hidden state (like BERT).
                - `"mean"`: Take the mean of all tokens hidden states.
                - `"cls_index"`: Supply a Tensor of classification token position (like ModernGPT2/GPT-2).
                - `"attn"`: Not implemented now, use multi-head attention.
        summary_use_proj (`bool`, *optional*, defaults to `True`):
            Argument used when doing sequence summary, used in the models [`ModernGPT2DoubleHeadsModel`] and
            [`TFModernGPT2DoubleHeadsModel`].

            Whether or not to add a projection after the vector extraction.
        summary_activation (`str`, *optional*):
            Argument used when doing sequence summary. Used in for the multiple choice head in
            [`ModernGPT2DoubleHeadsModel`].

            Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
        summary_proj_to_labels (`bool`, *optional*, defaults to `True`):
            Argument used when doing sequence summary, used in the models [`ModernGPT2DoubleHeadsModel`] and
            [`TFModernGPT2DoubleHeadsModel`].

            Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
        summary_first_dropout (`float`, *optional*, defaults to 0.1):
            Argument used when doing sequence summary, used in the models [`ModernGPT2DoubleHeadsModel`] and
            [`TFModernGPT2DoubleHeadsModel`].

            The dropout ratio to be used after the projection and activation.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the RMSNorm layers.
            Gemma2: Added.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period for RoPE.
            Gemma2: Added.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for RoPE.
            Gemma2: Added.
        query_pre_attn_scalar (`float`, *optional*):
            Scalar factor to scale queries before attention. If None, calculated as `head_dim**-0.5`.
            Gemma2: Added. Supersedes `scale_attn_weights`.
        attn_logit_softcapping (`float`, *optional*):
            If set, softcap attention logits to this value.
            Gemma2: Added.
        final_logit_softcapping (`float`, *optional*):
            If set, softcap final logits to this value.
            Gemma2: Added.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in QKV projections.
            Gemma2: Added.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        bos_token_id (`int`, *optional*, defaults to 50256):
            Id of the beginning of sentence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 50256):
            Id of the end of sentence token in the vocabulary.

    Example:

    ```python
    >>> from transformers import ModernGPT2Config, ModernGPT2Model

    >>> # Initializing a ModernGPT2 configuration
    >>> configuration = ModernGPT2Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = ModernGPT2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "moderngpt2"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
        "intermediate_size": "intermediate_size", # Ensure this is mapped if n_inner was used before
    }

    MODEL_SIZE_CONFIGS = {
        "small": {"n_layer": 12, "n_embd": 768, "n_head": 12},
        "medium": {"n_layer": 24, "n_embd": 1024, "n_head": 16},
        "large": {"n_layer": 36, "n_embd": 1280, "n_head": 20},
        "xl": {"n_layer": 48, "n_embd": 1600, "n_head": 25},
    }

    def __init__(
        self,
        model_size_name: str = None, # New argument for model size
        vocab_size=50257,
        n_positions=1024,
        n_embd=768, # Default to small if no model_size_name
        n_layer=12, # Default to small
        n_head=12,  # Default to small
        num_key_value_heads=None,
        head_dim=None,
        intermediate_size=None,
        activation_function="gelu_pytorch_tanh",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5, # Note: Gemma2 uses rms_norm_eps for its main norm layers
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        # scale_attn_weights=True, # Gemma2: Superseded by query_pre_attn_scalar
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        # scale_attn_by_inverse_layer_idx=False, # Gemma2: Removed
        # reorder_and_upcast_attn=False, # Gemma2: Removed
        rms_norm_eps=1e-6,  # Gemma2: Added
        rope_theta=10000.0,  # Gemma2: Added
        rope_scaling=None,  # Gemma2: Added
        query_pre_attn_scalar=None,  # Gemma2: Added
        attn_logit_softcapping=None,  # Gemma2: Added
        final_logit_softcapping=None,
        attention_bias=False,
        **kwargs,
    ):
        if model_size_name is not None:
            if model_size_name not in self.MODEL_SIZE_CONFIGS:
                raise ValueError(f"Unknown model_size_name: {model_size_name}")
            size_cfg = self.MODEL_SIZE_CONFIGS[model_size_name]
            n_layer = size_cfg["n_layer"]
            n_embd = size_cfg["n_embd"]
            n_head = size_cfg["n_head"]
            # Pop these from kwargs if they were also passed explicitly, to avoid TypeErrors with super().__init__
            kwargs.pop("n_layer", None)
            kwargs.pop("n_embd", None)
            kwargs.pop("hidden_size", None) # alias for n_embd
            kwargs.pop("n_head", None)
            kwargs.pop("num_attention_heads", None) # alias for n_head

        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head

        # Derived parameters based on potential model_size_name or explicit args
        calculated_intermediate_size = 4 * self.n_embd
        calculated_head_dim = self.n_embd // self.n_head
        calculated_num_key_value_heads = self.n_head

        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else calculated_num_key_value_heads
        self.head_dim = head_dim if head_dim is not None else calculated_head_dim
        self.intermediate_size = intermediate_size if intermediate_size is not None else calculated_intermediate_size

        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon # Original LayerNorm eps
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        # self.scale_attn_weights = scale_attn_weights # Gemma2: Superseded by query_pre_attn_scalar
        self.use_cache = use_cache
        # self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx # Gemma2: Removed
        # self.reorder_and_upcast_attn = reorder_and_upcast_attn # Gemma2: Removed

        self.rms_norm_eps = rms_norm_eps  # Gemma2: Added
        self.rope_theta = rope_theta  # Gemma2: Added
        self.rope_scaling = rope_scaling  # Gemma2: Added
        self.query_pre_attn_scalar = query_pre_attn_scalar # Gemma2: Added
        self.attn_logit_softcapping = attn_logit_softcapping  # Gemma2: Added
        self.final_logit_softcapping = final_logit_softcapping  # Gemma2: Added
        self.attention_bias = attention_bias  # Gemma2: Added

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # Gemma2: Map n_inner to intermediate_size for compatibility if n_inner is passed via kwargs
        # This should be handled by attribute_map or by ensuring intermediate_size is preferred
        if "n_inner" in kwargs and intermediate_size is None: # Only if intermediate_size wasn't set
            self.intermediate_size = kwargs.pop("n_inner")
        elif "n_inner" in kwargs: # if intermediate_size was set, pop n_inner to avoid conflict
            kwargs.pop("n_inner")
        
        # Remove loss_type if it exists in kwargs (not used by this model)
        if "loss_type" in kwargs:
            kwargs.pop("loss_type")

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


class ModernGPT2OnnxConfig(OnnxConfigWithPast):
    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "default",
        patching_specs: Optional[List[PatchingSpec]] = None,
        use_past: bool = False,
    ):
        super().__init__(config, task=task, patching_specs=patching_specs, use_past=use_past)
        if not getattr(self._config, "pad_token_id", None):
            # TODO: how to do that better?
            self._config.pad_token_id = 0

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        common_inputs = OrderedDict({"input_ids": {0: "batch", 1: "sequence"}})
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")
            common_inputs["attention_mask"] = {0: "batch", 1: "past_sequence + sequence"}
        else:
            common_inputs["attention_mask"] = {0: "batch", 1: "sequence"}

        return common_inputs

    @property
    def num_layers(self) -> int:
        return self._config.n_layer

    @property
    def num_attention_heads(self) -> int:
        return self._config.n_head

    @property
    def hidden_size(self) -> int: # Gemma2: Added for ONNX export
        return self._config.n_embd

    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        common_inputs = super(OnnxConfigWithPast, self).generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )

        # We need to order the input in the way they appears in the forward()
        ordered_inputs = OrderedDict({"input_ids": common_inputs["input_ids"]})

        # Need to add the past_keys
        if self.use_past:
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch

                batch, seqlen = common_inputs["input_ids"].shape
                # Not using the same length for past_key_values
                past_key_values_length = seqlen + 2
                past_shape = (
                    batch,
                    self.num_attention_heads, # TODO: This should be num_key_value_heads for K,V cache in GQA
                    past_key_values_length,
                    self._config.head_dim, # Gemma2: Uses head_dim
                )
                ordered_inputs["past_key_values"] = [ # Gemma2: Cache structure might differ with GQA
                    (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(self.num_layers)
                ]

        ordered_inputs["attention_mask"] = common_inputs["attention_mask"]
        if self.use_past:
            mask_dtype = ordered_inputs["attention_mask"].dtype
            ordered_inputs["attention_mask"] = torch.cat(
                [ordered_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
            )

        return ordered_inputs

    @property
    def default_onnx_opset(self) -> int:
        return 13


__all__ = ["ModernGPT2Config", "ModernGPT2OnnxConfig"]
