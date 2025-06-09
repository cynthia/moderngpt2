# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from typing import TYPE_CHECKING

from transformers.utils import _LazyModule


_import_structure = {
    "configuration_moderngpt2": ["ModernGPT2Config", "ModernGPT2OnnxConfig"],
    "modeling_moderngpt2": [
        "ModernGPT2Model",
        "ModernGPT2LMHeadModel",
        "ModernGPT2DoubleHeadsModel",
        "ModernGPT2ForSequenceClassification",
        "ModernGPT2ForTokenClassification",
        "ModernGPT2ForQuestionAnswering",
        "ModernGPT2PreTrainedModel",
        "load_tf_weights_in_moderngpt2",
    ],
    "modeling_flax_moderngpt2": [
        "FlaxModernGPT2Model",
        "FlaxModernGPT2LMHeadModel",
        "FlaxModernGPT2PreTrainedModel",
    ],
    "modeling_tf_moderngpt2": [
        "TFModernGPT2Model",
        "TFModernGPT2LMHeadModel",
        "TFModernGPT2DoubleHeadsModel",
        "TFModernGPT2ForSequenceClassification",
        "TFGPT2PreTrainedModel",
        "TFModernGPT2MainLayer",
    ],
    "tokenization_moderngpt2": ["ModernGPT2Tokenizer"],
    "tokenization_moderngpt2_fast": ["ModernGPT2TokenizerFast"],
}

if TYPE_CHECKING:
    from .configuration_moderngpt2 import ModernGPT2Config, ModernGPT2OnnxConfig
    from .modeling_moderngpt2 import (
        ModernGPT2Model,
        ModernGPT2LMHeadModel,
        ModernGPT2DoubleHeadsModel,
        ModernGPT2ForSequenceClassification,
        ModernGPT2ForTokenClassification,
        ModernGPT2ForQuestionAnswering,
        ModernGPT2PreTrainedModel,
        load_tf_weights_in_moderngpt2,
    )
    from .modeling_flax_moderngpt2 import (
        FlaxModernGPT2Model,
        FlaxModernGPT2LMHeadModel,
        FlaxModernGPT2PreTrainedModel,
    )
    from .modeling_tf_moderngpt2 import (
        TFModernGPT2Model,
        TFModernGPT2LMHeadModel,
        TFModernGPT2DoubleHeadsModel,
        TFModernGPT2ForSequenceClassification,
        TFGPT2PreTrainedModel,
        TFModernGPT2MainLayer,
    )
    from .tokenization_moderngpt2 import ModernGPT2Tokenizer
    from .tokenization_moderngpt2_fast import ModernGPT2TokenizerFast
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
