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
from transformers.utils.import_utils import define_import_structure


if TYPE_CHECKING:
    from transformers.models.moderngpt2.configuration_moderngpt2 import *
    from transformers.models.moderngpt2.modeling_flax_moderngpt2 import *
    from transformers.models.moderngpt2.modeling_moderngpt2 import *
    from transformers.models.moderngpt2.modeling_tf_moderngpt2 import *
    # MODIFIED: For renamed tokenizers
    from transformers.models.moderngpt2.tokenization_moderngpt2 import ModernGPT2Tokenizer
    from transformers.models.moderngpt2.tokenization_moderngpt2_fast import ModernGPT2TokenizerFast
    # Original GPT2 tokenizers, if they are to co-exist or be deprecated later
    # For now, keeping them as per the TODOs, assuming they are distinct or placeholders
    from transformers.models.moderngpt2.tokenization_gpt2 import *
    from transformers.models.moderngpt2.tokenization_gpt2_fast import *
    from transformers.models.moderngpt2.tokenization_gpt2_tf import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
