import warnings
from typing import List, Optional, Tuple, Union

import torch.distributed as dist
import torch.utils.checkpoint
import transformers
from internvl.conversation import get_conv_template
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.phi3.modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_internvl_chat import InternVLChatConfig
from .modeling_intern_vit import InternVisionModel, has_flash_attn
from.modeling_internvl_chat import InternVLChatModel

logger = logging.get_logger(__name__)

class InternVLSequenceClassificationModel(InternVLChatModel):
    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, add_classify_head = 'last_hidden_states', pooling = 'last'):
        super().__init__(config, vision_model, language_model)
        self.add_classify_head = add_classify_head # last_hidden_states, middle_hidden_states
        self.pooling = pooling

        self.classify_attention = nn.Linear(self.hidden_size, 1) 
        self.classifier_heads = nn.Linear(self.hidden_size, 2, bias=False)

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask = None,
            position_ids = None,
            image_flags = None,
            past_key_values = None,
            labels = None,
            label_class = None,
            use_cache = None,
            output_attentions = None,
            output_hidden_states = None,
            return_dict = None,
    ):
        outputs = super().forward(
            pixel_values = pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_flags=image_flags,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict)


        # two ways of add classification heads
        if self.add_classify_head == 'last_hidden_states':
            pooled_output = outputs.hidden_states[-1]
        elif self.add_classify_head == 'middle_hidden_states':
            pooled_output = outputs.hidden_states[int(len(outputs.hidden_states)/2)]
        else:
            raise ValueError("Unsupported add classify head place")

        pooled_output = self.classifier_heads(pooled_output)

        if self.pooling == "cls":
            pooled_output = pooled_output[:, 0, :] 
        elif self.pooling == "mean":
            pooled_output = pooled_output.mean(dim=1) 
        elif self.pooling == "max":
            pooled_output = pooled_output.max(dim=1).values
        elif self.pooling == "attention":
            attention_weights = torch.softmax(self.attention(pooled_output), dim=1)
            pooled_output = (pooled_output * attention_weights).sum(dim=1)
        elif self.pooling == 'last':
            if self.config.pad_token_id is None:
                last_non_pad_token = -1
            elif input_ids is not None:
                # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
                non_pad_mask = (input_ids != self.config.pad_token_id).to(pooled_output.device, torch.int32)
                token_indices = torch.arange(input_ids.shape[-1], device=pooled_output.device)
                last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
            pooled_output = pooled_output[torch.arange(pooled_output.shape[0], device=pooled_output.device), last_non_pad_token]
        else:
            raise ValueError("Unsupported pooling method")

        pooled_output = pooled_output - pooled_output.max(dim=1, keepdim=True).values

        classification_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            classification_loss = loss_fct(pooled_output.view(-1, self.config.num_labels), label_class.view(-1))

        if not return_dict:
            output = (pooled_output,) + outputs[1:]
            return (classification_loss,) + output if classification_loss is not None else output

        return CausalLMOutputWithPast(
            loss=classification_loss,
            logits=pooled_output,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )