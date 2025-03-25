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


class AttentionPooling(nn.Module):
    """
    Overview:
        Attention pooling layer on the sequence dimension of LLM/VLM hidden states.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        qkv_bias: bool = False,
        position_bias: bool = False,
        position_bias_scale: float = 3.0,
    ):
        super(AttentionPooling, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.position_bias = position_bias
        self.position_bias_scale = position_bias_scale

        self.k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        # 0.02 for better initialization
        self.query = nn.Parameter(torch.randn(hidden_size) * 0.02)

    def forward(self, hidden_states):
        B, S, C = hidden_states.shape

        # Multi-head projection for key and value
        k = self.k(hidden_states).reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B, H, S, D
        v = self.v(hidden_states).reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B, H, S, D

        # Expand query for batch dimension
        q = self.query.unsqueeze(0).expand(B, -1, -1)  # B, H, C
        q = q.unsqueeze(2)  # B, H, 1, C
        q = q.reshape(B, self.num_heads, 1, self.head_dim)  # B, H, 1, C

        # Attention weights
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, H, 1, S

        # Add position bias
        if self.position_bias:
            position_bias = torch.arange(S, device=k.device).float() / S * self.position_bias_scale
            attn = attn + position_bias.view(1, 1, 1, -1)  # Add position bias

        # Attention pooling
        attn = torch.softmax(attn, dim=-1)  # B, H, 1, S
        out = (attn @ v).squeeze(2)  # B, H, D
        out = out.reshape(B, -1)  # B, C

        return out

class InternVLSequenceClassificationModel(InternVLChatModel):
    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, add_classify_head = 'last_hidden_states', pooling = 'last'):
        super().__init__(config, vision_model, language_model)
        self.add_classify_head = add_classify_head # last_hidden_states, middle_hidden_states
        self.pooling = pooling

        self.classify_attention = AttentionPooling(hidden_size=self.hidden_size)
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

        if self.pooling == "cls":
            pooled_output = pooled_output[:, 0, :] 
        elif self.pooling == "mean":
            pooled_output = pooled_output.mean(dim=1) 
        elif self.pooling == "max":
            pooled_output = pooled_output.max(dim=1).values
        elif self.pooling == "attention":
            pooled_output = self.classify_attention(pooled_output)
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

        pooled_output = self.classifier_heads(pooled_output)

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

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template) 
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        generation_config['eos_token_id'] = eos_token_id
        with torch.no_grad():
            outputs = self.forward(pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=torch.tensor([1] * pixel_values.size(0), dtype=torch.long), return_dict=True)
            logits = outputs[0]
            predicted_label = torch.argmax(logits, dim=-1)
        # generation_output = self.generate(
        #     pixel_values=pixel_values,
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     **generation_config
        # )
        
        return predicted_label

    def chat_batch(self, tokenizer, pixel_values, questions, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False, batch_size=1):
        """
        Process multiple image comparisons in parallel
        
        Args:
            tokenizer: The tokenizer to use
            pixel_values: Tensor containing all images
            questions: List of questions to ask
            generation_config: Generation configuration
            history: Conversation history
            num_patches_list: List of number of patches for each image
            batch_size: Number of comparisons to process
            
        Returns:
            List of predicted labels for each comparison
        """
        assert batch_size == len(questions), "Batch size must match number of questions"
        
        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template) 
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        # Process each question
        all_input_ids = []
        all_attention_masks = []
        
        for i, question in enumerate(questions):
            if history is None and pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question

            history_i = [] if history is None else history
            for (old_question, old_answer) in history_i:
                template.append_message(template.roles[0], old_question)
                template.append_message(template.roles[1], old_answer)
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            # Calculate the start and end indices for this batch's images
            start_idx = 0
            if i > 0:
                # For each previous batch, add the number of patches
                for j in range(i):
                    start_idx += num_patches_list[j*2] + num_patches_list[j*2+1]
            
            # Get the number of patches for this batch's images
            num_patches_i = [num_patches_list[i*2], num_patches_list[i*2+1]]
            
            for num_patches in num_patches_i:
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
                query = query.replace('<image>', image_tokens, 1)

            model_inputs = tokenizer(query, return_tensors='pt')
            all_input_ids.append(model_inputs['input_ids'])
            all_attention_masks.append(model_inputs['attention_mask'])
            
            # Reset template for next question
            template = get_conv_template(self.template)
            template.system_message = self.system_message

        # Pad to the same length and batch together
        max_length = max(ids.size(1) for ids in all_input_ids)
        padded_input_ids = []
        padded_attention_masks = []
        
        for input_ids, attention_mask in zip(all_input_ids, all_attention_masks):
            pad_length = max_length - input_ids.size(1)
            if pad_length > 0:
                padded_input_ids.append(torch.cat([input_ids, torch.zeros(1, pad_length, dtype=torch.long)], dim=1))
                padded_attention_masks.append(torch.cat([attention_mask, torch.zeros(1, pad_length, dtype=torch.long)], dim=1))
            else:
                padded_input_ids.append(input_ids)
                padded_attention_masks.append(attention_mask)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_ids = torch.cat(padded_input_ids, dim=0).to(device)
        attention_mask = torch.cat(padded_attention_masks, dim=0).to(device)
        
        generation_config['eos_token_id'] = eos_token_id
        
        with torch.no_grad():
            outputs = self.forward(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=torch.tensor([1] * pixel_values.size(0), dtype=torch.long), 
                return_dict=True
            )
            logits = outputs[0]
            predicted_labels = torch.argmax(logits, dim=-1)
        
        return predicted_labels, logits