import torch
import torch.nn as nn
import torch.nn.functional as nnf
from typing import Tuple, Optional, List
from transformers import GPT2LMHeadModel
from utils import noise_injection, _get_clones

class att_gt_n_rt(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int = 8,
        num_heads: int = 8
    ) -> None:
        super(att_gt_n_rt, self).__init__()
        self.transformer = Transformer(d_model, num_layers, num_heads)

    def forward(self, x: torch.Tensor, rtf: torch.Tensor=None) -> torch.Tensor:
        rtf = rtf if rtf is not None else x
        outputs = self.transformer(x, rtf)
        return outputs

class MlpTransformer(nn.Module):

    def __init__(
        self,
        input_size: int,                   # the input size of mlp
        hidden_size: int,                  # the hidden layer size of mlp
        output_size: Optional[int] = None, # the output size of mlp
        act = nnf.relu,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        output_size = output_size if output_size is not None else input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act = act
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        query_size: int,
        key_value_size: int,
        num_heads: int,
        bias = True,
        dropout: float = 0.0
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = query_size // num_heads # the size of each head
        self.scale = self.head_size ** -0.5      # normalization factor for each head
        self.to_queries = nn.Linear(query_size, query_size, bias = bias)
        #  projecting key and value together and spliting them for computing efficiently
        self.to_keys_values = nn.Linear(key_value_size, 2 * query_size, bias = bias)
        self.project = nn.Linear(query_size, query_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        key_value = key_value if key_value is not None else query
        b, n, d_query = query.shape
        _, m, _ = key_value.shape
        queries = self.to_queries(query).reshape(b, n, self.num_heads, self.head_size) # (batch_size, n_seq, num_heads, head_size)
        keys_values = self.to_keys_values(key_value).reshape(b, m, 2, self.num_heads, self.head_size) # (batch_size, m_seq, 2, num_heads, head_size)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1] # (batch_size, m_seq, num_heads, head_size), (batch_size, m_seq, num_heads, head_size)
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale # (batch_size, n_seq, m_seq, num_heads)
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(dim = 1) # expending dimension, shape: (batch_size, 1, m_seq)
            attention = attention.masked_fill(mask.unsqueeze(dim = 3), float("-inf")) # expending dimension n_seq head and fill -inf according to mask

        attention = attention.softmax(dim = 2) # softmax alongside the dimension of key_value pairs
        outputs = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, d_query) # (batch_size, n_seq, d_query)
        outputs = self.project(outputs)
        return outputs, attention

class TransformerLayer(nn.Module):

    def __init__(
            self,
            query_size: int,
            key_value_size: int,
            num_heads: int,
            mlp_ratio = 4.0,
            bias = False,
            dropout: float = 0.0,
            act = nnf.relu,
            norm_layer: nn.Module = nn.LayerNorm
        ) -> None:
        super(TransformerLayer, self).__init__()
        self.norm1 = norm_layer(query_size)
        self.attn = MultiHeadAttention(query_size, key_value_size, num_heads, bias = bias, dropout = dropout)
        self.norm2 = norm_layer(query_size)
        self.mlp = MlpTransformer(query_size, int(query_size * mlp_ratio), act = act, dropout = dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        query_, self.attention = self.attn(self.norm1(query), key_value, mask)
        query = query + query_
        query = query + self.mlp(self.norm2(query))
        return query

class Transformer(nn.Module):

    def __init__(
            self,
            query_size: int,                      # query size
            num_layers: int,                      # number of layer
            num_heads: int,                       # number of head
            key_value_size: Optional[int] = None, # key/value size
            mlp_ratio: float = 2.0,               # ratio for hidden size in mlp
            act = nnf.relu,                       # activation
            norm_layer: nn.Module = nn.LayerNorm  # normalization
        ) -> None:
        super(Transformer, self).__init__()
        key_value_size = key_value_size if key_value_size is not None else query_size
        layers = []
        for _ in range(num_layers):
            layers.append(TransformerLayer(query_size, key_value_size, num_heads, mlp_ratio = mlp_ratio, act = act, norm_layer = norm_layer))
        self.layers = nn.Sequential(*layers)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        self.attentions = []
        for layer in self.layers:
            query = layer(query, key_value, mask)
            self.attentions.append(layer.attention)
        return query

class MappingNetwork(nn.Module):

    def __init__(
        self,
        clip_project_length: int,
        clip_hidden_size: int,
        prefix_length: int,
        d_model: int,              # the hidden size of language model
        num_layers: int = 8,
        num_heads: int = 8,
        k: int = 3,
        device: Optional[str] = None
    ) -> None:
        super(MappingNetwork, self).__init__()
        self.clip_project_length = clip_project_length
        # projector for input
        self.linear = nn.Linear(clip_hidden_size, clip_project_length * d_model)
        self.rt_linear = nn.Linear(clip_hidden_size, d_model)

        # learnable prefix embeddings
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, d_model), requires_grad = True)
        self.transformer = Transformer(d_model, num_layers, num_heads)
        self.k = k
        
        ## cross-attention layer
        self.crossatt = att_gt_n_rt(d_model, 1, num_heads)
        self.device = device
        
    def forward(self, x: torch.Tensor, rtf: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: clip cls feature with a shape of (batch_size, clip_hidden_size)
        Return:
            the embeddings of prefix with the shape of (batch_size, prefix_length, d_model)
        """
        x = self.linear(x).view(x.shape[0], self.clip_project_length, -1)  # (b, clip_project_length, d_model)
        prefix = self.prefix_const.unsqueeze(dim=0).expand(x.shape[0], *self.prefix_const.shape)  # (b, prefix_length, d_model)

        rtf = self.rt_linear(rtf)
        att_gt_n_rt = self.crossatt(x, rtf)

        inputs = torch.cat((att_gt_n_rt, prefix), dim = 1) # (b, clip_project_length + prefix_length, d_model)
        outputs = self.transformer(inputs)[:, self.clip_project_length:, :] # (b, prefix_length, d_model)

        return outputs

class TITTransformerLayer(nn.Module):

    def __init__(
            self,
            query_size: int,
            key_value_size: int,
            num_heads: int,
            mlp_ratio = 4.0,
            bias = False,
            dropout: float = 0.0,
            act = nnf.relu,
            norm_layer: nn.Module = nn.LayerNorm
        ) -> None:
        super(TITTransformerLayer, self).__init__()
        
        self.crossattn1 = TransformerLayer(query_size, key_value_size, num_heads, mlp_ratio = mlp_ratio, act = act, norm_layer = norm_layer)
        # attn1layers = []
        # for _ in range(5):
        #     attn1layers.append(subattn1)
        # self.attn1layers = nn.Sequential(*attn1layers)
        self.mlp1 = MlpTransformer(query_size, int(query_size * mlp_ratio), act = act, dropout = dropout)
        self.activefunc1 = nn.GELU()
        self.norm1 = norm_layer(query_size)
              
        self.crossattn2 = TransformerLayer(query_size, key_value_size, num_heads, mlp_ratio = mlp_ratio, act = act, norm_layer = norm_layer)
        
        self.selfattn3 = TransformerLayer(query_size, key_value_size, num_heads, mlp_ratio = mlp_ratio, act = act, norm_layer = norm_layer)
        self.mlp2 = MlpTransformer(query_size, int(query_size * mlp_ratio), act = act, dropout = dropout)
        self.activefunc2 = nn.GELU()
        self.norm2 = norm_layer(query_size)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
              
        # for i in range(key_value.shape[1]):
        attn1kv = torch.cat((query, key_value), dim = 1)
        query_ = self.crossattn1(key_value, attn1kv)
        query_ = self.mlp1(query_)
        attn2kv = self.norm1(self.activefunc1(query_))

        # attn2kv = torch.cat(attn2kv, dim = 1)
        # query = query + query_
        atten3qkv = self.crossattn2(query, attn2kv)
        atten3qkv = self.selfattn3(atten3qkv, atten3qkv)
        query = self.mlp2(atten3qkv)
        query = self.activefunc2(query)
        # query = query + self.mlp(self.norm2(query))
        
        return self.norm2(query), attn2kv

class TITTransformer(nn.Module):

    def __init__(
            self,
            query_size: int,                      # query size
            num_layers: int,                      # number of layer
            num_heads: int,                       # number of head
            key_value_size: Optional[int] = None, # key/value size
            mlp_ratio: float = 2.0,               # ratio for hidden size in mlp
            act = nnf.relu,                       # activation
            norm_layer: nn.Module = nn.LayerNorm  # normalization
        ) -> None:
        super(TITTransformer, self).__init__()
        key_value_size = key_value_size if key_value_size is not None else query_size
        layers = []
        for _ in range(num_layers):
            layers.append(TITTransformerLayer(query_size, key_value_size, num_heads, mlp_ratio = mlp_ratio, act = act, norm_layer = norm_layer))
        self.layers = nn.Sequential(*layers)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        # self.attentions = []
        for layer in self.layers:
            query, key_value = layer(query, key_value, mask)
            # self.attentions.append(layer.attention)
        return query

class TIT(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int = 8,
        num_heads: int = 8
    ) -> None:
        super(TIT, self).__init__()
        self.transformer = TITTransformer(d_model, num_layers, num_heads)

    def forward(self, x: torch.Tensor, rtf: torch.Tensor=None) -> torch.Tensor:
        rtf = rtf if rtf is not None else x
        outputs = self.transformer(x, rtf)
        return outputs
    
class MappingT2II2TNetwork(nn.Module):

    def __init__(
        self,
        clip_project_length: int,
        clip_hidden_size: int,
        prefix_length: int,
        d_model: int,              # the hidden size of language model
        num_layers: int = 8,
        num_heads: int = 8,
        k: int = 3,
        device: Optional[str] = None
    ) -> None:
        super(MappingT2II2TNetwork, self).__init__()
        self.clip_project_length = clip_project_length
        # projector for input
        self.linear = nn.Linear(clip_hidden_size, clip_project_length * d_model)
        self.rt_linear = nn.Linear(clip_hidden_size, d_model)

        # learnable prefix embeddings
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, d_model), requires_grad = True)
        self.transformer = Transformer(d_model, num_layers, num_heads)
        self.k = k
    
        ## cross-attention layer
        # self.crossatt = att_gt_n_rt(d_model, 1, num_heads)
        # self.device = device
        ## TIT layer
        self.crossatt = TIT(d_model, 1, num_heads)
        self.device = device
        
        
    def forward(self, x: torch.Tensor, rtf: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: clip cls feature with a shape of (batch_size, clip_hidden_size)
        Return:
            the embeddings of prefix with the shape of (batch_size, prefix_length, d_model)
        """
        x = self.linear(x).view(x.shape[0], self.clip_project_length, -1)  # (b, clip_project_length, d_model)
        prefix = self.prefix_const.unsqueeze(dim=0).expand(x.shape[0], *self.prefix_const.shape)  # (b, prefix_length, d_model)

        rtf = self.rt_linear(rtf)
        att_gt_n_rt = self.crossatt(x, rtf)

        inputs = torch.cat((att_gt_n_rt, prefix), dim = 1) # (b, clip_project_length + prefix_length, d_model)
        outputs = self.transformer(inputs)[:, self.clip_project_length:, :] # (b, prefix_length, d_model)

        return outputs

class T2II2TClipCaptionModel(nn.Module):

    def __init__(
        self,
        args,
        continuous_length: int = 10,
        clip_project_length: int = 10,
        clip_hidden_size: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        gpt_type: str = 'gpt2',
        soft_prompt_first: bool = False,
        only_hard_prompt: bool = False,
        prefix: bool = False,
        k: int = 3
    ) -> None:
        """
        Args:
            continuous_length: the length of soft prompts which will be fed into language model as continuous part
            clip_project_length: clip cls features (b, 1, d) -> (b, n, d)
            clip_hidden_size: the dimensions of CLIP features
            num_layers: the number of layer in projector
            num_heads: the number of heads each layer
            gpt_type: the language model
            soft_prompt_first: False -> hard prompt + soft prompt; True -> soft prompt + hard prompt
            only_hard_prompt: using the hard prompts only
        """
        super(T2II2TClipCaptionModel, self).__init__()
        self.soft_prompt_first = soft_prompt_first
        self.only_hard_prompt = only_hard_prompt
        self.continuous_length = continuous_length
        self.gpt, self.gpt_hidden_size = get_language_mode(gpt_type)
        self.mapping_network = MappingT2II2TNetwork(clip_project_length, clip_hidden_size, continuous_length, self.gpt_hidden_size, num_layers, num_heads, k, args.device)
        self.gpt_type = gpt_type
        self.args = args


    def word_embed(self, caption_tokens):
        if 'gpt' in self.gpt_type:
            caption_embeddings = self.gpt.transformer.wte(caption_tokens)         # (b, caption_length, gpt_hidden_size)
        elif 'opt' in self.gpt_type:
            caption_embeddings = self.gpt.model.decoder.embed_tokens(caption_tokens)
        return caption_embeddings
    
    def forward(
        self,
        continuous_prompt: torch.Tensor,
        caption_tokens: torch.Tensor,
        hard_prompts_length: Optional[List] = None,
        mask: Optional[torch.Tensor] = None,
        retrieved_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            continuous_prompt: tensor with a shape of (b, clip_hidden_size), in text-only training, the caption features are eaxtracted from CLIP and used as image features
            caption_tokens: caption tokens with a shape of (b, max_length_per_caption)
            hard_prompts_length: list with len = batch size, the length of hard prompts constructed for each caption
            mask: tensor with a shape of (b, discrete_length + continuous_length + max_length_per_caption), valid texts for attention computing
        Return:
            the output of language model
        """
        caption_embeddings = self.word_embed(caption_tokens) # caption_tokens = captions_tokens_with_hard_prompts
        
        continuous_embeddings = self.mapping_network(continuous_prompt, retrieved_features)#.view(-1, self.continuous_length, self.gpt_hidden_size) # (b, continuous_length, gpt_hidden_size)

        if hard_prompts_length is not None:   # with hard prompts
            if self.only_hard_prompt:
                embeddings = caption_embeddings
            elif self.soft_prompt_first:      # soft prompts + hard prompts
                embeddings = torch.cat((continuous_embeddings, caption_embeddings), dim = 1)
            else:                             # hard prompts + soft prompts
                embeddings = None
                for i in range(len(hard_prompts_length)):
                    length = hard_prompts_length[i]
                    temp_embeddings = torch.cat((caption_embeddings[i][:length], continuous_embeddings[i], caption_embeddings[i][length:]), dim = 0).unsqueeze(dim = 0)
                    if embeddings is None:
                        embeddings = temp_embeddings
                    else:
                        embeddings = torch.cat((embeddings, temp_embeddings), dim = 0)
        else: # without hard prompts
            embeddings = torch.cat((continuous_embeddings, caption_embeddings), dim = 1) # (b, continuous_length + caption_length, gpt_hidden_size)

        out = self.gpt(inputs_embeds = embeddings.type(self.gpt.dtype), attention_mask = mask)

        return out

class InteractMappingNetwork(nn.Module):

    def __init__(
        self,
        clip_project_length: int,
        clip_hidden_size: int,
        prefix_length: int,
        d_model: int,              # the hidden size of language model
        num_layers: int = 8,
        num_heads: int = 8,
        k: int = 3,
        device: Optional[str] = None
    ) -> None:
        super(InteractMappingNetwork, self).__init__()
        self.clip_project_length = clip_project_length
        # projector for input
        self.linear = nn.Linear(clip_hidden_size, clip_project_length * d_model)
        self.rt_linear = nn.Linear(clip_hidden_size, d_model)

        # learnable prefix embeddings
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, d_model), requires_grad = True)
        self.transformer = Transformer(d_model, num_layers, num_heads)
        self.k = k

        # vision-language Interaction
        feature_fusion_layer = BiAttentionBlock(
            v_dim=d_model,
            l_dim=d_model,
            embed_dim=clip_project_length * d_model // 2,
            num_heads=num_heads // 2,
            dropout=0.1,
            drop_path=0.0,
        )       

        if feature_fusion_layer is not None:
            self.fusion_layers = _get_clones(
                feature_fusion_layer, num_layers
            )
            
            for layer_id, layer in enumerate(self.fusion_layers):
                if layer_id in lora_disable_layers:
                    self.fusion_layers[layer_id].attn_disable_lora()       
        
        ## cross-attention layer
        self.crossatt = att_gt_n_rt(d_model, 1, num_heads)
        self.device = device
        
    def attn_disable_lora(self):
        self.attn = BiMultiHeadAttention(
            v_dim=self.v_dim, l_dim=self.l_dim, embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=self.dropout, use_moe_lora=False
        )        
        
    def forward(self, x: torch.Tensor, rtf: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: clip cls feature with a shape of (batch_size, clip_hidden_size)
        Return:
            the embeddings of prefix with the shape of (batch_size, prefix_length, d_model)
        """
        x = self.linear(x).view(x.shape[0], self.clip_project_length, -1)  # (b, clip_project_length, d_model)
        prefix = self.prefix_const.unsqueeze(dim=0).expand(x.shape[0], *self.prefix_const.shape)  # (b, prefix_length, d_model)

        rtf = self.rt_linear(rtf)
        # insert MM interaction
        x = self.layer_norm_v(x)
        rtf = self.layer_norm_l(rtf)
        delta_x, delta_rtf = self.attn(
            x, rtf
        )        
        x = x + self.drop_path(self.gamma_v * delta_x)
        rtf = rtf + self.drop_path(self.gamma_l * delta_rtf)
        # insert MM interaction
        att_gt_n_rt = self.crossatt(x, rtf)

        inputs = torch.cat((att_gt_n_rt, prefix), dim = 1) # (b, clip_project_length + prefix_length, d_model)
        outputs = self.transformer(inputs)[:, self.clip_project_length:, :] # (b, prefix_length, d_model)

        return outputs

def get_language_mode(lm_type):
    if 'gpt' in lm_type:
        model = GPT2LMHeadModel.from_pretrained(lm_type)
        hidden_size = model.config.hidden_size
    elif 'opt' in lm_type:
        from modeling_opt import OPTForCausalLM
        model = OPTForCausalLM.from_pretrained(lm_type, torch_dtype = torch.float16)
        hidden_size = model.config.word_embed_proj_dim
    return model, hidden_size

class ClipCaptionModel(nn.Module):

    def __init__(
        self,
        args,
        continuous_length: int = 10,
        clip_project_length: int = 10,
        clip_hidden_size: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        gpt_type: str = 'gpt2',
        soft_prompt_first: bool = False,
        only_hard_prompt: bool = False,
        prefix: bool = False,
        k: int = 3
    ) -> None:
        """
        Args:
            continuous_length: the length of soft prompts which will be fed into language model as continuous part
            clip_project_length: clip cls features (b, 1, d) -> (b, n, d)
            clip_hidden_size: the dimensions of CLIP features
            num_layers: the number of layer in projector
            num_heads: the number of heads each layer
            gpt_type: the language model
            soft_prompt_first: False -> hard prompt + soft prompt; True -> soft prompt + hard prompt
            only_hard_prompt: using the hard prompts only
        """
        super(ClipCaptionModel, self).__init__()
        self.soft_prompt_first = soft_prompt_first
        self.only_hard_prompt = only_hard_prompt
        self.continuous_length = continuous_length
        self.gpt, self.gpt_hidden_size = get_language_mode(gpt_type)
        self.mapping_network = MappingNetwork(clip_project_length, clip_hidden_size, continuous_length, self.gpt_hidden_size, num_layers, num_heads, k, args.device)
        self.gpt_type = gpt_type
        self.args = args


    def word_embed(self, caption_tokens):
        if 'gpt' in self.gpt_type:
            caption_embeddings = self.gpt.transformer.wte(caption_tokens)         # (b, caption_length, gpt_hidden_size)
        elif 'opt' in self.gpt_type:
            caption_embeddings = self.gpt.model.decoder.embed_tokens(caption_tokens)
        return caption_embeddings
    
    def forward(
        self,
        continuous_prompt: torch.Tensor,
        caption_tokens: torch.Tensor,
        hard_prompts_length: Optional[List] = None,
        mask: Optional[torch.Tensor] = None,
        retrieved_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            continuous_prompt: tensor with a shape of (b, clip_hidden_size), in text-only training, the caption features are eaxtracted from CLIP and used as image features
            caption_tokens: caption tokens with a shape of (b, max_length_per_caption)
            hard_prompts_length: list with len = batch size, the length of hard prompts constructed for each caption
            mask: tensor with a shape of (b, discrete_length + continuous_length + max_length_per_caption), valid texts for attention computing
        Return:
            the output of language model
        """
        caption_embeddings = self.word_embed(caption_tokens) # caption_tokens = captions_tokens_with_hard_prompts
        
        continuous_embeddings = self.mapping_network(continuous_prompt, retrieved_features)#.view(-1, self.continuous_length, self.gpt_hidden_size) # (b, continuous_length, gpt_hidden_size)

        if hard_prompts_length is not None:   # with hard prompts
            if self.only_hard_prompt:
                embeddings = caption_embeddings
            elif self.soft_prompt_first:      # soft prompts + hard prompts
                embeddings = torch.cat((continuous_embeddings, caption_embeddings), dim = 1)
            else:                             # hard prompts + soft prompts
                embeddings = None
                for i in range(len(hard_prompts_length)):
                    length = hard_prompts_length[i]
                    temp_embeddings = torch.cat((caption_embeddings[i][:length], continuous_embeddings[i], caption_embeddings[i][length:]), dim = 0).unsqueeze(dim = 0)
                    if embeddings is None:
                        embeddings = temp_embeddings
                    else:
                        embeddings = torch.cat((embeddings, temp_embeddings), dim = 0)
        else: # without hard prompts
            embeddings = torch.cat((continuous_embeddings, caption_embeddings), dim = 1) # (b, continuous_length + caption_length, gpt_hidden_size)

        out = self.gpt(inputs_embeds = embeddings.type(self.gpt.dtype), attention_mask = mask)

        return out

class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.mapping_network.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

# from .cl_lora_moe import LoraLinear

from cl_lora import LoraLinear


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def func_attention(query, context, smooth=1, raw_feature_norm="softmax", eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    else:
        raise ValueError("unknown first norm type:", raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax()(attn * smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT


class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1, cfg=None, use_moe_lora = False):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim 
        self.use_moe_lora = use_moe_lora

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)
        
        if use_moe_lora:
            self.v_proj = LoraLinear(base_layer=self.v_proj)
            self.l_proj = LoraLinear(base_layer=self.l_proj)
            
            self.values_v_proj = LoraLinear(base_layer=self.values_v_proj)
            self.values_l_proj = LoraLinear(base_layer=self.values_l_proj)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = True
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()
        
    

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        if self.use_moe_lora:
            nn.init.xavier_uniform_(self.v_proj.base_layer_.weight)
            self.v_proj.base_layer_.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.l_proj.base_layer_.weight)
            self.l_proj.base_layer_.bias.data.fill_(0)
            
            nn.init.xavier_uniform_(self.values_v_proj.base_layer_.weight)
            self.values_v_proj.base_layer_.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.values_l_proj.base_layer_.weight)
            self.values_l_proj.base_layer_.bias.data.fill_(0)


        else:
            nn.init.xavier_uniform_(self.v_proj.weight)
            self.v_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.l_proj.weight)
            self.l_proj.bias.data.fill_(0)
            
            nn.init.xavier_uniform_(self.values_v_proj.weight)
            self.values_v_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.values_l_proj.weight)
            self.values_l_proj.bias.data.fill_(0)
     

        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        self.out_l_proj.bias.data.fill_(0)

    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None, open_lora=True):
        """_summary_

        Args:
            v (_type_): bs, n_img, dim
            l (_type_): bs, n_text, dim
            attention_mask_v (_type_, optional): _description_. bs, n_img
            attention_mask_l (_type_, optional): _description_. bs, n_text

        Returns:
            _type_: _description_
        """
        # if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
        #     import ipdb; ipdb.set_trace()
        bsz, tgt_len, _ = v.size()

        if self.use_moe_lora and open_lora == False:
            query_states = self.v_proj(v, open_lora=False) * self.scale
            key_states = self._shape(self.l_proj(l, open_lora=False), -1, bsz)
            value_v_states = self._shape(self.values_v_proj(v, open_lora=False), -1, bsz)
            value_l_states = self._shape(self.values_l_proj(l, open_lora=False), -1, bsz)
        else:
            query_states = self.v_proj(v) * self.scale
            
   
            key_states = self._shape(self.l_proj(l), -1, bsz)
            value_v_states = self._shape(self.values_v_proj(v), -1, bsz)

            value_l_states = self._shape(self.values_l_proj(l), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))  # bs*nhead, nimg, ntxt

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(
                attn_weights, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(
                attn_weights, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_l = attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[0]
        if self.clamp_min_for_underflow:
            attn_weights_l = torch.clamp(
                attn_weights_l, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_l = torch.clamp(
                attn_weights_l, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        # mask vison for language
        if attention_mask_v is not None:
            attention_mask_v = (
                attention_mask_v[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            attn_weights_l.masked_fill_(attention_mask_v, float("-inf"))

        attn_weights_l = attn_weights_l.softmax(dim=-1)

        # mask language for vision
        if attention_mask_l is not None:
            attention_mask_l = (
                attention_mask_l[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            attn_weights.masked_fill_(attention_mask_l, float("-inf"))
        attn_weights_v = attn_weights.softmax(dim=-1)

        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)

        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}"
            )

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)

        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_l = self.out_l_proj(attn_output_l)
        
        
        return attn_output_v, attn_output_l

        # return torch.cat([attn_output_v, attn_output_l],dim=1)


# Bi-Direction MHA (text->image, image->text)
class BiAttentionBlock(nn.Module):
    def __init__(
        self,
        v_dim,
        l_dim,
        embed_dim,
        num_heads,
        dropout=0.1,
        drop_path=0.0,
        init_values=1e-4,
        cfg=None,
        use_moe_lora = False
    ):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(BiAttentionBlock, self).__init__()

        # pre layer norm
        self.v_dim = v_dim
        self.l_dim = l_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.use_moe_lora = use_moe_lora
        self.attn = BiMultiHeadAttention(
            v_dim=v_dim, l_dim=l_dim, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, use_moe_lora=self.use_moe_lora
        )

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)
        self.gamma_l = nn.Parameter(init_values * torch.ones((l_dim)), requires_grad=True)
        
        
        
    def attn_disable_lora(self):
        self.attn = BiMultiHeadAttention(
            v_dim=self.v_dim, l_dim=self.l_dim, embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=self.dropout, use_moe_lora=False
        )

    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None, open_lora=True):
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        
        # print(v.shape)
        # print(l.shape)
   
        delta_v, delta_l = self.attn(
            v, l, attention_mask_v=attention_mask_v, attention_mask_l=attention_mask_l, open_lora=open_lora
        )
        
        # delta_all = self.attn(
        #     v, l, attention_mask_v=attention_mask_v, attention_mask_l=attention_mask_l
        # )
        
        # delta_v = delta_all[:,:v.shape[1]]
        # delta_l = delta_all[:,v.shape[1]:]

        # v, l = v + delta_v, l + delta_l
        v = v + self.drop_path(self.gamma_v * delta_v)
        l = l + self.drop_path(self.gamma_l * delta_l)
        
        # torch.cat([v, memory_text], dim=some_dimension)
        return v, l
        # return torch.cat([v, l], dim=1)

    # def forward(self, v:List[torch.Tensor], l, attention_mask_v=None, attention_mask_l=None)

class ClipCaptionInteractModel(nn.Module):

    def __init__(
        self,
        args,
        continuous_length: int = 10,
        clip_project_length: int = 10,
        clip_hidden_size: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        gpt_type: str = 'gpt2',
        soft_prompt_first: bool = False,
        only_hard_prompt: bool = False,
        prefix: bool = False,
        k: int = 3
    ) -> None:
        """
        Args:
            continuous_length: the length of soft prompts which will be fed into language model as continuous part
            clip_project_length: clip cls features (b, 1, d) -> (b, n, d)
            clip_hidden_size: the dimensions of CLIP features
            num_layers: the number of layer in projector
            num_heads: the number of heads each layer
            gpt_type: the language model
            soft_prompt_first: False -> hard prompt + soft prompt; True -> soft prompt + hard prompt
            only_hard_prompt: using the hard prompts only
        """
        super(ClipCaptionModel, self).__init__()
        self.soft_prompt_first = soft_prompt_first
        self.only_hard_prompt = only_hard_prompt
        self.continuous_length = continuous_length
        self.gpt, self.gpt_hidden_size = get_language_mode(gpt_type)
        self.mapping_network = InteractMappingNetwork(clip_project_length, clip_hidden_size, continuous_length, self.gpt_hidden_size, num_layers, num_heads, k, args.device)
        # self.hard_prompt_layer = nn.Linear(in_features=768, out_features=768)  # 输入维度3，输出维度2
        self.gpt_type = gpt_type
        self.args = args


    def word_embed(self, caption_tokens):
        if 'gpt' in self.gpt_type:
            caption_embeddings = self.gpt.transformer.wte(caption_tokens)         # (b, caption_length, gpt_hidden_size)
        elif 'opt' in self.gpt_type:
            caption_embeddings = self.gpt.model.decoder.embed_tokens(caption_tokens)
        return caption_embeddings
    
    def forward(
        self,
        continuous_prompt: torch.Tensor,
        caption_tokens: torch.Tensor,
        hard_prompts_length: Optional[List] = None,
        mask: Optional[torch.Tensor] = None,
        retrieved_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            continuous_prompt: tensor with a shape of (b, clip_hidden_size), in text-only training, the caption features are eaxtracted from CLIP and used as image features
            caption_tokens: caption tokens with a shape of (b, max_length_per_caption)
            hard_prompts_length: list with len = batch size, the length of hard prompts constructed for each caption
            mask: tensor with a shape of (b, discrete_length + continuous_length + max_length_per_caption), valid texts for attention computing
        Return:
            the output of language model
        """
        caption_embeddings = self.word_embed(caption_tokens) # caption_tokens = captions_tokens_with_hard_prompts
        
        # caption_embeddings = self.hard_prompt_layer(caption_embeddings)
        
        continuous_embeddings = self.mapping_network(continuous_prompt, retrieved_features)#.view(-1, self.continuous_length, self.gpt_hidden_size) # (b, continuous_length, gpt_hidden_size)

        if hard_prompts_length is not None:   # with hard prompts
            if self.only_hard_prompt:
                embeddings = caption_embeddings
            elif self.soft_prompt_first:      # soft prompts + hard prompts
                embeddings = torch.cat((continuous_embeddings, caption_embeddings), dim = 1)
            else:                             # hard prompts + soft prompts
                embeddings = None
                for i in range(len(hard_prompts_length)):
                    length = hard_prompts_length[i]
                    temp_embeddings = torch.cat((caption_embeddings[i][:length], continuous_embeddings[i], caption_embeddings[i][length:]), dim = 0).unsqueeze(dim = 0)
                    if embeddings is None:
                        embeddings = temp_embeddings
                    else:
                        embeddings = torch.cat((embeddings, temp_embeddings), dim = 0)
        else: # without hard prompts
            embeddings = torch.cat((continuous_embeddings, caption_embeddings), dim = 1) # (b, continuous_length + caption_length, gpt_hidden_size)

        out = self.gpt(inputs_embeds = embeddings.type(self.gpt.dtype), attention_mask = mask)

        return out

if __name__ == "__main__":


    feature_fusion_layer = BiAttentionBlock(
        v_dim=256,
        l_dim=256,
        embed_dim=2048 // 2,
        num_heads=8 // 2,
        dropout=0.1,
        drop_path=0.0,
        use_moe_lora = True
    )
 
