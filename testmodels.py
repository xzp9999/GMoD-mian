from functools import partial

import torch
import torch.nn as nn
from torch import einsum
from timm.models.vision_transformer import DropPath, Mlp

import numpy as np
import scipy.stats as stats
import torch.nn.functional as F
from einops import rearrange

from torch_geometric.data import Data, DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv
from copy import deepcopy
import math
from torch.distributions import Categorical


# --- Convolution Modules ---
class CNN(nn.Module):
    def __init__(self, model, model_type='resnet'):
        super().__init__()
        if 'res' in model_type.lower():  # resnet, resnet-50, resnest-50, ...
            modules = list(model.children())[:-1]  # Drop the FC layer
            self.feature = nn.Sequential(*modules[:-1])
            self.average = modules[-1]
        elif 'dense' in model_type.lower():  # densenet, densenet-121, densenet121, ...
            modules = list(model.features.children())[:-1]
            self.feature = nn.Sequential(*modules)
            self.average = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise ValueError('Unsupported model_type!')

    def forward(self, input):

        wxh_features = self.feature(input)  # (B,2048,W,H)
        avg_features = self.average(wxh_features)  # (B,2048,1,1)
        avg_features = avg_features.view(avg_features.shape[0], -1)  # (B,2048)
        batch_size, feat_size, _, _ = wxh_features.shape
        wxh_features = wxh_features.reshape(batch_size, feat_size, -1).permute(0, 2, 1)  # (B,WxH,2048)
        return avg_features, wxh_features


'''class MVCNN(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):
        img = input[0]  # (B,V,C,W,H)
        pos = input[1]  # (B,V)
        B, V, C, W, H = img.shape

        img = img.view(B * V, C, W, H)
        avg, wxh = self.model(img)  # (B*V,F), (B*V,F,W,H)
        wxh = wxh.view(B, V, wxh.shape[-2], wxh.shape[-1])  # (B,V,WxH,F)

        msk = (pos == -1)  # (B,V)
        msk_wxh = msk.view(B, V, 1).repeat(1, 1, wxh.shape[2])

        wxh_features = wxh.reshape(B, -1, wxh.shape[-1])  # (B,WxH,F)
        return wxh_features, msk_wxh.reshape(B, -1)'''
class MVCNN(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):

        avg, wxh = self.model(input)  # (B*V,F), (B*V,F,W,H)

        return wxh, avg


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.normalize = nn.LayerNorm(embed_dim)

    def forward(self, input, query, pad_mask=None, att_mask=None):
        input = input.permute(1, 0, 2)  # (V,B,E)
        query = query.permute(1, 0, 2)  # (Q,B,E)
        embed, att = self.attention(query, input, input, key_padding_mask=pad_mask,
                                    attn_mask=att_mask)  # (Q,B,E), (B,Q,V)

        embed = self.normalize(embed + query)  # (Q,B,E)
        embed = embed.permute(1, 0, 2)  # (B,Q,E)
        return embed, att  # (B,Q,E), (B,Q,V)


class PointwiseFeedForward(nn.Module):
    def __init__(self, emb_dim, fwd_dim, dropout=0.0):
        super().__init__()
        self.fwd_layer = nn.Sequential(
            nn.Linear(emb_dim, fwd_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fwd_dim, emb_dim),
        )
        self.normalize = nn.LayerNorm(emb_dim)

    def forward(self, input):
        output = self.fwd_layer(input)  # (B,L,E)
        output = self.normalize(output + input)  # (B,L,E)
        return output


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, fwd_dim, dropout=0.0):
        super().__init__()
        self.self_attention = MultiheadAttention(embed_dim, num_heads, dropout)
        self.cross_attention = MultiheadAttention(embed_dim, num_heads, dropout)
        self.fwd_layer = PointwiseFeedForward(embed_dim, fwd_dim, dropout)

    def forward(self, input, query, cls_out, pad_mask=None, att_mask=None, txt_pad_mask=None):
        emb, att = self.self_attention(query, query, txt_pad_mask, att_mask)
        emb, att = self.cross_attention(input, emb, pad_mask, att_mask=None)
        emb, att = self.cross_attention(cls_out, emb, pad_mask, att_mask=None)
        emb = self.fwd_layer(emb)
        return emb, att


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, fwd_dim, dropout=0.0):
        super().__init__()
        self.attention = MultiheadAttention(embed_dim, num_heads, dropout)
        self.fwd_layer = PointwiseFeedForward(embed_dim, fwd_dim, dropout)

    def forward(self, input, query, pad_mask=None, att_mask=None):
        emb, att = self.attention(input, query, pad_mask, att_mask)
        emb = self.fwd_layer(emb)
        return emb, att


class MlmLayer(nn.Module):

    def __init__(self, feat_emb_dim, word_emb_dim, max_len):
        super().__init__()
        self.fc = nn.Linear(feat_emb_dim, word_emb_dim)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(word_emb_dim)  # 7863
        self.fc2 = nn.Linear(word_emb_dim, 7864)    # bos: 0  1~7863

    def forward(self, x):
        mlm_hidden = self.fc(x)
        mlm_hidden = self.gelu(mlm_hidden)
        mlm_hidden = self.ln(mlm_hidden)
        logits = self.fc2(mlm_hidden)
        return logits


class GraphAttentionModel(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(GraphAttentionModel, self).__init__()
        self.conv1 = GATConv(embedding_dim, embedding_dim)
        self.conv2 = GATConv(embedding_dim, embedding_dim)
        self.conv3 = GATConv(embedding_dim, embedding_dim)


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        return x


class Classifier(nn.Module):
    def __init__(self, num_topics, num_states, embed_dim=128, num_heads=1, dropout=0.1):
        super().__init__()

        # For classification
        self.topic_embedding = nn.Embedding(num_topics, embed_dim)
        self.state_embedding = nn.Embedding(num_states, embed_dim)
        self.topic_attention = MultiheadAttention(embed_dim, num_heads, dropout)
        self.state_attention = MultiheadAttention(embed_dim, num_heads)

        self.GAT = GraphAttentionModel(embed_dim)
        # Some constants
        self.num_topics = num_topics
        self.num_states = num_states

    def forward(self, all_features, all_pad_mask, edge_index, lbl=None, threshold=0.5, get_embed=False):
        topic_index = torch.arange(self.num_topics).unsqueeze(0).repeat(all_features.shape[0], 1).to(
            all_features.device)  # (B,T)
        state_index = torch.arange(self.num_states).unsqueeze(0).repeat(all_features.shape[0], 1).to(
            all_features.device)  # (B,C)
        topic_embed = self.topic_embedding(topic_index)  # (B,T,E)
        state_embed = self.state_embedding(state_index)  #

        edge_index = torch.tensor(edge_index).to(topic_embed.device)

        data_list = []
        for i in range(topic_embed.shape[0]):
            data = Data(x=topic_embed[i], edge_index=edge_index)
            data_list.append(data)
        batch = Batch.from_data_list(data_list)

        topic_embed = self.GAT(batch.x, batch.edge_index).reshape(topic_embed.shape)

        final_embed, cla_att = self.topic_attention(all_features, topic_embed, all_pad_mask)  # (B,T,E)
        # Classifier output
        emb, att = self.state_attention(state_embed, final_embed)  # (B,T,E), (B,T,C)

        if lbl != None:  # Teacher forcing
            emb = self.state_embedding(lbl)  # (B,T,E)
        else:
            emb = self.state_embedding((att[:, :, 1] > threshold).long())  # (B,T,E)

        if get_embed:
            return att, final_embed + emb  # (B,T,C), (B,T,E)
        else:
            return att  # (B,T,C)


# class ImgMappingToDict(nn.Module):
#
#     def __init__(self, feat_emb_dim, num_dict, vocab_size):
#         super().__init__()
#         self.fc = nn.Linear(feat_emb_dim, num_dict)
#         self.gelu = nn.GELU()
#         self.ln = nn.LayerNorm(num_dict)
#
#     def forward(self, x, word_embeddings):
#         mlm_hidden = self.fc(x)
#         mlm_hidden = self.gelu(mlm_hidden)
#         mlm_hidden = self.ln(mlm_hidden)
#
#         return
#
#
# class TextMappingToDict(nn.Module):
#
#     def __init__(self, feat_emb_dim, num_dict, vocab_size):
#         super().__init__()
#         self.fc = nn.Linear(feat_emb_dim, num_dict)
#         self.gelu = nn.GELU()
#         self.ln = nn.LayerNorm(num_dict)
#
#     def forward(self, x, word_embeddings):
#         mlm_hidden = self.fc(x)
#         mlm_hidden = self.gelu(mlm_hidden)
#         mlm_hidden = self.ln(mlm_hidden)
#
#         return


class Model(nn.Module):
    def __init__(self, classifier, cnn=None, num_layers=3, fc_features=2048, embed_dim=128, fwd_dim=256,
                 num_heads=8, dropout=0.1, num_dict=2048, edge_index=None, max_len=None, tokenizer=None,
                 momentum=0.995, alpha=0.995):

        super().__init__()
        # For img & txt embedding and feature extraction
        self.cnn = cnn
        self.fwd_dim = fwd_dim

        self.max_len = max_len
        self.tokenizer = tokenizer

        self.img_Linear = nn.Linear(fc_features, embed_dim)
        self.img_pos = nn.Embedding(128, embed_dim)
        self.normalize = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.token_embedding = nn.Embedding(7864, embed_dim)
        self.posit_embedding = nn.Embedding(max_len + 1, embed_dim)



        self.img_encoder = nn.ModuleList(
            [TransformerLayer(embed_dim, num_heads, fwd_dim, dropout) for _ in range(num_layers)])
        self.text_encoder = nn.ModuleList(
            [TransformerLayer(embed_dim, num_heads, fwd_dim, dropout) for _ in range(num_layers)])
        self.cross_decoder = nn.ModuleList(
            [DecoderLayer(embed_dim, num_heads, fwd_dim, dropout) for _ in range(num_layers*2)])

        self.temperature = nn.Parameter(torch.Tensor([1.]))

        self.mlm_layer = MlmLayer(feat_emb_dim=embed_dim, word_emb_dim=embed_dim, max_len=max_len)
        self.itm_head = nn.Linear(fwd_dim, 2)

        # feature dict
        self.feature_dict = nn.Embedding(num_dict, fwd_dim)
        self.feature_dict.weight.data.uniform_(-1.0 / num_dict, 1.0 / num_dict)
        self.temperature_for_dict = nn.Parameter(torch.Tensor([1.]))
        self.classifier = classifier
        self.edge_index = edge_index


        self.momentum = momentum
        self.classifier_m = deepcopy(self.classifier)
        self.model_pairs = [[self.classifier, self.classifier_m]]
        self.alpha = alpha

    def forward(self, image=None, caption_idx=None, caption_mask=None, label=None, bos_id=0, top_k=1, mode='one',
                num_iters=0, epoch=0, epochs=0, i=0, val_test_phase=False):
        label = label.long().to(image.device) if label != None else label
        txt = caption_idx

        # --- Get img features --
        out_img, _ = self.cnn(image)
        out_img = self.img_Linear(F.normalize(out_img, dim=-1))
        img_pos_index = torch.arange(out_img.shape[1]).unsqueeze(0).repeat(out_img.shape[0], 1).to(
            out_img.device)  # (B,L)
        img_pos = self.img_pos(img_pos_index)
        out_img = self.dropout(out_img + img_pos)

        # --- concate class token --
        # out_img = torch.cat([self.cls_token.repeat(out_img.shape[0], 1, 1), out_img], 1)
        # img_pad_mask = torch.cat([(torch.zeros(out_img.shape[0], 1) == 1).to(out_img.device), img_pad_mask], 1)
        if val_test_phase:
            return self.generate(out_img, None, txt=txt, caption_mask=caption_mask)
        elif txt != None:
            return self.forward_cross(out_img, txt, caption_mask, label, num_iters=num_iters, epoch=epoch, i=i, epochs=epochs)
        else:  # --- Inference Phase ---
            return self.infer(out_img, None, self.max_len, top_k, bos_id, caption_mask, label)

    def infer(self, out_img, img_pad_mask=None, max_len=100, top_k=1, bos_id=0, caption_mask=None, label=None):

        outputs = torch.ones((top_k, out_img.shape[0], 1), dtype=torch.long).to(
            out_img.device) * bos_id  # (K,B,1) <s>
        scores = torch.zeros((top_k, out_img.shape[0]), dtype=torch.float32).to(out_img.device)  # (K,B)

        for _ in range(1, max_len):
            possible_outputs = []
            possible_scores = []

            for k in range(top_k):
                output = outputs[k]  # (B,L)
                score = scores[k]  # (B)

                x, img_mlc = self.generate(out_img, img_pad_mask, txt=output, caption_mask=caption_mask, label=label)
                val, idx = torch.topk(x[:, -1, :], top_k)  # (B,K)
                log_val = val  # (B,K)

                for i in range(top_k):
                    new_output = torch.cat([output, idx[:, i].view(-1, 1)], dim=-1)  # (B,L+1)
                    new_score = score + log_val[:, i].view(-1)  # (B)
                    possible_outputs.append(new_output.unsqueeze(0))  # (1,B,L+1)
                    possible_scores.append(new_score.unsqueeze(0))  # (1,B)

            possible_outputs = torch.cat(possible_outputs, dim=0)  # (K^2,B,L+1)
            possible_scores = torch.cat(possible_scores, dim=0)  # (K^2,B)

            # Pruning the solutions
            val, idx = torch.topk(possible_scores, top_k, dim=0)  # (K,B)
            col_idx = torch.arange(idx.shape[1], device=idx.device).unsqueeze(0).repeat(idx.shape[0], 1)  # (K,B)
            outputs = possible_outputs[idx, col_idx]  # (K,B,L+1)
            scores = possible_scores[idx, col_idx]  # (K,B)

        val, idx = torch.topk(scores, 1, dim=0)  # (1,B)
        col_idx = torch.arange(idx.shape[1], device=idx.device).unsqueeze(0).repeat(idx.shape[0], 1)  # (K,B)
        output = outputs[idx, col_idx]  # (1,B,L)
        score = scores[idx, col_idx]  # (1,B)
        return output.squeeze(0)[:, 1:]  # (B,L)

    def generate(self, out_img, img_pad_mask, txt, caption_mask, label=None, threshold=0.1):

        # image encoder
        for layer in self.img_encoder:
            img_out, _ = layer(out_img, out_img, pad_mask=img_pad_mask)

        # text encoder
        text_out, text_seq_mask = self.txt_encoder(txt, caption_mask)

        img_mlc, cls_out = self.classifier(img_out, img_pad_mask, self.edge_index, lbl=label, threshold=threshold, get_embed=True)

        # Multi Model
        for layer in self.cross_decoder:
            # layer((k,v)  q)
            attn_output, _ = layer(img_out, text_out, cls_out, att_mask=text_seq_mask,
                                   txt_pad_mask=caption_mask)

        x = self.mlm_layer(attn_output)

        return x, img_mlc

    def txt_encoder(self, txt, caption_mask):
        # text encoder

        txt_pos_index = torch.arange(txt.shape[1] + 1).unsqueeze(0).repeat(txt.shape[0], 1).to(
            txt.device)  # (B,L)
        txt_pos = self.posit_embedding(txt_pos_index)  # (B,L,E)
        txt_embed = self.token_embedding(txt)  # (B,L,E)
        out_txt = txt_embed + txt_pos[:, :-1, :]
        text_seq_mask = self.generate_square_subsequent_mask(out_txt.size(1)).to(txt.device)
        for layer in self.text_encoder:
            # layer((k,v)  q)
            text_out, _ = layer(out_txt, out_txt, pad_mask=caption_mask, att_mask=text_seq_mask)

        return text_out, text_seq_mask

    def forward_cross(self, out_img, txt, caption_mask, label=None, img_pad_mask=None, threshold=0.15,
                      num_iters=0, epoch=0, i=0, epochs=0):

        # image encoder
        for layer in self.img_encoder:
            img_out, _ = layer(out_img, out_img, pad_mask=img_pad_mask)

        # text encoder
        text_out, text_seq_mask = self.txt_encoder(txt, caption_mask)

        # alpha = self._rampup_factor(self.alpha, epoch, epochs=epochs)
        final_value = 0.95
        self.alpha = self.alpha - (self.alpha - final_value) * (1 - math.cos(math.pi * epoch / epochs))

        with torch.no_grad():
            self._momentum_update()
            img_mlc_m, cls_out_m = self.classifier_m(img_out, img_pad_mask, self.edge_index, lbl=label
                                                     , threshold=threshold, get_embed=True)
            label_m = torch.zeros(label.size(0), label.size(1), 2).to(label.device)
            label_m.scatter_(-1, label.unsqueeze(-1), 1)

            img_mlc_m = (1 - self.alpha) * F.softmax(img_mlc_m.view(-1, 2), dim=1).view(-1, 114, 2) + self.alpha * label_m

        img_mlc, cls_out = self.classifier(img_out, img_pad_mask, self.edge_index, lbl=label,
                                           threshold=threshold, get_embed=True)

        cls_loss = -torch.sum(F.log_softmax(img_mlc, dim=1).view(-1, 2) * img_mlc_m.view(-1, 2), dim=1).mean()

        # Multi Model
        for layer in self.cross_decoder:
            # layer((k,v)  q)
            attn_output, _ = layer(img_out, text_out, cls_out, att_mask=text_seq_mask,
                                   txt_pad_mask=caption_mask)
        x = self.mlm_layer(attn_output)

        return x, cls_loss

    def _rampup_factor(self, alpha, epoch=0, epochs=0):
        final_value = 0
        alpha = final_value + 0.5 * (alpha - final_value) * (1 + math.cos(math.pi * epoch / epochs))
        # min(1, (epoch * num_iters_per_epoch + iters) / (3 * num_iters_per_epoch))
        return alpha

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    def feature_Dictionary(self, image_embeds, text_embeds):

        B, N_I, D = image_embeds.size(0), image_embeds.size(1), image_embeds.size(2)
        N_T = text_embeds.size(1)

        # image
        # image_embeds_no_cls = image_embeds[:, 1:, :]
        # text_embeds_no_cls = text_embeds[:, 1:, :]
        image_embeds_flattened = image_embeds.contiguous().view(-1, self.fwd_dim)  # (n,d)
        # distances from Image_feature to dict : (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(image_embeds_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.feature_dict.weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', image_embeds_flattened, rearrange(self.feature_dict.weight, 'n d -> d n'))

        min_indices_i = torch.argmin(d, dim=1)  # 返回矩阵每一行的最小值的下标
        dict_base_image = self.feature_dict(min_indices_i).view(B, N_I, D) \
            # compute loss for embedding
        mse_loss_i = torch.mean((dict_base_image.detach() - image_embeds) ** 2) + \
                     torch.mean((dict_base_image - image_embeds.detach()) ** 2)

        dict_base_image = image_embeds + (dict_base_image - image_embeds).detach()
        # dict_base_image = torch.cat([image_embeds[:, 0:1, :], dict_base_image], dim=1)

        # text
        text_embeds_flattened = text_embeds.contiguous().view(-1, self.fwd_dim)
        # distances from Image_feature to dict : (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(text_embeds_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.feature_dict.weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', text_embeds_flattened, rearrange(self.feature_dict.weight, 'n d -> d n'))

        min_indices_t = torch.argmin(d, dim=1)  # 返回矩阵每一行的最小值的下标
        dict_base_text = self.feature_dict(min_indices_t).view(B, N_T, D)

        mse_loss_t = torch.mean((dict_base_text.detach() - text_embeds) ** 2) + \
                     torch.mean((dict_base_text - text_embeds.detach()) ** 2)

        dict_base_text = text_embeds + (dict_base_text - text_embeds).detach()
        # dict_base_text = torch.cat([text_embeds[:, 0:1, :], dict_base_text], dim=1)

        mse_loss = mse_loss_t + mse_loss_i

        return dict_base_image, dict_base_text, mse_loss

    def generate_square_subsequent_mask_with_source(self, src_sz, tgt_sz, mode='eye'):
        mask = self.generate_square_subsequent_mask(src_sz + tgt_sz)
        if mode == 'one':  # model can look at surrounding positions of the current index ith
            mask[:src_sz, :src_sz] = self.generate_square_mask(src_sz)
        elif mode == 'eye':  # model can only look at the current index ith
            mask[:src_sz, :src_sz] = self.generate_square_identity_mask(src_sz)
        else:  # model can look at surrounding positions of the current index ith with some patterns
            raise ValueError('Mode must be "one" or "eye".')
        mask[src_sz:, src_sz:] = self.generate_square_subsequent_mask(tgt_sz)
        return mask

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate_square_identity_mask(self, sz):
        mask = (torch.eye(sz) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate_square_mask(self, sz):
        mask = (torch.ones(sz, sz) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate_square_subsequent_mask_with_cls(self, tgt_sz):
        mask = self.generate_square_subsequent_mask(2 * tgt_sz).transpose(0, 1)
        mask[:tgt_sz, :tgt_sz] = self.generate_square_identity_mask(tgt_sz)
        mask[:tgt_sz, tgt_sz:] = self.generate_square_subsequent_mask(tgt_sz)
        mask[tgt_sz:, tgt_sz:] = self.generate_square_subsequent_mask(tgt_sz)
        return mask
