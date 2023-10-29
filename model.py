import torch
import torch.nn as nn
from transformers.models.longt5.modeling_longt5 import LongT5ForConditionalGeneration
class ReAttentionBlock(nn.Module):
    def __init__(self, input_dim):
        super(ReAttentionBlock, self).__init__()
        self.multihead_self_attention = nn.MultiheadAttention(input_dim, 8, dropout = 0.1)
        self.ln1 = nn.LayerNorm(input_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, input_dim)
        )

        self.ln2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        self_attention_output, _ = self.multihead_self_attention(x, x, x)
        x = x + self_attention_output
        x = self.ln1(x)

        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.ln2(x)

        return x

class MyModel(LongT5ForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super(MyModel, self).__init__(*args, **kwargs)
        self.freeze_parameters()
        self.ReattentionBlock = ReAttentionBlock(input_dim = self.encoder.embed_tokens.embedding_dim)
        blocks = list(self.encoder.block)
        new_blocks = [blocks[0], self.ReattentionBlock] + [blocks[1], self.ReattentionBlock] + blocks[1:]
        self.encoder.block = nn.ModuleList(new_blocks) 
    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        decoder_head_mask=None,
        **kwargs
    ):
        return super(MyModel, self).forward(
            input_ids=input_ids,
            attention_mask=None,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            **kwargs
        )

def load_model():
    model = MyModel.from_pretrained("pszemraj/long-t5-tglobal-base-16384-book-summary")
    return model

if __name__ == '__main__':
    print(load_model())