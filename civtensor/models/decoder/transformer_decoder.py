from torch import nn
from civtensor.models.blocks.decoder_layer import DecoderLayer
from civtensor.models.embedding.transformer_embedding import  TransformerEmbedding
class TransformerDecoder(nn.Module):
    def __init__(self,  d_model, ffn_hidden, n_head, n_layers, drop_prob):
        super().__init__()
        # self.emb = TransformerEmbedding(d_model=d_model,
        #                                 drop_prob=drop_prob,
        #                                 max_len=max_len,
        #                                 vocab_size=dec_voc_size,
        #                                 device=device)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        # self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, src, trg_mask, src_mask):
        """
        accept trg and src is  the next state(embedding) and trg(encoded embedding)
        ans trg_msk and src_msk are from the observation

        """
        # trg = self.emb(trg)

        # todo how to get the predicted len without the mask
        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)

        # pass to LM head
        output = trg
        # output = self.linear(trg)
        return output

    def decode(self, trg):
        for layer in self.layers:
            trg = layer(trg, None, None, None)

        return trg