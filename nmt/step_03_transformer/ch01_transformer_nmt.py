import math
import torch
import torch.nn as nn
import torch.optim as optim


class PositionalEncoding(nn.Module):
    """Fixed sin/cos positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # position shape: [max_len, 1]
        position = torch.arange(max_len).unsqueeze(1)
        # math skill for number stability
        # div_term shape: [d_model / 2]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        # position * div_term shape is broadcast to : [max_len, d_model / 2]
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # pe is not a parameter, but should be part of the module's state.
        # model.to(device) will also move pe to the right device and saved into model as parameters do.
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TranslationTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_index=0,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 4096,
    ):
        super().__init__()
        self.d_model = d_model

        # vocabulary embedding (for both src and tgt)
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_index)
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # We use [T, B, D] format (Time, Batch, D_model) //Embedding_dim
        )

        self.fc_out = nn.Linear(d_model, vocab_size, bias=False)

        # --- Weight Tying: Output layer weights = Token embedding weights (a common practice)
        # they can be shared because :
        #   1. input and output symetry. 2. same shape: [vocab_size, d_model], fc is xWᵀ, W is [out_features, in_features]
        self.fc_out.weight = self.token_embed.weight

        self._reset_parameters()

    def _reset_parameters(self):
        # PyTorch's default initialization is quite good, so this is a simple approach.
        # Can be changed to Xavier/Kaiming etc. if requested.
        # The original paper used xavier_uniform_, but a simple normal init is also common.
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=self.d_model**-0.5)

    @staticmethod
    def _generate_square_subsequent_mask(sz: int, device):
        # Decoder's auto-regressive mask (upper triangle is -inf)
        # masked_scores = scores + mask
        return torch.triu(
            torch.full((sz, sz), float("-inf"), device=device), diagonal=1
        )

    @staticmethod
    def _make_pad_mask(seq: torch.Tensor, pad_id: int):
        """
        seq: [T, B] tensor of token ids
        Returns: key padding mask -> [B, T], where True means the position should be masked (i.e., it's a PAD token)
        """
        return (seq == pad_id).transpose(0, 1)  # Shape: [B, T]

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor):
        """
        src:    [S, B] (Source sequence length, Batch size)
        tgt_in: [T, B] (Target sequence length, Batch size) - Decoder input, left-shifted and starting with BOS
        Returns: logits [T, B, V] (Target sequence length, Batch size, Vocab size)
        """
        device = src.device
        S, B = src.shape
        T, _ = tgt_in.shape

        # Embedding + Positional Encoding
        src_emb = self.pos_encoder(self.token_embed(src) * (self.d_model**0.5))
        tgt_emb = self.pos_encoder(self.token_embed(tgt_in) * (self.d_model**0.5))

        # Masks
        src_key_padding_mask = self._make_pad_mask(
            src, self.token_embed.padding_idx
        )  # [B, S]
        tgt_key_padding_mask = self._make_pad_mask(
            tgt_in, self.token_embed.padding_idx
        )  # [B, T]
        tgt_mask = self._generate_square_subsequent_mask(T, device)  # [T, T]
        # Pass through the Transformer
        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        # Linear projection to vocabulary
        return self.fc_out(out)  # Shape: [T, B, V]
