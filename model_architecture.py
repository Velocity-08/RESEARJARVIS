import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Transformer Model
class TransformerChatbot(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_encoder_layers, num_decoder_layers, ff_hidden_dim, max_seq_len, dropout=0.1):
        super(TransformerChatbot, self).__init__()
        
        # Embedding layer (shared for input and output tokens)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer Encoder and Decoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=ff_hidden_dim, dropout=dropout),
            num_layers=num_encoder_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=ff_hidden_dim, dropout=dropout),
            num_layers=num_decoder_layers
        )
        
        # Final output layer (projection to vocabulary size)
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize weights and biases"""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask):
        # Add token and position embeddings
        src_seq_len, tgt_seq_len = src.size(1), tgt.size(1)
        src_pos = torch.arange(0, src_seq_len, device=src.device).unsqueeze(0)
        tgt_pos = torch.arange(0, tgt_seq_len, device=tgt.device).unsqueeze(0)
        
        src = self.token_embedding(src) + self.position_embedding(src_pos)
        tgt = self.token_embedding(tgt) + self.position_embedding(tgt_pos)

        # Pass through encoder and decoder
        memory = self.encoder(src, src_key_padding_mask=src_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask)
        
        # Project to vocabulary size
        logits = self.output_layer(output)
        return logits

# Define helper functions for masks
def generate_square_subsequent_mask(sz):
    """Generate a square mask for the sequence to prevent attending to future tokens."""
    mask = torch.triu(torch.ones(sz, sz)) == 1
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_padding_mask(seq, pad_idx):
    """Create a mask to ignore padded positions in the sequence."""
    return (seq == pad_idx).transpose(0, 1)

# Model Parameters
vocab_size = 5000  # Placeholder for vocab size
d_model = 512      # Embedding dimension
n_heads = 8        # Number of attention heads
num_encoder_layers = 6  # Encoder layers
num_decoder_layers = 6  # Decoder layers
ff_hidden_dim = 2048    # Feedforward network hidden dimension
max_seq_len = 100       # Maximum sequence length
dropout = 0.1           # Dropout rate

# Instantiate the model
model = TransformerChatbot(
    vocab_size=vocab_size,
    d_model=d_model,
    n_heads=n_heads,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    ff_hidden_dim=ff_hidden_dim,
    max_seq_len=max_seq_len,
    dropout=dropout
)

print(model)
  
