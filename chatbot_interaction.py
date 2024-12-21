import torch
import torch.nn as nn
from transformers import AutoTokenizer

# Load the TransformerChatbot model class
class TransformerChatbot(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_encoder_layers, num_decoder_layers, ff_hidden_dim, max_seq_len, dropout):
        super(TransformerChatbot, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=ff_hidden_dim,
            dropout=dropout
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask):
        src_emb = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt_emb = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]

        src_emb = self.dropout(src_emb)
        tgt_emb = self.dropout(tgt_emb)

        transformer_out = self.transformer(
            src_emb.permute(1, 0, 2),
            tgt_emb.permute(1, 0, 2),
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

        output = self.fc_out(transformer_out.permute(1, 0, 2))
        return output

# Load the model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = 50257  # Adjust to match your tokenizer's vocabulary size
d_model = 512
n_heads = 8
num_encoder_layers = 6
num_decoder_layers = 6
ff_hidden_dim = 2048
max_seq_len = 50
dropout = 0.1

# Load tokenizer (BERT or GPT tokenizer)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Initialize model
model = TransformerChatbot(vocab_size, d_model, n_heads, num_encoder_layers, num_decoder_layers, ff_hidden_dim, max_seq_len, dropout).to(device)
model.load_state_dict(torch.load("trained_model.pth", map_location=device))
model.eval()

# Function to create masks
def create_mask(src, tgt):
    src_seq_len = src.size(1)
    tgt_seq_len = tgt.size(1)

    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
    tgt_mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=device), diagonal=1).type(torch.bool)
    return src_mask, tgt_mask

# Function to generate a response
def generate_response(input_text):
    input_tokens = tokenizer.encode(input_text, truncation=True, max_length=max_seq_len - 2)
    input_tokens = [tokenizer.bos_token_id] + input_tokens + [tokenizer.eos_token_id]

    input_tensor = torch.tensor([input_tokens]).to(device)
    src_padding_mask = input_tensor == tokenizer.pad_token_id

    # Start with BOS token for the response
    response_tokens = [tokenizer.bos_token_id]
    for _ in range(max_seq_len):
        tgt_tensor = torch.tensor([response_tokens]).to(device)
        src_mask, tgt_mask = create_mask(input_tensor, tgt_tensor)

        tgt_padding_mask = tgt_tensor == tokenizer.pad_token_id
        outputs = model(input_tensor, tgt_tensor, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)
        next_token = torch.argmax(outputs[:, -1, :], dim=-1).item()

        # Stop generation at EOS token
        if next_token == tokenizer.eos_token_id:
            break

        response_tokens.append(next_token)

    response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
    return response_text

# Interactive loop
print("Chatbot is ready! Type 'exit' to end the chat.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    response = generate_response(user_input)
    print(f"Chatbot: {response}")
