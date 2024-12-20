import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn import Transformer
import json
from transformers import GPT2Tokenizer

# Load the GPT2 tokenizer and set vocab size
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
vocab_size = len(tokenizer)  # Get vocab size directly from the tokenizer

# Define the Transformer Chatbot Model
class TransformerChatbot(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_encoder_layers, num_decoder_layers, ff_hidden_dim, max_seq_len, dropout):
        super(TransformerChatbot, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.transformer = Transformer(
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

# Dataset Preparation
class ChatbotDataset(Dataset):
    def __init__(self, dataset_path, max_seq_len, vocab_size):
        with open(dataset_path, 'r') as f:
            self.data = json.load(f)
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query_tokens = self.data[idx]['gpt_query_tokens']  # Already tokenized
        response_tokens = self.data[idx]['gpt_query_tokens']  # Assuming responses are also tokenized similarly

        self.validate_tokens(query_tokens)
        self.validate_tokens(response_tokens)

        # Pad the sequences to max_seq_len
        query_tokens = self.pad_sequence(query_tokens)
        response_tokens = self.pad_sequence(response_tokens)

        return torch.tensor(query_tokens), torch.tensor(response_tokens)

    def pad_sequence(self, tokens):
        # Pad with 0 (assuming 0 is padding token) up to max_seq_len
        padded = tokens + [0] * (self.max_seq_len - len(tokens))
        return padded[:self.max_seq_len]  # Ensure no sequence exceeds max_seq_len

    def validate_tokens(self, tokens):
        for token in tokens:
            if token >= self.vocab_size:
                print(f"Warning: Token {token} exceeds vocab size {self.vocab_size - 1}")


# Training Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d_model = 512
n_heads = 8
num_encoder_layers = 6
num_decoder_layers = 6
ff_hidden_dim = 2048
max_seq_len = 50
dropout = 0.1
batch_size = 16
learning_rate = 0.001
epochs = 10
dataset_path = "tokenized_dataset.json"  # Your tokenized dataset file

# Load Dataset
dataset = ChatbotDataset(dataset_path, max_seq_len, vocab_size)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize Model
model = TransformerChatbot(vocab_size, d_model, n_heads, num_encoder_layers, num_decoder_layers, ff_hidden_dim, max_seq_len, dropout).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token index (assuming 0 is padding)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
def create_mask(src, tgt):
    src_seq_len = src.size(1)
    tgt_seq_len = tgt.size(1)

    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
    tgt_mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=device), diagonal=1).type(torch.bool)
    return src_mask, tgt_mask

print("Starting training...")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_idx, (src, tgt) in enumerate(data_loader):
        print(f"Processing batch {batch_idx + 1}/{len(data_loader)}")
        src, tgt = src.to(device), tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask, tgt_mask = create_mask(src, tgt_input)
        src_padding_mask = src == 0  # Assuming padding token is 0
        tgt_padding_mask = tgt_input == 0

        outputs = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)
        outputs = outputs.reshape(-1, outputs.size(-1))
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(outputs, tgt_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(data_loader)}")

# Save the trained model
model_save_path = "transformer_chatbot_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
