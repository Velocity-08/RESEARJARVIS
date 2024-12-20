import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn import Transformer
import json

# Define the TransformerChatbot model
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
class TokenizedChatbotDataset(Dataset):
    def __init__(self, dataset_path, max_seq_len):
        with open(dataset_path, 'r') as f:
            self.data = json.load(f)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query_tokens = self.data[idx]['query'][:self.max_seq_len]
        response_tokens = self.data[idx]['response'][:self.max_seq_len]

        query_tokens = self.pad_sequence(query_tokens)
        response_tokens = self.pad_sequence(response_tokens)

        return torch.tensor(query_tokens), torch.tensor(response_tokens)

    def pad_sequence(self, tokens):
        padded = tokens + [0] * (self.max_seq_len - len(tokens))
        return padded[:self.max_seq_len]

# Training Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = 30522  # Adjust based on your tokenizer
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
dataset_path = "tokenized_dataset.json"

# Load Tokenized Dataset
dataset = TokenizedChatbotDataset(dataset_path, max_seq_len)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize Model
model = TransformerChatbot(vocab_size, d_model, n_heads, num_encoder_layers, num_decoder_layers, ff_hidden_dim, max_seq_len, dropout).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the pad token ID
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
        src_padding_mask = src == 0  # Assuming 0 is the pad token ID
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
model_save_path = "trained_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved as '{model_save_path}'")
