import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=(fs, embedding_dim))
                for fs in [2, 3, 4, 5, 6]
            ]
        )

        self.fc = nn.Linear(5 * 32, 1)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]

        x = [F.max_pool1d(_, _.shape[2]).squeeze(2) for _ in x]

        x = self.dropout(torch.cat(x, dim=1))

        return self.fc(x)
