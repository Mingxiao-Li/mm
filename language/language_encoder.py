import torch.nn as nn
import json
import gzip
import torch


class LanguageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if self.config.use_pretrained_embeddings:
            self.embedding_layer = nn.Embedding.from_pretrained(
                embeddings=self._load_embeddings(),
                freeze = not self.config.fine_tune_embeddings,
            )
        else:
            self.embedding_layer = nn.Embedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.embedding_size,
                padding_idx=0,
            )

    def _load_embeddings(self):
        """Loads word embeddings from a pretrained embeddings file.
        PAD: index 0, [0.0, ... ,0.0]
        UNK: index 1, mean of all word embeddings
        :return: embeddings tensor of size [num_words x embedding_dim]
        """
        with gzip.open(self.config.embedding_file, "rt") as f:
            embeddings = torch.tensor(json.load(f))
        return embeddings

    def forward(self, inputs):
        inputs = inputs.long()
        embedded = self.embedding_layer(inputs)


if __name__ == "__main__":
    file = "../pretrained_file/embeddings.json.gz"
    with gzip.open(file,"rt") as f:
        embeddings = torch.tensor(json.load(f))
    print(embeddings.shape)