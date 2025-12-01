from torch import nn
from transformers import AutoModel


class IndoBertClassifier(nn.Module):
    def __init__(self, model_name,
                 dense_1=64, dense_2=16, dropout=0.05, num_labels=5):
        super(IndoBertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # Layers according to your Keras architecture
        self.pool = nn.AdaptiveMaxPool1d(1)   # GlobalMaxPool1D equivalent
        self.fc1 = nn.Linear(hidden_size, dense_1)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_1, dense_2)
        self.fc3 = nn.Linear(dense_2, num_labels)
        self.act_relu = nn.ReLU()
        self.act_sigmoid = nn.Sigmoid()       # multi-label case

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state  # shape: (batch, seq_len, hidden)

        # PyTorch pooling works on (N, C, L), so permute first
        x = embeddings.permute(0, 2, 1)        # (batch, hidden, seq_len)
        x = self.pool(x).squeeze(-1)           # (batch, hidden)

        x = self.fc1(x)
        x = self.act_relu(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.act_relu(x)

        logits = self.fc3(x)
        out = self.act_sigmoid(logits)
        return out
