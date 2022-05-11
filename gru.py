import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, in_dim, output_size, hidden_dim, n_layer, lookup):
        super(GRU, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim

        self.lstm = nn.GRU(in_dim, hidden_dim, n_layer, batch_first=True, bidirectional=False)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x):  # [1,20,5]->[1,20,hidden_dim]->[1,20,6]->[1,20,6]
        saved_log_probs = []
        output, _ = self.lstm(x)
        output = self.classifier(output)
        output = output.view(-1, 2)
        softmax = nn.Softmax(dim=1)
        output = softmax(output)
        output = torch.distributions.Categorical(output)
        action = output.sample()
        saved_log_probs.append(output.log_prob(action))
        action = output.sample().reshape(-1, 1)
        # action = action.repeat(1, 5).reshape(-1)
        action = action.reshape(-1)
        # print(np.shape(action))
        return action, saved_log_probs
