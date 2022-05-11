from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, bidirectional=True):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional)
        self.Wt_softmax = nn.Linear(num_layers * hidden_size, output_size)
        self.softmax = nn.Softmax()

    def forward(self, input, hx):
        output, hx = self.lstm(input, hx)
        output = output.squeeze(1)
        output = self.Wt_softmax(output)
        probs = self.softmax(output)
        actions = probs.multinomial()
        return actions

    def reset_parameters(self):
        self.lstm.reset_parameters()
