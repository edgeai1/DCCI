from torch import optim

from Layer import Layer
from gru import *


class Controller:
    def __init__(self, controllerClass, input_size, output_size, hidden_size, num_layers, lr=0.003, skipSupport=False, kwargs={}):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.kwargs = kwargs
        if isinstance(controllerClass, str):
            self.controller = torch.load(controllerClass)
        else:
            # self.controller = LSTM(input_size, output_size, hidden_size, num_layers)
            # self.controller = controllerClass(input_size, output_size, hidden_size, num_layers, **kwargs)
            self.controller = GRU(input_size, output_size, hidden_size, num_layers, **kwargs)
        self.optimizer = optim.Adam(self.controller.parameters(), lr=lr)
        self.skipSupport = skipSupport
        self.actionSeqs = []
        self.tempSeqs = []

    def update_controller(self, avgR):
        for i in range(len(self.tempSeqs)):
            temps = self.tempSeqs[i]
            loss = 0
            for j in range(len(temps)):
                loss += temps[j] * (avgR)
            self.optimizer.zero_grad()
            loss = loss.float()
            loss = loss.sum()
            loss.backward()
            self.optimizer.step()
        self.actionSeqs = []
        self.tempSeqs = []

    def rolloutActions(self, layers):
        num_input = self.input_size
        input = torch.Tensor(len(layers), 1, num_input)
        for i in range(len(layers)):
            input[i] = Layer(layers[i][0]).toTorchTensor(skipSupport=self.skipSupport)
        actions, saved_log_probs = self.controller(input)
        self.actionSeqs.append(actions)
        self.tempSeqs.append(saved_log_probs)
        return actions, saved_log_probs


def getEpsilon(iter, max_iter=15.0):
    return min(1, max(0, (1 - iter / float(max_iter)) ** 4))  # return 0


def getConstrainedReward(R_a, R_c, acc, params, acc_constraint, size_constraint, epoch, soft=True):
    eps = getEpsilon(epoch) if soft else 0
    if (size_constraint and params > size_constraint) or (acc_constraint and acc < acc_constraint):
        return (eps - 1) + eps * (R_a * R_c)
    return R_a * R_c


previousModels = {}
