import numpy as np
import spicy as sp
import os

from data_generator import Data_generator

class RNN:

    def __init__(self, hidden_size: int, data_generator, sequence_size: int, learning_rate: float):
        
        self.hidden_size = hidden_size
        self.data_generator = data_generator
        self.vocab_size = self.data_generator.vocab_size
        self.sequence_size = sequence_size
        self.learning_rate = learning_rate
        self.X = None
        
        # init the model parameters
        self.Wxa_init = np.random.uniform(-np.sqrt(1. / self.vocab_size), np.sqrt(1. / self.vocab_size), (hidden_size, self.vocab_size))
        self.Waa_self = np.random.uniform(-np.sqrt(1. / hidden_size), np.sqrt(1. / hidden_size), (hidden_size, hidden_size))
        self.Wya_init = np.random.uniform(-np.sqrt(1. / hidden_size), np.sqrt(1. / hidden_size), (self.vocab_size, hidden_size))
        self.Ba = np.zeros((hidden_size, 1))
        self.By = np.zeros((self.vocab_size, 1))

        # init the gradient
        self.dWxa_init, self.dWaa_self, self.dWya_init = np.zeros_like(self.Wxa_init), np.zeros_like(self.Waa_self), np.zeros(self.Wya_init)
        self.dBa, self.dBy = np.zeros_like(self.Ba), np.zeros_like(self.By)

        # parameter update with Adam
        self.mWxa_init, self.vWxa_init = np.zeros_like(self.Wxa_init), np.zeros_like(self.Wxa_init)
        self.mWaa_self, self.vWaa_self = np.zeros_like(self.Waa_self), np.zeros_like(self.Waa_self)
        self.mWya_init, self.vWya_init = np.zeros_like(self.Wya_init), np.zeros_like(self.Wya_init)
        self.mBa, self.vBa = np.zeros_like(self.Ba), np.zeros_like(self.Ba)
        self.mBy, self.vBy = np.zeros_like(self.By), np.zeros_like(self.By)

        def softmax(self, x: np.ndarray) -> float:
            x = x - np.max(x)
            p = np.exp(x)
            return p / np.sum(p)
        
        def forward(self, X, a_prev):
            x, a, y_pred = {}, {}, {}
            self.X = X

            a[-1] = np.copy(a_prev)

            for t in range(len(self.X)):
                x[t] = np.zeros((self.vocab_size, 1))
                if (self.X[t] != None):
                    x[t][self.X[t]] = 1
                a[t] = np.tanh(np.dot(self.Wxa_init, x[t]) + np.dot(self.Waa_self, a[t - 1]) + self.Ba)

                y_pred[t] = self.softmax(np.dot(self.Wya_init, a[t]) + self.By)

                return x, a, y_pred

if __name__ == "__main__":
    RNN()