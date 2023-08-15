import numpy as np

class Neuron:
    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = np.random.randn(input_size + 1)


    def __call__(self, v_input):
        return v_input @ self.weights[1:] + self.weights[0]



if __name__ == '__main__':
    neuron = Neuron(3)

    print(neuron(np.array([1, 2, 3])))
