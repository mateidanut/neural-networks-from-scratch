import numpy as np


class Layer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.neurons = np.random.randn(input_size + 1, output_size)


    def __call__(self, v_input):
        return np.append(1, v_input) @ self.neurons


if __name__ == '__main__':
    layer = Layer(3, 4)

    print(layer(np.array([1, 2, 3])))
