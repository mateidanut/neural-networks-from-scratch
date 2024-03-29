import numpy as np


class Layer:
    def __init__(self, input_size, output_size, weight_multiplier=0.01):
        self.input_size = input_size
        self.output_size = output_size

        self.neurons = weight_multiplier * np.random.randn(input_size + 1, output_size)
        #self.neurons = np.ones((input_size + 1, output_size))
        self.neurons[0, :] = 0


    def forward(self, batched_input):

        batched_input_with_bias = np.insert(
            batched_input,
            0,
            1,
            axis=1
        )

        print('INITIAL INPUT SHAPE', batched_input.shape)
        print('INPUT SHAPE', batched_input_with_bias.shape)
        print('LAYER SHAPE', self.neurons.shape)

        return batched_input_with_bias @ self.neurons


if __name__ == '__main__':
    layer = Layer(3, 4)

    print(layer.forward(np.array([
            [1, 2, 3],
            [1, 1, 1],
        ])
        ))
