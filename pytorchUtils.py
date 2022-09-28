import torch

def num_corrects(outputs, label_batch):
    """
        How many number of outputs of a model is equal to 
        true labels
    :param outputs: outputs of a model
    :type outputs: :py:class:`torch.Tensor`
    :param label_batch: True labels of a model
    :type label_batch: :py:class:`torch.Tensor`
    """
    out = outputs.argmax(1)
    corrects = out == label_batch
    return torch.sum(corrects).item()



class MaxoutMLP(torch.nn.Module):
    """
        Maxout using Multilayer Perceptron
    """
    def __init__(self, input_size, linear_layers, linear_neurons):
        """

        :param input_size: number of values (pixels or hidden unit's output_ that
                            will be inputted to the layer
        :param linear_layers:  number of linear layers before max operation
        :param linear_neurons: number of nerons in each linear layer before max operation
        """
        super(MaxoutMLP, self).__init__()

        # initialize variables
        self.input_size = input_size
        self.linear_layers = linear_layers
        self.linear_neurons = linear_neurons

        # batch normalization layer
        self.BN = torch.nn.BatchNorm1d(self.linear_neurons)

        # pytorch not able to reach the parameters of linear layer inside
        # a list
        self.params = torch.nn.ParameterList()
        self.z = []

        for layer in range(self.linear_layers):
            self.z.append(torch.nn.Linear(self.input_size, self.linear_neurons))
            self.params.extend(list(self.z[layer].parameters()))

    def forward(self, input_, is_norm=False, **kwargs):
        """

        :param input_: input to maxout layer
        :param is_norm:whether to perform normalization before max operation
        :param kwargs:
        :return:
        """
        h = None
        for layer in range(self.linear_layers):
            z = self.z[layer](input_)

            # norm + norm constraint
            if is_norm:
                z =self.BN(z)

            if layer == 0:
                h = z
            else:
                h = torch.max(h, z)
        return h