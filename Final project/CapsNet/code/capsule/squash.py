import torch

def squash(input_tensor, epsilon=1e-7):
    """Squashes the input tensor.

    See the article "Dynamic Routing Between Capsules" by Sara Sabour, Nicholas Frosst
    and Geoffrey E. Hinton for the squashing operation.
    
    Args:
        input_tensor: The tensor to be squashed
        epsilon: A small number used for making it numerically stable. Default value is 1e-7.

    Returns:
        output_tensor: Squashed vector

    Examples:
        >>> t = torch.tensor([1.])
        >>> squash(t)
        tensor([0.5000])

        >>> t = torch.tensor([[1.,1.],[2.,3.]])
        >>> squash(t)
        tensor([[0.4714, 0.4714],
                [0.5151, 0.7726]])
    """
    squared_input = torch.sum(input_tensor*input_tensor, dim=-1, keepdim=True)
    output_tensor = (squared_input/(1.+ squared_input))*(input_tensor / torch.sqrt(squared_input + epsilon))
    return output_tensor

if __name__ == "__main__":
    import doctest
    doctest.testmod()
