from neural_tangents import stax

from typing import Callable, Optional, List

def create_mlp_stax(
    depth: int,
    hidden_width: int,
    output_dim: int,
    activation_stax_fn = stax.Relu(),
    W_std: float = 1.0,
    b_std: Optional[float] = 0.0
) -> tuple[Callable, Callable, Callable]:
    layers = []
    for _ in range(depth):
        layers.append(stax.Dense(hidden_width, W_std=W_std, b_std=b_std))
        layers.append(activation_stax_fn)
    layers.append(stax.Dense(output_dim, W_std=W_std, b_std=b_std))
    return stax.serial(*layers)

def create_mlp_stax_2(
    depth_hidden: int,
    hidden_width: int,
    output_dim: int,
    input_dim: int, 
    activation_stax_fn: Callable = stax.Relu(),
    W_std: float = 1.0,
    b_std: Optional[float] = 0.0
) -> tuple[Callable, Callable, Callable, List[int]]: 
    """
    Creates a simple MLP using neural_tangents.stax.
    Returns: init_fn, apply_fn, stax_kernel_fn, list_of_layer_widths
    """
    layers = []
    actual_widths = [input_dim]

    for _ in range(depth_hidden):
        layers.append(stax.Dense(hidden_width, W_std=W_std, b_std=b_std))
        layers.append(activation_stax_fn)
        actual_widths.append(hidden_width)
        
    layers.append(stax.Dense(output_dim, W_std=1.0, b_std=b_std)) # output layer has different W_std
    actual_widths.append(output_dim)
    
    init_fn, apply_fn, stax_kernel_fn_for_inf_limit = stax.serial(*layers)
    return init_fn, apply_fn, stax_kernel_fn_for_inf_limit, actual_widths
