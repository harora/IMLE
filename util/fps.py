import numpy
from chainer import cuda
import torch

def l2_norm(x, y):
    """Calculate l2 norm (distance) of `x` and `y`.
    Args:
        x (numpy.ndarray or cupy): (batch_size, num_point, coord_dim)
        y (numpy.ndarray): (batch_size, num_point, coord_dim)
    Returns (numpy.ndarray): (batch_size, num_point,)
    """
    return ((x - y) ** 2).sum(axis=2)

def farthest_point_sampling(pts, k, initial_idx=None, metrics=l2_norm,
                            skip_initial=False, indices_dtype=numpy.int32,
                            distances_dtype=numpy.float32):
    """Batch operation of farthest point sampling
    Code referenced from below link by @Graipher
    https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python
    Args:
        pts (numpy.ndarray or cupy.ndarray): 2-dim array (num_point, coord_dim)
            or 3-dim array (batch_size, num_point, coord_dim)
            When input is 2-dim array, it is treated as 3-dim array with
            `batch_size=1`.
        k (int): number of points to sample
        initial_idx (int): initial index to start farthest point sampling.
            `None` indicates to sample from random index,
            in this case the returned value is not deterministic.
        metrics (callable): metrics function, indicates how to calc distance.
        skip_initial (bool): If True, initial point is skipped to store as
            farthest point. It stabilizes the function output.
        xp (numpy or cupy):
        indices_dtype (): dtype of output `indices`
        distances_dtype (): dtype of output `distances`
    Returns (tuple): `indices` and `distances`.
        indices (numpy.ndarray or cupy.ndarray): 2-dim array (batch_size, k, )
            indices of sampled farthest points.
            `pts[indices[i, j]]` represents `i-th` batch element of `j-th`
            farthest point.
        distances (numpy.ndarray or cupy.ndarray): 3-dim array
            (batch_size, k, num_point)
    """
    
    pts = pts[numpy.newaxis ,:, :]
    
    ndim = pts.shape[2]
    if ndim == 2:
        # insert batch_size axis
        pts = pts[None, ...]
    assert ndim == 3
    xp = cuda.get_array_module(pts)
    batch_size, num_point, coord_dim = pts.shape
    indices = xp.zeros((batch_size, k, ), dtype=indices_dtype)

    # distances[bs, i, j] is distance between i-th farthest point `pts[bs, i]`
    # and j-th input point `pts[bs, j]`.
    distances = xp.zeros((batch_size, k, num_point), dtype=distances_dtype)
    if initial_idx is None:
        indices[:, 0] = xp.random.randint(len(pts))
    else:
        indices[:, 0] = initial_idx

    batch_indices = xp.arange(batch_size)
    farthest_point = pts[batch_indices, indices[:, 0]]
    # minimum distances to the sampled farthest point
    try:
        min_distances = metrics(farthest_point[:, None, :], pts)
    except Exception as e:
        import IPython; IPython.embed()

    if skip_initial:
        # Override 0-th `indices` by the farthest point of `initial_idx`
        indices[:, 0] = xp.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, 0]]
        min_distances = metrics(farthest_point[:, None, :], pts)

    distances[:, 0, :] = min_distances
    for i in range(1, k):
        indices[:, i] = xp.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, i]]
        dist = metrics(farthest_point[:, None, :], pts)
        distances[:, i, :] = dist
        min_distances = xp.minimum(min_distances, dist)

    pts = pts[:,indices,:]
    return pts[0][0]

def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)