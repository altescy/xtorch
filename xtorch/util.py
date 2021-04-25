from typing import Optional, Union, cast

import torch


def tiny_value_of_dtype(dtype: torch.dtype) -> float:
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype in (torch.float, torch.double):
        return 1e-13
    if dtype == torch.half:
        return 1e-4
    raise TypeError("Does not support dtype " + str(dtype))


def info_value_of_dtype(dtype: torch.dtype) -> Union[torch.finfo, torch.iinfo]:
    if dtype == torch.bool:
        raise TypeError("Does not support torch.bool")
    if dtype.is_floating_point:
        return torch.finfo(dtype)
    return torch.iinfo(dtype)


def min_value_of_dtype(dtype: torch.dtype) -> float:
    return info_value_of_dtype(dtype).min


def masked_softmax(
    vector: torch.Tensor,
    mask: Optional[torch.BoolTensor],
    dim: int = -1,
    memory_efficient: bool = False,
) -> torch.Tensor:
    if mask is None:
        return torch.nn.functional.softmax(vector, dim=dim)

    while mask.dim() < vector.dim():
        mask = cast(torch.BoolTensor, mask.unsqueeze(1))

    if not memory_efficient:
        result = torch.nn.functional.softmax(vector * mask, dim=dim)
        result = result * mask
        result = result / (
            result.sum(dim=dim, keepdim=True) + tiny_value_of_dtype(result.dtype)
        )
    else:
        masked_vector = vector.masked_fill(~mask, min_value_of_dtype(vector.dtype))
        result = torch.nn.functional.softmax(masked_vector, dim=dim)

    return result
