import pytest
import torch

from models import get_model
from utils import fix_seed

SEED = 123
network = "vgg16"


@pytest.fixture
def model() -> None:
    fix_seed(SEED)
    return get_model(network)


@pytest.mark.parametrize("data",[
    torch.randn(1, 1, 224, 224),
])
def test_get_model(model, data: torch.Tensor) -> None:
    pred = model(data)
    assert type(pred) == torch.Tensor, \
        f"The type of pred should be Tensor, but {type(pred)}"
    assert len(pred) == len(data), \
        f"The length of pred should be 1, but {len(pred)}"
    assert pred.dim() == 2, \
        f"The dimension of pred should be 1, but {pred.dim()}"


@pytest.mark.parametrize("data",[
    torch.randn(1, 224, 224),
])
def test_error_get_model(model, data: torch.Tensor) -> None:
    with pytest.raises(RuntimeError, match=r"Caught RuntimeError .*"):
        pred = model(data)