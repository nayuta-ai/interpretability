import pytest
import torch.nn as nn

from models import get_model


@pytest.mark.parametrize("model_name", [
    "vgg16",
    "vgg",
])
def test_get_model(model_name: str) -> None:
    """ test get_model
    Args:
        model_name (str): name of model
    """
    try:
        model = get_model(model_name)
        assert isinstance(model, nn.Module), \
            f"Should nn.Module, but {type(model)}"
    except:
        with pytest.raises(
            ValueError, match=f"The {model_name} name doesn't exist."
        ):
            _ = get_model(model_name)
