import pytest
import torch

from models.vgg16_gap import VGG16, Classifier, ConvLayer, Features


@pytest.mark.parametrize("data", [
    torch.randn(1, 1, 224, 224),
    torch.randn(1, 224, 224),
])
def test_vgg16(data: torch.Tensor) -> None:
    """ test VGG16
    Args:
        data (torch.Tensor): input data
    """
    model = VGG16()
    assert type(model) == VGG16, \
        f"Should be VGG16, but {type(model)}"
    
    try:
        pred = model.forward(data)
        assert type(pred) == torch.Tensor, \
            f"The type of pred should be Tensor, but {type(pred)}"
        assert len(pred) == 1, \
            f"The length of pred should be 1, but {len(pred)}"
    except:
        with pytest.raises(RuntimeError, match=r"Expected 4-dimensional input"):
            _ = model.forward(data)


@pytest.mark.parametrize('in_ch, out_ch', [
    (1, 64),
    (64, 64),
    (64, 128),
    (128, 128),
    (128, 256),
    (256, 256),
    (256, 512),
    (512, 512),
  ])
def test_conv_layer(in_ch: int, out_ch: int):
    """ test ConvLayer
    Args:
        in_ch (int): input channel size
        out_ch (int): output channel size
    """
    layer = ConvLayer(in_ch, out_ch)
    dummy_feature = torch.randn(1, in_ch, 224, 224)
    pred = layer.forward(dummy_feature)
    assert type(pred) == torch.Tensor, \
        f"The type of pred should be Tensor, but {type(pred)}"
    assert len(pred[0]) == out_ch, \
        f"The len of pred[0] should be Tensor, but {len(pred[0])}"


@pytest.mark.parametrize("in_ch, ch_num", [
    (1, 512)
])
def test_features(in_ch: int, ch_num: int) -> None:
    """ test Features
    Args:
        in_ch (int): input channel size
        ch_num (int): hidden channel size
    """
    model = Features(in_ch, ch_num)
    dummy_feature = torch.randn(1, in_ch, 224, 224)
    pred = model.forward(dummy_feature)
    assert type(pred) == torch.Tensor, \
        f"The type of pred should be Tensor, but {type(pred)}"
    assert len(pred[0]) == ch_num, \
        f"The len of pred[0] should be Tensor, but {len(pred[0])}"


@pytest.mark.parametrize("ch_num, n_classes", [
    (512, 1)
])
def test_classifier(ch_num: int, n_classes: int) -> None:
    """ test Classifier
    Args:
        ch_num (int): hidden channel size
        n_classes (int): output channel size
    """
    model = Classifier(ch_num, n_classes)
    dummy_feature = torch.randn(1, ch_num, 1, 1)
    pred = model.forward(dummy_feature)
    assert type(pred) == torch.Tensor, \
        f"The type of pred should be Tensor, but {type(pred)}"
    assert len(pred) == n_classes, \
        f"The len of pred should be Tensor, but {len(pred)}"
