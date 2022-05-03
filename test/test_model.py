import pytest
import torch

from models.vgg16_gap import *


def test_vgg16():
    model = VGG16()
    dummy_feature = torch.randn(1, 1, 224, 224)
    pred = model.forward(dummy_feature)
    assert type(pred) == torch.Tensor, \
        f"The type of pred should be Tensor, but {type(pred)}"
    assert len(pred) == 1, \
        f"The length of pred should be 1, but {len(pred)}"


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
    layer = conv_layer(in_ch, out_ch)
    dummy_feature = torch.randn(1, in_ch, 224, 224)
    pred = layer.forward(dummy_feature)
    assert type(pred) == torch.Tensor, \
        f"The type of pred should be Tensor, but {type(pred)}"
    assert len(pred[0]) == out_ch, \
        f"The len of pred[0] should be Tensor, but {len(pred[0])}"


def test_features(in_ch: int = 1, ch_num: int = 512):
    model = features(in_ch, ch_num)
    dummy_feature = torch.randn(1, in_ch, 224, 224)
    pred = model.forward(dummy_feature)
    assert type(pred) == torch.Tensor, \
        f"The type of pred should be Tensor, but {type(pred)}"
    assert len(pred[0]) == ch_num, \
        f"The len of pred[0] should be Tensor, but {len(pred[0])}"


def test_classifier(ch_num: int = 512, n_classes: int = 1):
    model = classifier(ch_num, n_classes)
    dummy_feature = torch.randn(1, ch_num, 1, 1)
    pred = model.forward(dummy_feature)
    assert type(pred) == torch.Tensor, \
        f"The type of pred should be Tensor, but {type(pred)}"
    assert len(pred) == n_classes, \
        f"The len of pred should be Tensor, but {len(pred)}"
