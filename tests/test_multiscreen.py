import torch
import pytest

def test_multiscreen():
    from multiscreen.multiscreen import MultiScreen

    model = MultiScreen(512)

    tokens = torch.randn(1, 1024, 512)

    assert model(tokens).shape == tokens.shape
