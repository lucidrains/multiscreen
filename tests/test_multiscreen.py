import torch
import pytest
param = pytest.mark.parametrize

@param('cross_attend', (False, True))
def test_multiscreen(
    cross_attend
):
    from multiscreen.multiscreen import MultiScreen

    model = MultiScreen(512)

    tokens = torch.randn(1, 1024, 512)

    context = None

    if cross_attend:
        context = torch.randn(1, 2048, 512)

    assert model(tokens, context = context).shape == tokens.shape
