import torch
import pytest
param = pytest.mark.parametrize

@param('cross_attend', (False, True))
def test_multiscreen(
    cross_attend
):
    from multiscreen.multiscreen import GatedScreeningTile

    model = GatedScreeningTile(512)

    tokens = torch.randn(1, 1024, 512)

    context = context_mask = None

    if cross_attend:
        context = torch.randn(1, 2048, 512)
        context_mask = torch.randint(0, 2, (1, 2048)).bool()

    out = model(tokens, context = context, mask = context_mask)

    assert out.shape == tokens.shape
