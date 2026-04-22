import torch
import pytest
param = pytest.mark.parametrize

@param('cross_attend', (False, True))
def test_tile(
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

def test_multiscreen():
    from multiscreen.multiscreen import MultiScreen

    model = MultiScreen(num_tokens = 256, dim = 512, depth = 2)

    token_ids = torch.randint(0, 256, (1, 128))

    logits = model(token_ids)
    assert logits.shape == (1, 128, 256)
