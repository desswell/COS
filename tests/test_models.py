import torch

from src.models.own_cnn import OwnCNN


def test_own_cnn_forward_shape():
    model = OwnCNN(num_classes=10)
    x = torch.randn(4, 1, 64, 173)
    y = model(x)
    assert y.shape == (4, 10)


def test_own_cnn_param_count_in_expected_range():
    model = OwnCNN(num_classes=10)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # We target ~250K (see spec section 6.1); allow generous bounds
    assert 100_000 < n_params < 500_000, f"got {n_params} params"


def test_own_cnn_returns_logits_not_probs():
    model = OwnCNN(num_classes=10)
    x = torch.randn(2, 1, 64, 173)
    y = model(x)
    # Logits — should NOT sum to 1
    assert not torch.allclose(y.sum(dim=1), torch.ones(2), atol=1e-3)


def test_own_cnn_trains_one_step_without_error():
    model = OwnCNN(num_classes=10)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(2, 1, 64, 173)
    target = torch.tensor([0, 5])
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
