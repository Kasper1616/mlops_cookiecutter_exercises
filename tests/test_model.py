import pytest
import torch

from s2_cookiecutter.model import MyAwesomeModel


@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(batch_size: int) -> None:
    """Test the model."""
    model = MyAwesomeModel()
    data = torch.randn(batch_size, 1, 28, 28)
    output = model(data)
    assert output.shape == (batch_size, 10), f"Bad output shape: {output.shape}"


# def test_error_on_wrong_shape():
#     model = MyAwesomeModel()
#     with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
#         model(torch.randn(1,2,3,4))
#     with pytest.raises(ValueError, match='Expected each sample to have shape [1, 28, 28]'):
#         model(torch.randn(1,1,28,28))