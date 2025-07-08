from src import const


def test_forward_pass():
    from src.model import AlexNet
    import torch

    image_size = (
        const.FINAL_IMAGE_SIZE_W,
        const.FINAL_IMAGE_SIZE_H,
    )  # Example input size
    model = AlexNet(input_shape=image_size)
    input_tensor = torch.randn(
        1, 1, image_size[0], image_size[1]
    )  # Example input tensor
    output = model(input_tensor)

    assert output is not None, "Model should return an output"
