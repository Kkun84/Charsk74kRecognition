from torch import nn, Tensor
from torchvision import models


class Model(nn.Module):
    model_name_list = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
    }

    def __init__(
        self, *, model_name: str, pretrained: bool, num_classes: int, **kwargs
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.pretrained = pretrained
        self.num_classes = num_classes

        self.model = self.model_name_list[model_name](
            pretrained=pretrained,
            num_classes=num_classes,
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        return x


if __name__ == '__main__':
    from torchinfo import summary

    for model_name in Model.model_name_list:
        model = Model(
            model_name=model_name,
            pretrained=False,
            num_classes=10,
        )
        print(model.model_name)
        summary(model, [2, 3, 100, 100])
        print()
