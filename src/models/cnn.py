"""
author: Oleksandr Kovalyk-Borodyak

This module contains a PyTorch implementation of the CNN models.
"""
import torch
from torch import Tensor
import torch.nn as nn
from typing import Tuple

class CNN(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool, hidden_units: int, hidden_layers: int, dropout: float, freeze_layers: int = 0, **kwargs) -> None:
        super(CNN, self).__init__()

        self.num_classes = num_classes
        self.pretrained = pretrained
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.dropout = dropout

        self.model, self.linear_in = self.get_base_model(pretrained=pretrained, **kwargs)
        print(f"Model {self.__class__.__name__} nº layers: {len(list(self.model.parameters()))}")

        # Freeze layers
        if freeze_layers > 0:
            assert len(list(self.model.parameters())) >= freeze_layers, "freeze_layers must be less than the number of layers in the model"
            for i, param in enumerate(self.model.parameters()):
                if i < freeze_layers:
                    param.requires_grad = False

        self._dropout = nn.Dropout(dropout)
        self._hidden_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout)
            ) for _ in range(hidden_layers - 1)])
        self._output = nn.Linear(hidden_units, num_classes)

    def get_base_model(self, **kwargs) -> Tuple[nn.Module, nn.Sequential]:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        x = self._dropout(x)
        x = self.linear_in(x)
        x = self._dropout(x)
        for hidden_layer in self._hidden_layers:
            x = hidden_layer(x)
        x = self._output(x)
        return x

class CNNBothEyes(CNN):

    def __init__(self, num_classes: int, pretrained: bool, hidden_units: int, hidden_layers: int, dropout: float, freeze_layers: int = 0, **kwargs) -> None:
        super(CNNBothEyes, self).__init__(num_classes,
                                       pretrained,
                                       hidden_units,
                                       hidden_layers,
                                       dropout,
                                       freeze_layers)

    def get_base_model(self, **kwargs) -> Tuple[nn.Module, nn.Sequential]:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:

        outputs = []
        for i in range(2):
            output = self.model(x[:, i, ...].view(-1, 3, 299, 299))
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        outputs = self._dropout(outputs)
        outputs = self.linear_in(outputs)
        for hidden_layer in self._hidden_layers:
            outputs = hidden_layer(outputs)

        outputs = self._output(outputs)
        return outputs

class ResNet18BothEyes(CNNBothEyes):

    def __init__(self, num_classes: int, pretrained: bool, hidden_units: int, hidden_layers: int, dropout: float, freeze_layers: int = 0, **kwargs) -> None:
        super().__init__(num_classes, pretrained, hidden_units, hidden_layers, dropout, freeze_layers, **kwargs)

    def get_base_model(self, **kwargs) -> Tuple[nn.Module, nn.Sequential]:

        from torchvision.models import resnet18

        model = resnet18(**kwargs)
        model.fc = nn.Identity() # type: ignore
        linear_in = nn.Sequential(nn.Linear(512 * 2, self.hidden_units), nn.ReLU())
        return model, linear_in

class ResNet50BothEyes(CNNBothEyes):

    def __init__(self, num_classes: int, pretrained: bool, hidden_units: int, hidden_layers: int, dropout: float, freeze_layers: int = 0, **kwargs) -> None:
        super().__init__(num_classes, pretrained, hidden_units, hidden_layers, dropout, freeze_layers, **kwargs)

    def get_base_model(self, **kwargs) -> Tuple[nn.Module, nn.Sequential]:

        from torchvision.models import resnet50

        model = resnet50(**kwargs)
        model.fc = nn.Identity() # type: ignore
        linear_in = nn.Sequential(nn.Linear(2048 * 2, self.hidden_units), nn.ReLU())
        return model, linear_in

class DenseNet121BothEyes(CNNBothEyes):

    def __init__(self, num_classes: int, pretrained: bool, hidden_units: int, hidden_layers: int, dropout: float, freeze_layers: int = 0, **kwargs) -> None:
        super().__init__(num_classes, pretrained, hidden_units, hidden_layers, dropout, freeze_layers, **kwargs)

    def get_base_model(self, **kwargs) -> Tuple[nn.Module, nn.Sequential]:

        from torchvision.models import densenet121

        model = densenet121(**kwargs)
        model.classifier = nn.Identity() # type: ignore
        linear_in = nn.Sequential(nn.Linear(1024 * 2, self.hidden_units), nn.ReLU())
        return model, linear_in

class InceptionV3BothEyes(CNNBothEyes):

    def __init__(self, num_classes: int, pretrained: bool, hidden_units: int, hidden_layers: int, dropout: float, freeze_layers: int = 0, **kwargs) -> None:
        super().__init__(num_classes, pretrained, hidden_units, hidden_layers, dropout, freeze_layers, **kwargs)

    def get_base_model(self, **kwargs) -> Tuple[nn.Module, nn.Sequential]:

        from torchvision.models import inception_v3
        model = inception_v3(**kwargs)
        model.fc = nn.Identity() # type: ignore
        linear_in = nn.Sequential(nn.Linear(2048 * 2, self.hidden_units), nn.ReLU())
        return model, linear_in

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            outputs = []
            for i in range(2):
                output = self.model(x[:, i, ...].view(-1, 3, 299, 299))[0]
                outputs.append(output)

            outputs = torch.cat(outputs, dim=1)

            outputs = self._dropout(outputs)


            outputs = self.linear_in(outputs)
            for hidden_layer in self._hidden_layers:
                outputs = hidden_layer(outputs)

            outputs = self._output(outputs)
            return outputs
        else:
            outputs = []
            for i in range(2):
                output = self.model(x[:, i, ...].view(-1, 3, 299, 299))
                outputs.append(output)

            outputs = torch.cat(outputs, dim=1)
            outputs = self._dropout(outputs)
            outputs = self.linear_in(outputs)
            for hidden_layer in self._hidden_layers:
                outputs = hidden_layer(outputs)

            outputs = self._output(outputs)
            return outputs

class MobileNetV3BothEyes(CNNBothEyes):

    def __init__(self, num_classes: int, pretrained: bool, hidden_units: int, hidden_layers: int, dropout: float, freeze_layers: int = 0, **kwargs) -> None:
        super().__init__(num_classes, pretrained, hidden_units, hidden_layers, dropout, freeze_layers, **kwargs)

    def get_base_model(self, **kwargs) -> Tuple[nn.Module, nn.Sequential]:

        from torchvision.models import mobilenet_v3_small
        model = mobilenet_v3_small(**kwargs)
        model.classifier = nn.Identity() # type: ignore
        linear_in = nn.Sequential(nn.Linear(576 * 2, self.hidden_units), nn.ReLU())
        return model, linear_in

class VGG16BothEyes(CNNBothEyes):

    def __init__(self, num_classes: int, pretrained: bool, hidden_units: int, hidden_layers: int, dropout: float, freeze_layers: int = 0, **kwargs) -> None:
        super().__init__(num_classes, pretrained, hidden_units, hidden_layers, dropout, freeze_layers, **kwargs)

    def get_base_model(self, **kwargs) -> Tuple[nn.Module, nn.Sequential]:

        from torchvision.models import vgg16_bn
        model = vgg16_bn(**kwargs)
        model.classifier = nn.Identity() # type: ignore
        linear_in = nn.Sequential(nn.Linear(512 * 7 * 7 * 2, self.hidden_units), nn.ReLU())
        return model, linear_in

class ResNet18(CNN):

    def __init__(self,
                 num_classes: int,
                 pretrained: bool,
                 hidden_units: int,
                 hidden_layers: int,
                 dropout: float,
                 freeze_layers: int = 0) -> None:

        super(ResNet18, self).__init__(num_classes,
                                       pretrained,
                                       hidden_units,
                                       hidden_layers,
                                       dropout,
                                       freeze_layers)

    def get_base_model(self, **kwargs) -> Tuple[nn.Module, nn.Sequential]:
        from torchvision.models import resnet18
        model = resnet18(**kwargs)
        model.fc = nn.Identity() # type: ignore
        linear_in = nn.Sequential(nn.Linear(512, self.hidden_units), nn.ReLU())
        return model, linear_in

class ResNet50(CNN):

    def __init__(self, num_classes: int, pretrained: bool, hidden_units: int, hidden_layers: int, dropout: float, freeze_layers: int = 0, **kwargs) -> None:
        super().__init__(num_classes, pretrained, hidden_units, hidden_layers, dropout, freeze_layers, **kwargs)
    def get_base_model(self, **kwargs) -> Tuple[nn.Module, nn.Sequential]:
        from torchvision.models import resnet50
        model = resnet50(**kwargs)
        model.fc = nn.Identity() # type: ignore
        # linear_in = nn.Linear(2048, self.hidden_units)
        linear_in = nn.Sequential(nn.Linear(2048, self.hidden_units), nn.ReLU())
        return model, linear_in

class DenseNet121(CNN):

    def __init__(self, num_classes: int, pretrained: bool, hidden_units: int, hidden_layers: int, dropout: float, freeze_layers: int = 0, **kwargs) -> None:
        super().__init__(num_classes, pretrained, hidden_units, hidden_layers, dropout, freeze_layers, **kwargs)

    def get_base_model(self, **kwargs) -> Tuple[nn.Module, nn.Sequential]:
        from torchvision.models import densenet121
        model = densenet121(**kwargs)
        model.classifier = nn.Identity() # type: ignore
        linear_in = nn.Sequential(nn.Linear(1024, self.hidden_units), nn.ReLU()) # nn.Linear(1024, self.hidden_units)
        return model, linear_in

class InceptionV3(CNN):

    def __init__(self, num_classes: int, pretrained: bool, hidden_units: int, hidden_layers: int, dropout: float, freeze_layers: int = 0, **kwargs) -> None:
        super().__init__(num_classes, pretrained, hidden_units, hidden_layers, dropout, freeze_layers, **kwargs)

    def get_base_model(self, **kwargs) -> Tuple[nn.Module, nn.Sequential]:

        from torchvision.models import inception_v3
        model = inception_v3(**kwargs)
        # model.dropout = nn.Identity()
        model.fc = nn.Identity() # type: ignore
        linear_in = nn.Sequential(nn.Linear(2048, self.hidden_units), nn.ReLU()) # nn.Linear(2048, self.hidden_units)
        return model, linear_in

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            x = self.model(x)[0]
            # x = self.dropout(x)
            x = self.linear_in(x)
            x = self._dropout(x)
            for hidden_layer in self._hidden_layers:
                x = hidden_layer(x)
            x = self._output(x)
            return x
        else:
            x = self.model(x)
            # x = self.dropout(x)
            x = self.linear_in(x)
            x = self._dropout(x)
            for hidden_layer in self._hidden_layers:
                x = hidden_layer(x)
            x = self._output(x)
            return x

class MobileNetV3(CNN):

    def __init__(self, num_classes: int, pretrained: bool, hidden_units: int, hidden_layers: int, dropout: float, freeze_layers: int = 0, **kwargs) -> None:
        super().__init__(num_classes, pretrained, hidden_units, hidden_layers, dropout, freeze_layers, **kwargs)

    def get_base_model(self, **kwargs) -> Tuple[nn.Module, nn.Sequential]:

        from torchvision.models import mobilenet_v3_small

        model = mobilenet_v3_small(**kwargs)
        model.classifier = nn.Identity() # type: ignore
        linear_in = nn.Sequential(nn.Linear(576, self.hidden_units), nn.ReLU())
        # linear_in = nn.Linear(576, self.hidden_units)
        return model, linear_in

class VGG16(CNN):

    def __init__(self, num_classes: int, pretrained: bool, hidden_units: int, hidden_layers: int, dropout: float, freeze_layers: int = 0, **kwargs) -> None:
        super().__init__(num_classes, pretrained, hidden_units, hidden_layers, dropout, freeze_layers, **kwargs)

    def get_base_model(self, **kwargs) -> Tuple[nn.Module, nn.Sequential]:

        from torchvision.models import vgg16_bn

        model = vgg16_bn(**kwargs)
        model.classifier = nn.Identity() # type: ignore
        linear_in = nn.Sequential(nn.Linear(512 * 7 * 7, self.hidden_units), nn.ReLU())
        return model, linear_in

if __name__ == "__main__":
    from torchinfo import summary
    model = ResNet50(2, True, 512, 2, 0.5)
    torch_info = summary(model=model, input_size=(1, 3, 224, 224), device="cpu", verbose=1)
