from torchvision import transforms
from ray import tune

# SRC_DIR = os.path.join(os.path.dirname(__file__))
# sys.path.append(SRC_DIR)

from src.scripts.pytorch_train import cnn_experiments



def run_experiment() -> None:
    
    # TODO: With InceptionV3, use 299x299 images,
    train_transform = transforms.Compose(
        transforms=[transforms.ToPILImage(),
                    transforms.Resize((299, 299)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(degrees=(0, 180)),
                    transforms.RandomAffine(degrees=(0, 180),
                                            translate=(0.1, 0.1),
                                            scale=(0.9, 1.1)),
                    transforms.RandomPerspective(distortion_scale=0.1,
                                                 p=0.5),
                    transforms.GaussianBlur(kernel_size=3,
                                            sigma=(0.1, 2.0)),
                    transforms.ColorJitter(brightness=0.1),
                    transforms.ToTensor(),])
    
    # TODO: With InceptionV3, use 299x299 images,
    val_transform = transforms.Compose(transforms=[transforms.ToPILImage(),
                                                   transforms.Resize((299, 299)),
                                                   transforms.ToTensor()])

    # models = [
    #         #   "vgg16",
    #         #   "resnet50",
    #         #   "densnet121",
    #         #   "mobilenetv3",
    #         #   "inceptionv3"
    #           ]
    # run_ids = [
    #         #    "7f5b919de3bc48f6aad9b2190112e204",
    #         #    "fc2829e2d037470b86a01c82adfa4d9a",
    #         #    "db5a75a0d7a247e99d422d72c02440f0",
    #         #    "5e6c14f2c3b842b083cf3747f62a5383",
    #         #    "3de150c4b9904b0692eb699aa3ef7bca"
    #            ]

    # both eyes
    models = [
        "resnet50BE",
        #   "vgg16BE",
        # "densnet121BE",
        # "mobilenetv3BE",
        # "inceptionv3BE",
        ]
    run_ids = [
        "ff83eddcce454019a80e8cd5f5137e51",
        # "f3fb85b0cb7b40a7bb899b003832566e",
        # "1744454e24f9466c8db4269b8eb20c94",
        # "4fa8be8e72834395b0c9c8b25e4ab5e5",
        # "080cf332cc6f4738b16d34acb99de559",
        ]
    
    for i, model in enumerate(models):
        # CNN config    
        config = {"model_name": model,
            "lr": tune.loguniform(1e-5, 1e-4),
            "weight_decay": tune.loguniform(1e-6, 1e-5),
            "epochs": tune.choice([10, 20, 30, 40, 50]),
            "batch_size": tune.choice([2, 4,  8, 16]),
            "pretrained": True,
            "num_classes": 2,
            "dropout": tune.uniform(0, 0.5),
            "hidden_units": tune.choice([128, 256, 512, 1024, 2048]),
            "hidden_layers": tune.choice([1, 2, 3]),
            "freeze_layers": tune.choice([0, 10, 20, 30]),
            "class_weights": [0.48702595, 1.89147287],
            "label_smoothing": tune.uniform(0, 0.2),
            "data_mode": "single"}
        
        # Debug config
        # config = {"model_name": "inceptionv3",
        #     "lr": 1e-5,
        #     "weight_decay": 1e-6,
        #     "epochs": 1, #tune.choice([10, 20, 30, 40, 50]),
        #     "batch_size": 2,
        #     "pretrained": True,
        #     "num_classes": 2,
        #     "dropout": 0.1,
        #     "hidden_units": 128,
        #     "hidden_layers": 2,
        #     "freeze_layers": 12,
        #     "class_weights": [0.48702595, 1.89147287],
        #     "label_smoothing": 0.1,
        #     "data_mode": "single"} 

        cnn_experiments(experiment_mode="kfold",
                        config=config,
                        train_transform=train_transform,
                        val_transform=val_transform,
                        run_id=run_ids[i])
    
def main():
    run_experiment()
    
if __name__ == "__main__":
    main()
    # print("Hello World!")
