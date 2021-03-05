import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy


class Model1(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()

        # conv layer parameters
        f=3
        s=1
        p=1

        # max pool parameters
        s_pool = 2
        f_pool = 2

        num_filters1 = 128
        num_filters2 = 128
        num_filters3 = 128
        num_filters4 = 128
        num_filters5 = 128
        num_filters6 = 128

        self.num_classes = num_classes
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(image_channels, num_filters1, f, s, p),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters1),

            nn.Conv2d(num_filters1, num_filters2, f, s, p),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters2),
            nn.MaxPool2d(f_pool, s_pool),

            nn.Conv2d(num_filters2, num_filters3, f, s, p),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters3),

            nn.Conv2d(num_filters3, num_filters4, f, s, p),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters4),
            nn.MaxPool2d(f_pool, s_pool),   

            nn.Conv2d(num_filters4, num_filters5, f, s, p),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters5),
            
            nn.Conv2d(num_filters5, num_filters6, f, s, p),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters6),
        )
        

        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 8*8*num_filters6
        num_nodes1 = 256

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(self.num_output_features),
            nn.Linear(self.num_output_features, num_nodes1),
            nn.ReLU(),
            nn.BatchNorm1d(num_nodes1),
            nn.Dropout(0.4),
            nn.Linear(num_nodes1, self.num_classes)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)

        batch_size = x.shape[0]

        x = self.feature_extractor(x)
        x = self.classifier(x)
        out = x
        
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out

def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


if __name__ == "__main__":
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(1)
    epochs = 20
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model = Model1(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()
    trainer.get_final_results()
    create_plots(trainer, "task3e")
