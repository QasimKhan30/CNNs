from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]

class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        return nn.functional.cross_entropy(logits,target)

class Classifier(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        """
        Dimensions through each layer
        
        """
        
        """
        Increasing the number of channels afther the first layer for each convolution increases the
        computational cost of the convolution
        
        
        
        You should only really increase output channels if you stride (increase by striding factor).
        
        This will keep computation cost constant
        cost is almost spread out evenly between all layers
        
        Width and height after striding will be divided by the stride and then input and output channels
        will be multipied by stride so it cancels out
        """

        cnn_layers = [
            
            # First Convolutional layer
            # Input Dimensions: (B,3,64,64)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
            # Dimensions after MaxPool: (B,32,32,32)

            # Second Convolutional layer
            # Input Dimensions: (B,32,32,32)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            # Dimensions after MaxPool: (B,64,16,16)

            # Third Convolutional layer
            # Input Dimensions: (B,64,16,16)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
            # Dimensions after MaxPool: (B,128,8,8)
            
            # Fourth Convolutional layer
            # Input Dimensions: (B,128,8,8)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8 -> 4
            # Dimensions after MaxPool: (B,256,4,4)
            nn.Dropout2d(0.4),
            
            # Flattened dimensions = (256 * 4 * 4)
            nn.Flatten(),
            
            # Map the 4096 (256 * 4 * 4) features to the number of classes
            nn.Linear(4096, num_classes)
            
            # We will be left with a set of logits which represent the scores for each class
        ]
        
        self.network = nn.Sequential(*cnn_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input        
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        logits = self.network(z)

        return logits
        

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)



class Detector(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))
        
        
        # Encoding section
        
        # Strided conv layer 1
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),  # (B,16,H/2,W/2)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        # Strided conv layer 2
        self.down2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),           # (B,32,H/4,W/4)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # Strided conv layer 3
        self.down3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),           # (B,64,H/8,W/8)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Skip connections - Increases channels so that it can be used in the decoding section (matches channels of decoding layers)
        self.skip2 = nn.Conv2d(32, 64, kernel_size=1)  # transforms d2 (B,32,H/4,W/4) -> (B,64,H/4,W/4)
        self.skip1 = nn.Conv2d(16, 32, kernel_size=1)  # transforms d1 (B,16,H/2,W/2) -> (B,32,H/2,W/2)
        
        # Bottleneck / end of downsampling
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),                     # (B,128,H/8,W/8)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Decoding Section

        # Up conv layer 1
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B,64,H/4,W/4)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Up conv layer 2
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # (B,32,H/2,W/2)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Up conv layer 3
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),   # (B,16,H,W)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Heads
        self.seg_head = nn.Conv2d(16, num_classes, kernel_size=1)
        self.depth_head = nn.Conv2d(16, 1, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        
        # Encoder:
        d1 = self.down1(z)   # (B,16,H/2,W/2)
        d2 = self.down2(d1)  # (B,32,H/4,W/4)
        d3 = self.down3(d2)  # (B,64,H/8,W/8)
        b = self.bottleneck(d3)  # (B,128,H/8,W/8)
        
        # Decoder with skip connections:
        
        # Up1: (B,128,H/8,W/8) -> (B,64,H/4,W/4)
        u1 = self.up1(b)     # (B,64,H/4,W/4)
        skip2 = self.skip2(d2)  # (B,64,H/4,W/4)
        u1 = u1 + skip2         # Skip connection from down2
        
        # Up2: (B,64,H/4,W/4) -> (B,32,H/2,W/2)
        u2 = self.up2(u1)    # (B,32,H/2,W/2)
        skip1 = self.skip1(d1)  # (B,32,H/2,W/2)
        u2 = u2 + skip1         # Skip connection from down1
        
        # Up3 (No skip connection on final layer): (B,32,H/2,W/2) -> (B,16,H,W)
        u3 = self.up3(u2)    # (B,16,H,W)
        
        # Heads: segmentation and depth
        seg_logits = self.seg_head(u3)  # (B, num_classes, H, W)
        depth = self.depth_head(u3)     # (B, 1, H, W)
        depth = torch.sigmoid(depth)    # constrain values of depth to 0 or 1
        depth = depth.squeeze(1)        # (B, H, W)
        return seg_logits, depth


    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth

        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()