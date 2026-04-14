"""
blocks.py
Block (node type) definitions and lookup helpers.
"""

from typing import Optional

SECTIONS: dict = {
    "Model Creation": {
        "Layers": [
            {
                "label": "Linear", "color": (100, 180, 255),
                "params": ["in_features", "out_features"],
                "inputs": ["x"], "outputs": ["out"],
                "defaults": {},
                "tooltip": "Fully-connected layer - every input connects to every output.",
                "when_to_use": "Use after Flatten to classify or transform flat feature vectors.",
            },
            {
                "label": "Conv2D", "color": (120, 220, 140),
                "params": ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
                "inputs": ["x"], "outputs": ["out"],
                "defaults": {"kernel_size": "3", "stride": "1", "padding": "1"},
                "tooltip": "2D convolution - slides a small filter over an image to detect features.",
                "when_to_use": "Use as the main building block for image models (CNNs).",
            },
            {
                "label": "ConvTranspose2D", "color": (120, 220, 140),
                "params": ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
                "inputs": ["x"], "outputs": ["out"],
                "defaults": {"kernel_size": "3", "stride": "1", "padding": "0"},
                "tooltip": "Upsampling convolution - increases spatial resolution.",
                "when_to_use": "Use in decoders and segmentation models to upsample feature maps.",
            },
            {
                "label": "Flatten", "color": (100, 180, 255),
                "params": ["start_dim", "end_dim"],
                "inputs": ["x"], "outputs": ["out"],
                "defaults": {"start_dim": "1", "end_dim": "-1"},
                "tooltip": "Collapses spatial dimensions into a single flat vector.",
                "when_to_use": "Place between the last Conv/Pool layer and the first Linear layer.",
            },
        ],
        "Activations": [
            {
                "label": "ReLU", "color": (255, 180, 80),
                "params": [], "inputs": ["x"], "outputs": ["out"], "defaults": {},
                "tooltip": "Replaces negative values with zero - the most common activation.",
                "when_to_use": "Default choice after Linear and Conv2D layers.",
            },
            {
                "label": "Sigmoid", "color": (255, 180, 80),
                "params": [], "inputs": ["x"], "outputs": ["out"], "defaults": {},
                "tooltip": "Squashes values to (0, 1) - outputs a probability.",
                "when_to_use": "Use on the final layer for binary classification tasks.",
            },
            {
                "label": "Tanh", "color": (255, 180, 80),
                "params": [], "inputs": ["x"], "outputs": ["out"], "defaults": {},
                "tooltip": "Squashes values to (-1, 1) - zero-centred sigmoid.",
                "when_to_use": "Common in RNNs and GANs; less used in modern feedforward nets.",
            },
            {
                "label": "GELU", "color": (255, 180, 80),
                "params": [], "inputs": ["x"], "outputs": ["out"], "defaults": {},
                "tooltip": "Smooth activation used in transformers and BERT-style models.",
                "when_to_use": "Use instead of ReLU in transformer or attention-based architectures.",
            },
            {
                "label": "Softmax", "color": (255, 180, 80),
                "params": ["dim"],
                "inputs": ["x"], "outputs": ["out"],
                "defaults": {"dim": "1"},
                "tooltip": "Converts raw scores to probabilities that sum to 1.",
                "when_to_use": "Not needed with CrossEntropyLoss - only add if using NLLLoss.",
            },
            {
                "label": "LeakyReLU", "color": (255, 180, 80),
                "params": ["negative_slope"],
                "inputs": ["x"], "outputs": ["out"],
                "defaults": {"negative_slope": "0.01"},
                "tooltip": "Like ReLU but allows a small gradient for negative values.",
                "when_to_use": "Use when you see dead neuron problems, or as the activation in GANs.",
            },
        ],
        "Normalization": [
            {
                "label": "BatchNorm2D", "color": (200, 130, 255),
                "params": ["num_features", "eps", "momentum"],
                "inputs": ["x"], "outputs": ["out"],
                "defaults": {"eps": "1e-5", "momentum": "0.1"},
                "tooltip": "Normalises activations across the batch - stabilises and speeds up training.",
                "when_to_use": "Place after Conv2D, before activation. num_features = out_channels of the previous Conv.",
            },
            {
                "label": "LayerNorm", "color": (200, 130, 255),
                "params": ["normalized_shape", "eps"],
                "inputs": ["x"], "outputs": ["out"],
                "defaults": {"eps": "1e-5"},
                "tooltip": "Normalises activations across features of a single sample.",
                "when_to_use": "Use in transformer models. normalized_shape = the last dimension size.",
            },
            {
                "label": "GroupNorm", "color": (200, 130, 255),
                "params": ["num_groups", "num_channels"],
                "inputs": ["x"], "outputs": ["out"],
                "defaults": {},
                "tooltip": "Normalises within groups of channels - works with small batch sizes.",
                "when_to_use": "Use instead of BatchNorm when batch size is 1 or 2.",
            },
            {
                "label": "Dropout", "color": (200, 130, 255),
                "params": ["p"],
                "inputs": ["x"], "outputs": ["out"],
                "defaults": {"p": "0.5"},
                "tooltip": "Randomly zeros out neurons during training to prevent overfitting.",
                "when_to_use": "Place before the final Linear layer. p=0.5 is a good start; reduce if underfitting.",
            },
        ],
        "Pooling": [
            {
                "label": "MaxPool2D", "color": (255, 120, 120),
                "params": ["kernel_size", "stride", "padding"],
                "inputs": ["x"], "outputs": ["out"],
                "defaults": {"kernel_size": "2", "stride": "2", "padding": "0"},
                "tooltip": "Halves spatial size by keeping the maximum value in each patch.",
                "when_to_use": "Use after Conv2D to reduce resolution. kernel_size=2, stride=2 halves H and W.",
            },
            {
                "label": "AvgPool2D", "color": (255, 120, 120),
                "params": ["kernel_size", "stride", "padding"],
                "inputs": ["x"], "outputs": ["out"],
                "defaults": {"kernel_size": "2", "stride": "2", "padding": "0"},
                "tooltip": "Halves spatial size by averaging values in each patch.",
                "when_to_use": "Similar to MaxPool2D but smoother. Less common in classification models.",
            },
            {
                "label": "AdaptiveAvgPool2D", "color": (255, 120, 120),
                "params": ["output_size"],
                "inputs": ["x"], "outputs": ["out"],
                "defaults": {"output_size": "1"},
                "tooltip": "Pools to a fixed output size regardless of input resolution.",
                "when_to_use": "Use before a Linear layer to handle variable input sizes. output_size=1 is global average pooling.",
            },
        ],
        "I/O": [
            {
                "label": "Input", "color": (80, 220, 200),
                "params": ["shape"], "inputs": [], "outputs": ["out"], "defaults": {},
                "tooltip": "Defines the shape of one input sample entering the model.",
                "when_to_use": "Every model needs exactly one. Shape is auto-filled from your dataset.",
            },
            {
                "label": "Output", "color": (80, 220, 200),
                "params": ["shape"], "inputs": ["x"], "outputs": [], "defaults": {},
                "tooltip": "Marks the end of the model and its output shape.",
                "when_to_use": "Every model needs exactly one. Shape = number of classes for classification.",
            },
        ],
    },
    "Training": {
        "Pipeline Inputs": [
            {
                "label": "ModelBlock", "color": (80, 180, 255),
                "params": [], "inputs": ["images"], "outputs": ["predictions"], "defaults": {},
                "tooltip": "Represents your Model tab - forwards batches through it during training.",
                "when_to_use": "Required. Wire DataLoaderBlock.images into its images input.",
            },
            {
                "label": "DataLoaderBlock", "color": (180, 100, 255),
                "params": [], "inputs": [], "outputs": ["images", "labels"], "defaults": {},
                "tooltip": "Represents your Data Prep tab - supplies batches of images and labels.",
                "when_to_use": "Required. Wire images to ModelBlock and labels to your Loss node.",
            },
        ],
        "Loss Functions": [
            {
                "label": "CrossEntropyLoss", "color": (255, 160, 100),
                "params": ["weight", "ignore_index", "reduction"],
                "inputs": ["pred", "target"], "outputs": ["loss"],
                "defaults": {"reduction": "mean"},
                "tooltip": "Standard loss for multi-class classification.",
                "when_to_use": "Default choice for classification. Do not add Softmax to your model when using this.",
            },
            {
                "label": "MSELoss", "color": (255, 160, 100),
                "params": ["reduction"],
                "inputs": ["pred", "target"], "outputs": ["loss"],
                "defaults": {"reduction": "mean"},
                "tooltip": "Mean Squared Error - penalises large prediction errors heavily.",
                "when_to_use": "Use for regression tasks where the output is a continuous number.",
            },
            {
                "label": "BCELoss", "color": (255, 160, 100),
                "params": ["reduction"],
                "inputs": ["pred", "target"], "outputs": ["loss"],
                "defaults": {"reduction": "mean"},
                "tooltip": "Binary Cross-Entropy - requires Sigmoid on the model output.",
                "when_to_use": "Use for binary classification. Your model's last layer must be Sigmoid.",
            },
            {
                "label": "BCEWithLogits", "color": (255, 160, 100),
                "params": ["reduction"],
                "inputs": ["pred", "target"], "outputs": ["loss"],
                "defaults": {"reduction": "mean"},
                "tooltip": "BCE with built-in Sigmoid - more numerically stable than BCELoss.",
                "when_to_use": "Preferred over BCELoss. Do not add Sigmoid to your model when using this.",
            },
            {
                "label": "NLLLoss", "color": (255, 160, 100),
                "params": ["reduction"],
                "inputs": ["pred", "target"], "outputs": ["loss"],
                "defaults": {"reduction": "mean"},
                "tooltip": "Negative Log-Likelihood - requires log-probabilities as input.",
                "when_to_use": "Use with a LogSoftmax output. CrossEntropyLoss is usually simpler.",
            },
            {
                "label": "HuberLoss", "color": (255, 160, 100),
                "params": ["delta", "reduction"],
                "inputs": ["pred", "target"], "outputs": ["loss"],
                "defaults": {"delta": "1.0", "reduction": "mean"},
                "tooltip": "Regression loss that is less sensitive to outliers than MSELoss.",
                "when_to_use": "Use for regression when your targets contain outliers.",
            },
            {
                "label": "KLDivLoss", "color": (255, 160, 100),
                "params": ["reduction"],
                "inputs": ["pred", "target"], "outputs": ["loss"],
                "defaults": {"reduction": "mean"},
                "tooltip": "Measures divergence between two probability distributions.",
                "when_to_use": "Use in VAEs or knowledge distillation. Requires log-probability inputs.",
            },
        ],
        "Optimizers": [
            {
                "label": "Adam", "color": (100, 220, 180),
                "params": ["lr", "betas", "eps", "weight_decay"],
                "inputs": ["params"], "outputs": [],
                "defaults": {"lr": "0.001", "betas": "0.9, 0.999", "eps": "1e-8", "weight_decay": "0.0"},
                "tooltip": "Adaptive optimizer - adjusts each parameter's learning rate automatically.",
                "when_to_use": "Best default choice. Start with lr=0.001 and adjust if loss does not decrease.",
            },
            {
                "label": "AdamW", "color": (100, 220, 180),
                "params": ["lr", "betas", "eps", "weight_decay"],
                "inputs": ["params"], "outputs": [],
                "defaults": {"lr": "0.001", "betas": "0.9, 0.999", "eps": "1e-8", "weight_decay": "0.01"},
                "tooltip": "Adam with decoupled weight decay - better regularisation than plain Adam.",
                "when_to_use": "Preferred over Adam when you want regularisation. Good for transformers.",
            },
            {
                "label": "SGD", "color": (100, 220, 180),
                "params": ["lr", "momentum", "weight_decay"],
                "inputs": ["params"], "outputs": [],
                "defaults": {"lr": "0.01", "momentum": "0.9", "weight_decay": "0.0"},
                "tooltip": "Classic gradient descent with optional momentum.",
                "when_to_use": "Good for CNNs with a learning rate schedule. Needs careful lr tuning.",
            },
            {
                "label": "RMSprop", "color": (100, 220, 180),
                "params": ["lr", "alpha", "eps", "weight_decay"],
                "inputs": ["params"], "outputs": [],
                "defaults": {"lr": "0.01", "alpha": "0.99", "eps": "1e-8", "weight_decay": "0.0"},
                "tooltip": "Adaptive optimizer that divides gradients by a running average.",
                "when_to_use": "Works well for RNNs. Less common for image classification.",
            },
            {
                "label": "Adagrad", "color": (100, 220, 180),
                "params": ["lr", "lr_decay", "weight_decay"],
                "inputs": ["params"], "outputs": [],
                "defaults": {"lr": "0.01", "lr_decay": "0.0", "weight_decay": "0.0"},
                "tooltip": "Adapts learning rate per parameter - good for sparse data.",
                "when_to_use": "Use for NLP or sparse feature problems. Can learn too slowly over time.",
            },
            {
                "label": "LBFGS", "color": (100, 220, 180),
                "params": ["lr", "max_iter", "history_size"],
                "inputs": ["params"], "outputs": [],
                "defaults": {"lr": "1.0", "max_iter": "20", "history_size": "100"},
                "tooltip": "Quasi-Newton optimizer - very accurate but memory-intensive.",
                "when_to_use": "Use for small datasets where high accuracy matters more than speed.",
            },
        ],
    },
    "Data Prep": {
        "Datasets": [
            {
                "label": "MNIST", "color": (220, 180, 255),
                "params": ["root", "train", "download"],
                "inputs": [], "outputs": ["img"],
                "defaults": {"root": "./data", "train": "True", "download": "True"},
                "tooltip": "Handwritten digit images - 60,000 training samples, 10 classes.",
                "when_to_use": "Classic beginner dataset. Great for learning the basics of image classification.",
            },
            {
                "label": "CIFAR10", "color": (220, 180, 255),
                "params": ["root", "train", "download"],
                "inputs": [], "outputs": ["img"],
                "defaults": {"root": "./data", "train": "True", "download": "True"},
                "tooltip": "Colour photos across 10 categories - cars, dogs, planes, etc.",
                "when_to_use": "Good next step after MNIST. Tests CNNs on real 3-channel images.",
            },
            {
                "label": "CIFAR100", "color": (220, 180, 255),
                "params": ["root", "train", "download"],
                "inputs": [], "outputs": ["img"],
                "defaults": {"root": "./data", "train": "True", "download": "True"},
                "tooltip": "Like CIFAR10 but with 100 fine-grained classes.",
                "when_to_use": "Use when you want a harder challenge than CIFAR10.",
            },
            {
                "label": "FashionMNIST", "color": (220, 180, 255),
                "params": ["root", "train", "download"],
                "inputs": [], "outputs": ["img"],
                "defaults": {"root": "./data", "train": "True", "download": "True"},
                "tooltip": "Grayscale clothing images - a harder drop-in replacement for MNIST.",
                "when_to_use": "Use when MNIST feels too easy. Same shape (1x28x28), 10 classes.",
            },
            {
                "label": "ImageFolder", "color": (220, 180, 255),
                "params": ["root"],
                "inputs": [], "outputs": ["img"],
                "defaults": {"root": "./data"},
                "tooltip": "Loads images from a folder organised as root/class_name/image.jpg.",
                "when_to_use": "Use for your own custom image dataset. Each subfolder becomes a class.",
            },
        ],
        "Augmentation": [
            {
                "label": "Resize", "color": (255, 200, 120),
                "params": ["size"],
                "inputs": ["img"], "outputs": ["img"], "defaults": {},
                "tooltip": "Rescales images to a fixed size.",
                "when_to_use": "Use when your model expects a specific input resolution (e.g. 224 for ResNet-style).",
            },
            {
                "label": "CenterCrop", "color": (255, 200, 120),
                "params": ["size"],
                "inputs": ["img"], "outputs": ["img"], "defaults": {},
                "tooltip": "Crops a square from the centre of the image.",
                "when_to_use": "Use at validation time after Resize to get a fixed-size crop.",
            },
            {
                "label": "RandomCrop", "color": (255, 200, 120),
                "params": ["size", "padding"],
                "inputs": ["img"], "outputs": ["img"], "defaults": {"padding": "4"},
                "tooltip": "Crops a random region - augments training data with spatial variety.",
                "when_to_use": "Use during training to improve generalisation. Common for CIFAR models.",
            },
            {
                "label": "RandomHFlip", "color": (255, 200, 120),
                "params": ["p"],
                "inputs": ["img"], "outputs": ["img"], "defaults": {"p": "0.5"},
                "tooltip": "Randomly flips images horizontally with probability p.",
                "when_to_use": "Almost always helpful for natural images. Avoid for digits or text.",
            },
            {
                "label": "RandomVFlip", "color": (255, 200, 120),
                "params": ["p"],
                "inputs": ["img"], "outputs": ["img"], "defaults": {"p": "0.5"},
                "tooltip": "Randomly flips images vertically with probability p.",
                "when_to_use": "Use for medical or satellite images. Avoid when orientation matters.",
            },
            {
                "label": "ColorJitter", "color": (255, 200, 120),
                "params": ["brightness", "contrast", "saturation", "hue"],
                "inputs": ["img"], "outputs": ["img"], "defaults": {},
                "tooltip": "Randomly changes brightness, contrast, saturation and hue.",
                "when_to_use": "Use to make models robust to lighting changes. Keep values small (e.g. 0.2).",
            },
            {
                "label": "RandomRotation", "color": (255, 200, 120),
                "params": ["degrees"],
                "inputs": ["img"], "outputs": ["img"], "defaults": {},
                "tooltip": "Rotates images by a random angle within plus or minus degrees.",
                "when_to_use": "Use when object orientation varies in your dataset. degrees=15 is a safe start.",
            },
            {
                "label": "GaussianBlur", "color": (255, 200, 120),
                "params": ["kernel_size", "sigma"],
                "inputs": ["img"], "outputs": ["img"], "defaults": {"kernel_size": "3", "sigma": "0.1, 2.0"},
                "tooltip": "Applies random Gaussian blur to simulate out-of-focus images.",
                "when_to_use": "Use to improve robustness to blur. sigma accepts a range like '0.1, 2.0'.",
            },
            {
                "label": "RandomErasing", "color": (255, 200, 120),
                "params": ["p", "scale", "ratio"],
                "inputs": ["img"], "outputs": ["img"], "defaults": {"p": "0.5", "scale": "0.02, 0.33", "ratio": "0.3, 3.3"},
                "tooltip": "Randomly erases a rectangular patch - forces the model to use context.",
                "when_to_use": "Place after ToTensor. Helps with occlusion robustness.",
            },
            {
                "label": "Normalize", "color": (255, 200, 120),
                "params": ["mean", "std"],
                "inputs": ["img"], "outputs": ["img"], "defaults": {"mean": "0.5", "std": "0.5"},
                "tooltip": "Subtracts mean and divides by std - centres pixel values around zero.",
                "when_to_use": "Always use after ToTensor. For CIFAR10 use mean=[0.491,0.482,0.447] std=[0.247,0.243,0.262].",
            },
            {
                "label": "ToTensor", "color": (255, 200, 120),
                "params": [],
                "inputs": ["img"], "outputs": ["img"], "defaults": {},
                "tooltip": "Converts a PIL image to a PyTorch tensor and scales pixels to [0, 1].",
                "when_to_use": "Always required - place before Normalize and after all other transforms.",
            },
            {
                "label": "Grayscale", "color": (255, 200, 120),
                "params": ["num_output_channels"],
                "inputs": ["img"], "outputs": ["img"], "defaults": {"num_output_channels": "1"},
                "tooltip": "Converts an RGB image to grayscale.",
                "when_to_use": "Use when colour is not informative for your task. Set num_output_channels=1.",
            },
        ],
        "DataLoader": [
            {
                "label": "DataLoader (train)", "color": (200, 160, 255),
                "params": ["batch_size", "shuffle", "num_workers", "pin_memory"],
                "inputs": ["img"], "outputs": [],
                "defaults": {"batch_size": "32", "shuffle": "True", "num_workers": "0", "pin_memory": "False"},
                "tooltip": "Wraps the training dataset and feeds shuffled batches to the model.",
                "when_to_use": "Required. batch_size=32 or 64 is a good start. Keep shuffle=True for training.",
            },
            {
                "label": "DataLoader (val)", "color": (180, 140, 235),
                "params": ["batch_size", "num_workers", "pin_memory"],
                "inputs": ["img"], "outputs": [],
                "defaults": {"batch_size": "32", "num_workers": "0", "pin_memory": "False"},
                "tooltip": "Wraps the validation dataset - used to track generalisation each epoch.",
                "when_to_use": "Add a second dataset chain with train=False ending here for proper validation metrics.",
            },
        ],
    },
    "Inference": {
        "Datasets": [
            {
                "label": "Inf MNIST", "color": (140, 210, 255),
                "params": ["root", "train", "download"],
                "inputs": [], "outputs": ["images"],
                "defaults": {"root": "./data", "train": "False", "download": "True"},
                "tooltip": "Inference-only MNIST dataset source.",
                "when_to_use": "Use as the image source in the Inference tab before transforms and checkpoint nodes.",
            },
            {
                "label": "Inf FashionMNIST", "color": (140, 210, 255),
                "params": ["root", "train", "download"],
                "inputs": [], "outputs": ["images"],
                "defaults": {"root": "./data", "train": "False", "download": "True"},
                "tooltip": "Inference-only FashionMNIST dataset source.",
                "when_to_use": "Use as the image source in the Inference tab before transforms and checkpoint nodes.",
            },
            {
                "label": "Inf CIFAR10", "color": (140, 210, 255),
                "params": ["root", "train", "download"],
                "inputs": [], "outputs": ["images"],
                "defaults": {"root": "./data", "train": "False", "download": "True"},
                "tooltip": "Inference-only CIFAR10 dataset source.",
                "when_to_use": "Use as the image source in the Inference tab before transforms and checkpoint nodes.",
            },
            {
                "label": "Inf CIFAR100", "color": (140, 210, 255),
                "params": ["root", "train", "download"],
                "inputs": [], "outputs": ["images"],
                "defaults": {"root": "./data", "train": "False", "download": "True"},
                "tooltip": "Inference-only CIFAR100 dataset source.",
                "when_to_use": "Use as the image source in the Inference tab before transforms and checkpoint nodes.",
            },
            {
                "label": "Inf ImageFolder", "color": (140, 210, 255),
                "params": ["root"],
                "inputs": [], "outputs": ["images"],
                "defaults": {"root": "./data"},
                "tooltip": "Inference-only ImageFolder dataset source.",
                "when_to_use": "Use for custom images in the Inference tab before transforms and checkpoint nodes.",
            },
        ],
        "Preprocess": [
            {
                "label": "Inf Resize", "color": (120, 190, 255),
                "params": ["size"],
                "inputs": ["images"], "outputs": ["images"],
                "defaults": {"size": "224"},
                "tooltip": "Rescales images to a fixed size for inference.",
                "when_to_use": "Use when the checkpoint expects a fixed resolution.",
            },
            {
                "label": "Inf CenterCrop", "color": (120, 190, 255),
                "params": ["size"],
                "inputs": ["images"], "outputs": ["images"],
                "defaults": {"size": "224"},
                "tooltip": "Applies a center crop in the inference graph.",
                "when_to_use": "Use after resize for a deterministic inference crop.",
            },
            {
                "label": "Inf Normalize", "color": (120, 190, 255),
                "params": ["mean", "std"],
                "inputs": ["images"], "outputs": ["images"],
                "defaults": {"mean": "0.5", "std": "0.5"},
                "tooltip": "Normalizes images before checkpoint inference.",
                "when_to_use": "Use when the trained model expects normalized inputs.",
            },
            {
                "label": "Inf ToTensor", "color": (120, 190, 255),
                "params": [],
                "inputs": ["images"], "outputs": ["images"],
                "defaults": {},
                "tooltip": "Converts images to tensors for checkpoint inference.",
                "when_to_use": "Usually required before Normalize and checkpoint nodes.",
            },
            {
                "label": "Inf Grayscale", "color": (120, 190, 255),
                "params": ["num_output_channels"],
                "inputs": ["images"], "outputs": ["images"],
                "defaults": {"num_output_channels": "1"},
                "tooltip": "Converts inference images to grayscale.",
                "when_to_use": "Use if your checkpoint was trained on grayscale inputs.",
            },
        ],
        "load checkpoint": [
            {
                "label": "pth", "color": (120, 210, 255),
                "params": ["path"],
                "inputs": ["images"], "outputs": ["predictions"],
                "defaults": {"path": ""},
                "tooltip": "Checkpoint loader node for .pth model weights.",
                "when_to_use": "Connect dataset/transform output into this node, then send predictions to an inference output node.",
            },
            {
                "label": "pt", "color": (120, 210, 255),
                "params": ["path"],
                "inputs": ["images"], "outputs": ["predictions"],
                "defaults": {"path": ""},
                "tooltip": "Checkpoint loader node for .pt model weights.",
                "when_to_use": "Connect dataset/transform output into this node, then send predictions to an inference output node.",
            },
            {
                "label": "ckpt", "color": (120, 210, 255),
                "params": ["path"],
                "inputs": ["images"], "outputs": ["predictions"],
                "defaults": {"path": ""},
                "tooltip": "Checkpoint loader node for generic .ckpt checkpoint files.",
                "when_to_use": "Connect dataset/transform output into this node, then send predictions to an inference output node.",
            },
        ],
        "Outputs": [
            {
                "label": "InferenceOutput", "color": (80, 220, 200),
                "params": ["top_k"],
                "inputs": ["predictions"], "outputs": [],
                "defaults": {"top_k": "5"},
                "tooltip": "Terminal node for inference results.",
                "when_to_use": "Place after a checkpoint node so RUN -> Inference knows where the inference flow ends.",
            },
        ],
    },
}


def get_block_def(label: str) -> Optional[dict]:
    """Return the block definition dict for a given label, or None."""
    for section in SECTIONS.values():
        for block_list in section.values():
            for block in block_list:
                if block["label"] == label:
                    return block
    return None


def all_block_labels() -> list[str]:
    """Return a flat list of every block label across all sections."""
    labels = []
    for section in SECTIONS.values():
        for block_list in section.values():
            for block in block_list:
                labels.append(block["label"])
    return labels
