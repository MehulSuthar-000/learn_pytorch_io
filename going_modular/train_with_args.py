"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import argparse
import data_setup, engine, model_builder, utils

from torchvision import transforms

arg_parser = argparse.ArgumentParser(
    description="Train a PyTorch image classification model."
)

# Add arguments to the parser
arg_parser.add_argument(
    "--train_dir",
    type=str,
    default="data/pizza_steak_sushi/train",
    help="Path to training directory."
)

arg_parser.add_argument(
    "--test_dir",
    type=str,
    default="data/pizza_steak_sushi/test",
    help="Path to testing directory."
)

arg_parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Number of samples per batch."
)

arg_parser.add_argument(
    "--hidden_units",
    type=int,
    default=10,
    help="Number of hidden units between layers."
)

arg_parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.001,
    help="Learning rate for the optimizer."
)

arg_parser.add_argument(
    "--num_epochs",
    type=int,
    default=5,
    help="Number of epochs to train for."
)

arg_parser.add_argument(
    "--model_name",
    type=str,
    default="05_going_modular_script_mode_tinyvgg_model.pth",
    help="Name of the model to be saved."
)

args = arg_parser.parse_args()


# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
train_transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.TrivialAugmentWide(num_magnitude_bins=31),
  transforms.ToTensor()
])
test_transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloader(
    train_dir=args.train_dir,
    test_dir=args.test_dir,
    train_transform=train_transform,
    test_transform=test_transform,
    batch_size=args.batch_size
)

# Create model with help from model_builder.py
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=args.hidden_units,
    output_shape=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.learning_rate)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=args.num_epochs,
             device=device)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name=args.model_name)
