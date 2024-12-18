import torch
import torchvision
import argparse
import model_builder

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Creating a parser
parser = argparse.ArgumentParser(
    description="Predict on an image using a trained PyTorch model."
)

# Get an image path
parser.add_argument("--image",
                    help="target image filepath to predict on")

# Get a model path
parser.add_argument("--model_path",
                    default="models/05_going_modular_script_mode_tinyvgg_model.pth",
                    type=str,
                    help="target model to use for prediction filepath")

args = parser.parse_args()

# Setup class names
class_names = ["pizza", "steak", "sushi"]

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Get the image path
IMG_PATH = args.image
print(f"[INFO] Predicting on {IMG_PATH}")

# Function to load in the model
def load_model(filepath=args.model_path):
  # Need to use same hyperparameters as saved model 
  model = model_builder.TinyVGG(input_shape=3,
                                hidden_units=10,
                                output_shape=3).to(device)

  print(f"{bcolors.OKGREEN} [INFO] Loading in model from: {filepath} {bcolors.ENDC}")
  # Load in the saved model state dictionary from file                               
  model.load_state_dict(torch.load(filepath, weights_only=True))

  return model

# Function to load in model + predict on select image
def predict_on_image(image_path=IMG_PATH, filepath=args.model_path):
  # Load the model
  model = load_model(filepath)

  # Load in the image and turn it into torch.float32 (same type as model)
  image = torchvision.io.read_image(str(IMG_PATH)).type(torch.float32)

  # Preprocess the image to get it between 0 and 1
  image = image / 255.

  # Resize the image to be the same size as the model
  transform = torchvision.transforms.Resize(size=(224, 224))
  image = transform(image) 

  # Predict on image
  model.eval()
  with torch.inference_mode():
    # Put image to target device
    image = image.to(device)

    # Get pred logits
    pred_logits = model(image.unsqueeze(dim=0)) # make sure image has batch dimension (shape: [batch_size, height, width, color_channels])

    # Get pred probs
    pred_prob = torch.softmax(pred_logits, dim=1)

    # Get pred labels
    pred_label = torch.argmax(pred_prob, dim=1)
    pred_label_class = class_names[pred_label]

  print(f"{bcolors.OKGREEN} [INFO] Pred class: {pred_label_class}, Pred prob: {pred_prob.max():.3f} {bcolors.ENDC}")

if __name__ == "__main__":
  predict_on_image()
