
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.axes_grid1 import ImageGrid
import utils

def plot_layer(model, layer, channels, image_path):

  image = Image.open(image_path)
  print("Image shape:", image.size)

  #model = torchvision.models.resnet18(pretrained=True)

  model_layers = model.feature_extractor.children()
  model_layers = [l for l in model_layers]
  #print(model_layers)
  #layers.pop()
  #layers.pop()

  #print(layers)
  #print(len(layers))

  #first_conv_layer = model.conv1
  #print("First conv layer weight shape:", first_conv_layer.weight.shape)
  #print("First conv layer:", first_conv_layer)

  # Resize, and normalize the image with the mean and standard deviation
  image_transform = torchvision.transforms.Compose([
      torchvision.transforms.Resize((224, 224)),
      torchvision.transforms.ToTensor(),
      #torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])
  image = image_transform(image)[None]
  print("Image shape:", image.shape)

  activation = get_activations(image, layer, model_layers)

  #activation = first_conv_layer(image)
  print("Activation shape:", activation.shape)

  #indices = [14, 26, 32, 49, 52]
  #indices = range(10)

  fig = plt.figure(1,(10,10))
  grid = ImageGrid(fig, 111,
                   nrows_ncols=(2,len(channels)),
                   axes_pad=0.1,)

  #plot only layer output
  for n, i in enumerate(channels, start=0):
    img = torch_image_to_numpy(activation.data[0,i,:,:])
    img = cv2.resize(img, dsize=(112,112), interpolation=cv2.INTER_NEAREST)
    img = img.squeeze()
    grid[n].imshow(img,cmap='gray',interpolation='none')

  # plot kernals
  for n, i in enumerate(channels, start=len(channels)):
    print("Weight shape: ", model_layers[layer].weight.data.shape)
    img = torch_image_to_numpy(model_layers[layer].weight.data[i,0,:,:])
    img = cv2.resize(img, dsize=(112,112), interpolation=cv2.INTER_NEAREST)
    img = img.squeeze()
    grid[n].imshow(img,interpolation='none')


  fig.savefig("images/layer_ouputs.png")


def get_activations(in_, layer, model_layers):
  in_ = utils.to_cuda(in_)

  for i in range(layer+1):
    in_ = model_layers[i](in_)
  out = in_

  #try:
  #  out = model_layers[layer](in_)
  #  return get_activations(out, layer+1, model_layers)
  #except:
  return out


def torch_image_to_numpy(image: torch.Tensor):
    """
    Function to transform a pytorch tensor to numpy image
    Args:
        image: shape=[3, height, width]
    Returns:
        image: shape=[height, width, 3] in the range [0, 1]
    """
    # Normalize to [0 - 1.0]
    image = image.detach().cpu() # Transform image to CPU memory (if on GPU VRAM)
    image = image - image.min()
    image = image / image.max()
    image = image.numpy()
    if len(image.shape) == 2: # Grayscale image, can just return
        return image
    assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(image.shape)
    image = np.moveaxis(image, 0, 2)
    return image



