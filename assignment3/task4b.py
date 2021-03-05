
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

image = Image.open("images/zebra.jpg")
print("Image shape:", image.size)

model = torchvision.models.resnet18(pretrained=True)

layers = model.children()
layers = [l for l in layers]
layers.pop()
layers.pop()

#print(layers)
#print(len(layers))

#first_conv_layer = model.conv1
#print("First conv layer weight shape:", first_conv_layer.weight.shape)
#print("First conv layer:", first_conv_layer)

# Resize, and normalize the image with the mean and standard deviation
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = image_transform(image)[None]
print("Image shape:", image.shape)

def get_activations(in_, layer):
  try:
    out = layers[layer](in_)
    return get_activations(out, layer+1)
  except:
    return out

activation = get_activations(image, 0)


#activation = first_conv_layer(image)
print("Activation shape:", activation.shape)


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


indices = [14, 26, 32, 49, 52]
indices = range(10)

from mpl_toolkits.axes_grid1 import ImageGrid
fig = plt.figure(1,(10,10))
grid = ImageGrid(fig, 111,
                 nrows_ncols=(1,len(indices)),
                 axes_pad=0.1,)
                 
for n, i in enumerate(indices, start=0):
  img = torch_image_to_numpy(activation.data[0,i,:,:])
  img = img.squeeze()
  grid[n].imshow(img,cmap='gray',interpolation='none')

#for n, i in enumerate(indices, start=0):
#  img = torch_image_to_numpy(first_conv_layer.weight.data[i,:,:,:])
#  img = cv2.resize(img, dsize=(112,112), interpolation=cv2.INTER_NEAREST)
#  img = img.squeeze()
#  grid[n].imshow(img,interpolation='none')


fig.savefig("4c.png")


