from PIL import Image, ImageSequence, ImageOps
import numpy as np
import node_helpers
import torch

def comfyimage2Image(comfyimage):
  comfyimage = comfyimage.numpy()[0] * 255
  image_np = comfyimage.astype(np.uint8)
  image = Image.fromarray(image_np)
  return image

def image2Comfyimage(img):
  output_images = []
  output_masks = []
  w, h = None, None

  excluded_formats = ['MPO']
  
  for i in ImageSequence.Iterator(img):
    i = node_helpers.pillow(ImageOps.exif_transpose, i)

    if i.mode == 'I':
      i = i.point(lambda i: i * (1 / 255))
    image = i.convert("RGB")

    if len(output_images) == 0:
      w = image.size[0]
      h = image.size[1]
    
    if image.size[0] != w or image.size[1] != h:
        continue
    
    image = np.array(image).astype(np.float16) / 255.0
    image = torch.from_numpy(image)[None,]
    if 'A' in i.getbands():
      mask = np.array(i.getchannel('A')).astype(np.float16) / 255.0
      mask = 1. - torch.from_numpy(mask)
    else:
      mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
    output_images.append(image)
    output_masks.append(mask.unsqueeze(0))

  if len(output_images) > 1 and img.format not in excluded_formats:
    output_image = torch.cat(output_images, dim=0)
    output_mask = torch.cat(output_masks, dim=0)
  else:
    output_image = output_images[0]
    output_mask = output_masks[0]

  return (output_image, output_mask)

def calculate_image_resize(width, height, min_dim, max_dim):
  if min_dim <= height <= max_dim and min_dim <= width <= max_dim:
    return width, height
    
  ratio = width / height
  
  if height < width:
    new_height = min_dim
    new_width = int(new_height * ratio)

    if new_width > max_dim:
      new_width = max_dim
      new_height = int(new_width / ratio)
  else:
    new_width = min_dim
    new_height = int(new_width / ratio)

    if new_height > max_dim:
      new_height = max_dim
      new_width = int(new_height * ratio)
  
  return new_width, new_height

def resize_img(input_image, width, height, mode=Image.BILINEAR):
  input_image = input_image.resize((width, height), mode)
  return input_image

class GlovyResizeNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "image":("IMAGE",),
                "min_dim":("INT",{
                    "default": 768,
                    "min": 1
                }),
                "max_dim":("INT",{
                    "default": 1280,
                    "min": 1
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Glovy"

    def generate(self, image, min_dim, max_dim):      
      im = comfyimage2Image(image)
      width, height = im.size
      
      new_width, new_height = calculate_image_resize(width, height, min_dim, max_dim)
      im = resize_img(im, new_width, new_height)
      print(image2Comfyimage(im))
      return (image2Comfyimage(im)[0], )

NODE_CLASS_MAPPINGS = {
    "GlovyResizeNode": GlovyResizeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GlovyResizeNode":"GlovyResizeNode",
}
