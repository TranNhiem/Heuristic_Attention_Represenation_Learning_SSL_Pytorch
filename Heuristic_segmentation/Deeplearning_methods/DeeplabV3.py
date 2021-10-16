'''
This is implemenation DeepLabV3 
Paper: 
Link Reference: https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/

train with coco2017, torch need 1.6up
'''
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models

import torch
import torchvision



def Generated_mask(image=None):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    # model = models.segmentation.deeplabv3_resnet50(True,True)

    model.eval()


    filename ="test_image/n00005787_654.JPEG"

    from PIL import Image
    from torchvision import transforms
    input_image = Image.open(filename)
    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    print("using cuda:",torch.cuda.is_available())
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]

    print(output.shape)
    output_predictions = output.argmax(0)

    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette(colors)

    import matplotlib.pyplot as plt
    plt.imsave(filename+"_test.jpg",r)
    print(filename+"_test.jpg")


Generated_mask()