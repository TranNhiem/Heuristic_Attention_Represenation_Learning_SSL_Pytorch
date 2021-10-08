import torchvision
import matplotlib.pyplot as plt

root = r'COCO2017/train2017'
annFile_train = r'./COCO2017/annotations/instances_train2017.json'
annFile_val = r'./COCO2017/annotations/instances_val2017.json'

coco_dataset_train = torchvision.datasets.CocoDetection(root, annFile_train, transform=None, target_transform=None, transforms=None)

coco_dataset_val = torchvision.datasets.CocoDetection(root, annFile_val, transform=None, target_transform=None, transforms=None)