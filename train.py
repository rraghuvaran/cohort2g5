import numpy as np
import torch
import transform as transforms
from torch.utils.data import DataLoader
import dataset
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence

# Applying Transforms to the Data
image_transforms = { 
    'train': transforms.Compose([
        transforms.Resize(size=(256,256)),
        #transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=(256,256)),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=(256,256)),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# Load the Data
# Set train and valid directory paths
root_directory = 'data'
train_directory = 'train'
valid_directory = 'val'
test_directory = 'test'
# Batch size
bs = 10
# Number of classes
num_classes = 73 
# Load Data from folders
data = {
    'train': dataset.Unimib2016FoodDataset(root=root_directory, subdir=train_directory, transforms=image_transforms['train']),
    'validation': dataset.Unimib2016FoodDataset(root=root_directory, subdir=valid_directory, transforms=image_transforms['valid']),
    'test': dataset.Unimib2016FoodDataset(root=root_directory, subdir=test_directory, transforms=image_transforms['test'])
}
# Size of Data, to be used for calculating Average Loss and Accuracy
train_data_size = len(data['train'])
validation_data_size = len(data['validation'])
test_data_size = len(data['test'])

def collate_batch(batch):
  img_list, target_list, = [], []
  for (_img,_target) in batch:
    img_list.append(_img)
    target_list.append(_target)
  print(type(target_list))
  target_list = pad_sequence(target_list, batch_first=True, padding_value=0)

  return img_list.to(device),target_list.to(device),

# Create iterators for the Data loaded using DataLoader module
train_data = DataLoader(data['train'], batch_size=bs, shuffle=True, collate_fn=collate_batch)
valid_data = DataLoader(data['validation'], batch_size=bs, shuffle=True, collate_fn=collate_batch)
test_data = DataLoader(data['test'], batch_size=bs, shuffle=True, collate_fn=collate_batch)
# Print the train, validation and test set data sizes
print(train_data_size, validation_data_size, test_data_size)

examples = next(iter(train_data))
for label, img  in enumerate(examples):
   plt.imshow(img.permute(1,2,0))
   plt.show()
   print(f"Label: {label}")
