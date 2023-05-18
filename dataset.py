import os
import numpy as np
import torch
import pandas as pd
from sklearn import preprocessing
from PIL import Image


class Unimib2016FoodDataset(torch.utils.data.Dataset):
    def __init__(self, root, subdir, transforms):
        self.root = root
        self.subdir = subdir
        self.imgdir_path = os.path.join(root,  subdir, 'original')
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root,  subdir, 'original'))))
        print(self.imgs)
        self.annotations_df = pd.read_csv(os.path.join(self.root, 'annotations.csv'))
        item_name_le = preprocessing.LabelEncoder()
        self.annotations_df['item_label_id'] = item_name_le.fit_transform(self.annotations_df['item_name'])
        item_class_le = preprocessing.LabelEncoder()
        self.annotations_df['item_class_id'] = item_class_le.fit_transform(self.annotations_df['item_class'])
        self.image_df = self.annotations_df.drop_duplicates(subset=['image_name']).sort_values(by='image_name')

    def __getitem__(self, idx):
        # load images and masks
        image_name = self.imgs[idx]
        img_path = os.path.join(self.imgdir_path, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        areas = []
        image_base_name = image_name.rsplit('.', maxsplit=1)[0]
        image_item_df = self.annotations_df.loc[self.annotations_df['image_name'] == image_base_name]
        num_objs = image_item_df.shape[1] 
        print(f"Image {image_name} has {num_objs}")
        for row_index, row in image_item_df.iterrows():
            #print(f"{image_name}-{row_index}-{row.item_name}")
            # get bounding box coordinates for each object 
            bx_list_str = row.bounding_box
            res = np.array(bx_list_str.strip('][').split(', ')).reshape((4, 2)).T.astype(float)
            xmin = np.min(res[0])
            xmax = np.max(res[0])
            ymin = np.min(res[1])
            ymax = np.max(res[1])
            area = (xmax - xmin) * (ymax - ymin)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(row.item_label_id)
            areas.append(area)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        print(f"Img: {img} Target: {target}")

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    TRAIN_DIR = './train'
    # sanity check of the Dataset pipeline with sample visualization
    dataset = Unimib2016FoodDataset('./data', 'train', None)
    print(f"Number of training images: {len(dataset)}")
    for i in range(10):
        s, t = dataset[i]
        print(s, t)
