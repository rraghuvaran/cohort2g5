import os
import re
import numpy as np
import torch
import pandas as pd
from sklearn import preprocessing
from skimage.draw import polygon
from PIL import Image
from torch.nn.utils.rnn import pad_sequence



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
        print(type(img))
        height = img.height
        width = img.width
        print(height,width)

        boxes = []
        labels = []
        areas = []
        masks = []
        image_base_name = image_name.rsplit('.', maxsplit=1)[0]
        image_base_name = re.match(r'([0-9_]*)(\(0\))?',image_base_name).group(1)
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

            px_list_str = row.boundary_points
            px_array = np.array(px_list_str.strip('][').split(', ')).reshape((-1,2)).T.astype(int)
            print(px_array)
            mask_img = np.zeros((width, height), dtype=np.uint8)
            r = px_array[0]
            c = px_array[1]
            rr, cc = polygon(r, c)
            print(type(mask_img))
            mask_img[rr, cc] = 1
            masks.append(mask_img)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.int64)
        masks = torch.as_tensor(np.asarray(masks), dtype=torch.uint8)

        image_id = torch.tensor([idx])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["areas"] = areas
        target["iscrowd"] = iscrowd
        target["masks"] = masks

        # Preprocessing
        #target = {
        #    key: value.numpy() for key, value in target.items()
        #}  # all tensors should be converted to np.ndarrays

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # Typecasting
        #img = torch.from_numpy(img).type(torch.float32)
        #target = {
        #    key: torch.from_numpy(value).type(torch.int64)
        #    for key, value in target.items()
        #}
        print(f"Img: {img} {type(img)} Target: {target}")
        #target["boxes"] = pad_sequence(target["boxes"], batch_first=True, padding_value=-1)
        #target["labels"] = pad_sequence(target["labels"], batch_first=True, padding_value=-1)

        return img, target

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    TRAIN_DIR = './train'
    # sanity check of the Dataset pipeline with sample visualization
    dataset = Unimib2016FoodDataset('./data', 'train', None)
    print(f"Number of training images: {len(dataset)}")
    for i in range(1):
        s, t = dataset[i]
        print(s, t)
