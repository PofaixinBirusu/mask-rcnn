import cv2
import os
import json
import torch
import numpy as np
from mask_rcnn.util import drwa_mask
from mask_rcnn.util import draw_rec
from torch.utils import data
from torchvision import transforms
from mask_rcnn.util import cv2pil


class PikachuDataset(data.Dataset):
    def __init__(self, path):
        super(PikachuDataset, self).__init__()
        for _, _, filelist in os.walk(path):
            self.img_list = [path+"/"+filename for filename in filelist if filename.endswith(".jpg")]
            self.json_list = [path+"/json/"+filename.replace("jpg", "json") for filename in filelist if filename.endswith(".jpg")]
            break
        print(self.img_list)
        print(self.json_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        json_path = self.json_list[index]
        img = cv2.imread(img_path)
        with open(json_path, "r", encoding="utf-8") as f:
            label = json.load(f)
        objs = label["outputs"]["object"]
        class_name = ["pikachu", "ball"]
        mask = np.zeros(shape=(img.shape[0], img.shape[1], 1), dtype=np.uint8)
        boxes, cls = [], []
        for obj in objs:
            polygon, pts = obj["polygon"], []
            xmin, ymin, xmax, ymax = 1e6, 1e6, -1, -1
            for i in range(len(polygon.keys()) // 2):
                x, y = polygon["x%d" % (i + 1)], polygon["y%d" % (i + 1)]
                pts.append([x, y])
                xmin, ymin, xmax, ymax = min(xmin, x), min(ymin, y), max(xmax, x), max(ymax, y)
            if xmax-xmin+1 < 5 or ymax-ymin+1 < 5:
                continue
            boxes.append([xmin//2, ymin//2, xmax//2, ymax//2])
            cls.append(0) if obj["name"] == "pikachu" else cls.append(1)
            color = (2,) if obj["name"] == "ball" else (1,)
            mask = cv2.fillPoly(mask, pts=np.array([pts]), color=color)
        img = cv2.resize(img, dsize=(720, 640))
        mask = cv2.resize(mask, dsize=(720, 640))
        # print(mask.shape)
        # print(img.shape)
        # cv2.imshow("pikachu", img)
        # cv2.imshow("mask", mask)
        # cv2.waitKey(0)
        # for box in boxes:
        #     img = draw_rec(img, box, "xxx")
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        img2tensor = transforms.ToTensor()
        return img2tensor(cv2pil(img)), torch.Tensor(boxes), torch.Tensor(cls).long(), mask


if __name__ == '__main__':
    dataset = PikachuDataset("C:/Users/XR/Desktop/pikachu")
    for i in range(len(dataset)):
        x, _, _, _ = dataset[i]
        print(x.shape)