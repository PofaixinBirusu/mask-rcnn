import cv2
import numpy as np
from PIL import Image
import random
import torch

CAN_USE_GPU = torch.cuda.is_available()


def cv2pil(cvimg):
    return Image.fromarray(cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB))


def pil2cv(pilimg):
    return cv2.cvtColor(np.asarray(pilimg), cv2.COLOR_RGB2BGR)


def draw_rec(img, box, cls_name):
    xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
    img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(255, 0, 0), thickness=2)
    img = cv2.rectangle(img, (xmin, ymin), (xmax, ymin+15), color=(255, 0, 0), thickness=-1)
    img = cv2.putText(img, cls_name, (xmin, ymin+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                      color=(255, 255, 255), thickness=2)
    return img


def drwa_mask(img, roi_mask, boxes):
    # colors = [(200, 200, 200), (239, 211, 127), (225, 169, 0), (211, 127, 254),
    #           (197, 84, 127), (183, 42, 0), (169, 0, 254), (155, 42, 127),
    #           (141, 84, 0), (127, 254, 254), (112, 211, 127), (98, 169, 0),
    #           (84, 127, 254), (70, 84, 127), (56, 42, 0), (42, 0, 254),
    #           (28, 42, 127), (14, 84, 0), (0, 254, 254), (14, 211, 127)]
    colors = [
        (237, 191, 32), (255, 102, 203), (148, 101, 255), (103, 238, 49), (13, 255, 195),
        (255, 195, 100), (108, 177, 46), (244, 92, 181), (238, 133, 81), (240, 224, 88),
        (181, 240, 88), (205, 138, 162), (114, 8, 203), (0, 180, 255), (255, 128, 206),
        (241, 115, 102), (102, 241, 158), (200, 241, 102), (208, 29, 224), (46, 178, 247)
    ]
    # roi是box大小的01矩阵，1表示要上色
    img_mask = np.copy(img)
    img_h, img_w = img.shape[0], img.shape[1]
    for i, roi in enumerate(roi_mask):
        color = np.array(colors[i+4], dtype=np.uint8)
        xmin, ymin, xmax, ymax = boxes[i]
        left, top, right, bottom = 0, 0, 0, 0
        if xmin < 0:
            xmin, left = 0, -xmin
        if ymin < 0:
            ymin, top = 0, -ymin
        if xmax >= img_w:
            xmax, right = img_w-1, xmax-img_w+1
        if ymax >= img_h:
            ymax, bottom = img_h-1, ymax-img_h+1
        # xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), min(xmax, img_w-1), min(ymax, img_h-1)
        color_mask = img[ymin:ymax+1, xmin:xmax+1, :]
        if bottom == 0:
            bottom = -img_h
        if right == 0:
            right = -img_w
        roi = roi[top:-bottom, left:-right]
        # print(roi)
        h, w = color_mask.shape[0], color_mask.shape[1]
        ind = np.nonzero(roi.reshape(-1) == 1)[0]
        color_mask = color_mask.reshape((-1, 3))
        color_mask[ind, :] = color
        color_mask = color_mask.reshape((h, w, 3))
        # cv2.imshow("color_mask", color_mask)
        # break
        img_mask[ymin:ymax+1, xmin:xmax+1, :] = color_mask
    cv2.imshow("mask", img_mask)
    img = cv2.addWeighted(img, 0.3, img_mask, 0.7, 0)
    return img


def one_hot(label, n_class):
    # label的size: batch_size x h x w
    batch_size, h, w = label.shape[0], label.shape[1], label.shape[2]
    label = torch.LongTensor(label)
    one_hot_label = torch.zeros(batch_size, n_class, h, w).permute([0, 2, 3, 1])\
        .scatter_(dim=3, index=label.view(batch_size, h, w, 1), value=1).permute([0, 3, 1, 2])
    # if CAN_USE_GPU:
    #     one_hot_label = one_hot_label.cuda()
    return one_hot_label.numpy()


def imread(path, tw, th):
    img = cv2.imread(path)
    new_w, new_h = [int(wh*min(tw/img.shape[1], th/img.shape[0])) for wh in (img.shape[1], img.shape[0])]
    canvas = np.full((th, tw, 3), 128, dtype=np.uint8)
    canvas[(th-new_h)//2:(th-new_h)//2+new_h, (tw-new_w)//2:(tw-new_w)+new_w] = cv2.resize(img, (new_w, new_h))
    return canvas, new_w, new_h


if __name__ == '__main__':
    pass
