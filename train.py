import torch
from data import PikachuDataset
from mask_rcnn.rpn import RPN
from mask_rcnn.rcnn import RCNN
from mask_rcnn.model import GoogLeNet
from mask_rcnn.util import drwa_mask
from mask_rcnn.util import draw_rec
from torch.utils import data
from torchvision import transforms
import numpy
import cv2

CAN_USE_GPU = torch.cuda.is_available()

rpn = RPN(45, 40, scalars=(4, 8, 16))
rcnn = RCNN()
rpn_conv = GoogLeNet()
rcnn_conv = GoogLeNet()
if CAN_USE_GPU:
    rpn, rcnn, rpn_conv = rpn.cuda(), rcnn.cuda(), rpn_conv.cuda()
    rcnn_conv = rcnn_conv.cuda()
rpn_conv_param_path = "./params/rpn_conv.pth"
rpn_param_path = "./params/rpn.pth"
rcnn_conv_param_path = "./params/rcnn_conv.pth"
rcnn_param_path = "./params/rcnn.pth"

rpn.load_state_dict(torch.load(rpn_param_path))
rpn_conv.load_state_dict(torch.load(rpn_conv_param_path))
rcnn.load_state_dict(torch.load(rcnn_param_path))
rcnn_conv.load_state_dict(torch.load(rcnn_conv_param_path))


dataset = PikachuDataset("C:/Users/XR/Desktop/pikachu")


def rpn_single_test(img):
    rpn.eval()
    rpn_conv.eval()
    tensor2pil = transforms.ToPILImage()
    with torch.no_grad():
        feature_map = rpn_conv(img.unsqueeze(0).cuda())
        boxes = rpn.region_propose(feature_map, k2=20)
        # print(boxes)
        img = tensor2pil(img)
        img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
        for i, box in enumerate(boxes):
            box = box.int()
            img = cv2.rectangle(img, (box[0].item(), box[1].item()),
                                (box[2].item(), box[3].item()), color=(0, 255, 0), thickness=2)
            # img = cv2.putText(img, "%s %.1f" % (class_name[cls[i]], sorce[i]),
            #                   (box[0].item(), box[1].item()), cv2.FONT_HERSHEY_SIMPLEX,
            #                   0.4, (0, 255, 0), 1)
        cv2.imshow("test", img)
        cv2.waitKey(0)


def rcnn_single_test(img):
    rpn.eval()
    rcnn.eval()
    rpn_conv.eval()
    rcnn_conv.eval()
    tensor2pil = transforms.ToPILImage()
    class_name = ["pikachu", "monster ball"]
    with torch.no_grad():
        feature_map = rpn_conv(img.unsqueeze(0).cuda())
        regions = rpn.region_propose(feature_map, k1=500, k2=100)
        feature_map = rcnn_conv(img.unsqueeze(0).cuda())
        boxes, class_index, cls_pred, mask = rcnn.adjust_regions(feature_map, regions, thresh=0.2)
        img = tensor2pil(img)
        img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
        cv2.imshow("origin", img)
        for i, box in enumerate(boxes):
            if cls_pred[i] < 0.2:
                continue
            box = box.int().cpu()
            # img = cv2.rectangle(img, (box[0].item(), box[1].item()),
            #                     (box[2].item(), box[3].item()), color=(0, 255, 0), thickness=2)
            # img = cv2.putText(img, "%s %.1f" % (class_name[class_index[i]], cls_pred[i]),
            #                   (box[0].item(), box[1].item()), cv2.FONT_HERSHEY_SIMPLEX,
            #                   0.4, (0, 255, 0), 1)
            img = draw_rec(img, list(box.numpy()), "%s  %.2f" % (class_name[class_index[i]], cls_pred[i]))
        img = drwa_mask(img, mask, list(boxes.int().cpu().numpy()))
        cv2.imshow("test", img)
        cv2.waitKey(0)


def train_rpn():
    batch_size, learning_rate, epoch = 1, 0.0005, 1000
    optimizer_rpn = torch.optim.Adam(rpn.parameters(), lr=learning_rate)
    optimizer_backnet = torch.optim.Adam(rpn_conv.parameters(), lr=learning_rate)
    dataloader = data.DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size)

    def backward(loss):
        optimizer_rpn.zero_grad()
        optimizer_backnet.zero_grad()
        loss.backward()
        optimizer_rpn.step()
        optimizer_backnet.step()

    def save():
        print("parameters saving ......")
        torch.save(rpn.state_dict(), rpn_param_path)
        torch.save(rpn_conv.state_dict(), rpn_conv_param_path)

    for epoch_count in range(1, 1+epoch):
        loss_val = 0
        rpn_conv.train()
        rpn.train()
        for imgs, gts, cls, _ in dataloader:
            torch.cuda.empty_cache()
            if CAN_USE_GPU:
                imgs, gts, cls = imgs.cuda(), gts.cuda(), cls.cuda()
            feature_map = rpn_conv(imgs)
            loss = rpn.loss(feature_map, gts, sample_num=32)
            print(loss)
            backward(loss)
            loss_val += loss.item()
        print("epoch: %d  loss: %.3f" % (epoch_count, loss_val))
        save()
        if epoch_count % 100 == 0:
            test_rpn()


def train_rpn2():
    batch_size, learning_rate, epoch = 1, 0.0005, 1000
    optimizer_rpn = torch.optim.Adam(rpn.parameters(), lr=learning_rate)
    dataloader = data.DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size)
    # rpn_conv不训练
    rpn_conv.eval()
    for param in rpn_conv.parameters():
        param.requires_grad = False

    def backward(loss):
        optimizer_rpn.zero_grad()
        loss.backward()
        optimizer_rpn.step()

    def save():
        print("parameters saving ......")
        torch.save(rpn.state_dict(), rpn_param_path)
    min_loss = 99
    for epoch_count in range(1, 1+epoch):
        loss_val = 0
        rpn.train()
        for imgs, gts, cls in dataloader:
            torch.cuda.empty_cache()
            if CAN_USE_GPU:
                imgs, gts, cls = imgs.cuda(), gts.cuda(), cls.cuda()
            feature_map = rpn_conv(imgs)
            loss = rpn.loss(feature_map, gts, sample_num=32)
            print(loss)
            backward(loss)
            loss_val += loss.item()
        print("epoch: %d  loss: %.3f" % (epoch_count, loss_val))
        if min_loss > loss_val:
            min_loss = loss_val
            print("min loss: %.3f" % min_loss)
            save()
        # if epoch_count % 10 == 0:
        #     test_rpn()


def test_rpn():
    for i in range(0, 32):
        rpn_single_test(dataset[i][0])


def train_rcnn():
    cpu = torch.device("cpu")
    gpu = torch.device("cuda:0")

    batch_size, learning_rate, epoch = 1, 0.0005, 30
    rcnn.to(cpu)
    rcnn_conv.to(cpu)
    optimizer_rcnn = torch.optim.Adam(rcnn.parameters(), lr=learning_rate)
    optimizer_backnet = torch.optim.Adam(rcnn_conv.parameters(), lr=learning_rate)
    rcnn.to(gpu)
    rcnn_conv.to(gpu)
    dataloader = data.DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size)

    # rpn不训练
    rpn.eval()
    rpn.freeze()
    # rpn_conv不训练
    rpn_conv.eval()
    for param in rpn_conv.parameters():
        param.requires_grad = False

    def backward(loss):
        optimizer_rcnn.zero_grad()
        optimizer_backnet.zero_grad()
        loss.to(cpu)
        loss.backward()
        rcnn.to(cpu)
        rcnn_conv.to(cpu)
        optimizer_rcnn.step()
        optimizer_backnet.step()

    def save():
        print("parameters saving ......")
        torch.save(rcnn.state_dict(), rcnn_param_path)
        torch.save(rcnn_conv.state_dict(), rcnn_conv_param_path)

    def rpn2gpu():
        rcnn_conv.to(cpu)
        rcnn.to(cpu)
        rpn_conv.to(gpu)
        rpn.to(gpu)
        torch.cuda.empty_cache()

    def rcnn2gpu():
        rpn_conv.to(cpu)
        rpn.to(cpu)
        rcnn_conv.to(gpu)
        rcnn.to(gpu)
        torch.cuda.empty_cache()

    for epoch_count in range(1, 1+epoch):
        loss_val = 0
        rcnn_conv.train()
        rcnn.train()
        for imgs, gts, cls, mask in dataloader:
            torch.cuda.empty_cache()
            if CAN_USE_GPU:
                imgs, gts, cls = imgs.cuda(), gts.cuda(), cls.cuda()
            # print(type(mask))
            rpn2gpu()
            feature_map = rpn_conv(imgs)
            regions = rpn.region_propose(feature_map, k1=5000, k2=500)
            rcnn2gpu()
            feature_map = rcnn_conv(imgs)
            loss = rcnn.loss(feature_map, regions, gts, cls, mask, sample_num=120)
            print(loss)
            backward(loss)
            loss_val += loss.item()
        print("epoch: %d  loss: %.3f" % (epoch_count, loss_val))
        save()
        # if epoch_count % 10 == 0:
        #     test_rcnn()


def train_rcnn2():
    batch_size, learning_rate, epoch = 1, 0.0005, 1000
    optimizer_rcnn = torch.optim.Adam(rcnn.parameters(), lr=learning_rate)
    dataloader = data.DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size)
    # rpn不训练
    rpn.eval()
    rpn.freeze()
    # rpn_conv不训练
    rpn_conv.eval()
    for param in rpn_conv.parameters():
        param.requires_grad = False

    def backward(loss):
        optimizer_rcnn.zero_grad()
        loss.backward()
        optimizer_rcnn.step()

    def save():
        print("parameters saving ......")
        torch.save(rcnn.state_dict(), rcnn_param_path)
    min_loss = 99
    for epoch_count in range(1, 1+epoch):
        loss_val = 0
        rcnn.train()
        for imgs, gts, cls in dataloader:
            torch.cuda.empty_cache()
            if CAN_USE_GPU:
                imgs, gts, cls = imgs.cuda(), gts.cuda(), cls.cuda()
            feature_map = rpn_conv(imgs)
            regions = rpn.region_propose(feature_map, k1=1000, k2=500)
            loss = rcnn.loss(feature_map, regions, gts, cls, sample_num=60)
            print(loss)
            backward(loss)
            loss_val += loss.item()
        print("epoch: %d  loss: %.3f" % (epoch_count, loss_val))
        if min_loss > loss_val:
            min_loss = loss_val
            print("min loss: %.3f" % min_loss)
            save()


def test_rcnn():
    for i in range(0, 32):
        rcnn_single_test(dataset[i][0])


if __name__ == '__main__':
    # train_rpn()
    # test_rpn()
    # train_rcnn()
    test_rcnn()
    # train_rpn2()
    # train_rcnn2()