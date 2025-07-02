from backbones.clip import CLIPTextEncoder as text_encoder
from backbones.utils import tokenize
import torch
from segment_anything import build_sam_vit_b

import argparse
import os

import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import random
from torchvision import transforms
import time
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

import cv2


def draw_mask(image, mask_generated):
    masked_image = image.copy()

    masked_image = np.where(mask_generated.astype(int),
                          np.array([0,0,255], dtype='uint8'),
                          masked_image)

    masked_image = masked_image.astype(np.uint8)

    return cv2.addWeighted(image, 0.8, masked_image, 0.2, 0)


def save_mask(mask, mask_name, save_dir):
    im = Image.fromarray(np.uint8(mask * 255))
    img_name = mask_name.split(os.sep)[-1]
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    im.save(os.path.join(save_dir, imidx + '.png'))


def cv_random_flip(img, label):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label


class GetData(Dataset):

    # 初始化为整个class提供全局变量，为后续方法提供一些量
    def __init__(self, img_dir, mask_dir, hardcase_dir, transformsize=None, train=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.hardcase_dir = hardcase_dir
        self.train = train
        self.img_path_list = sorted(os.listdir(self.img_dir))
        self.mask_path_list = sorted(os.listdir(self.mask_dir))
        self.hardcase_list = sorted(os.listdir(self.hardcase_dir))
        self.img_transform = transforms.Compose([
            transforms.Resize((transformsize, transformsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((transformsize, transformsize), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
            
    def __getitem__(self, idx):
        assert len(self.img_path_list) == len(self.mask_path_list)
        img_name = self.img_path_list[idx]  # 只获取了文件名
        mask_name = self.mask_path_list[idx]
        is_hardcase = mask_name in self.hardcase_list
        img_item_path = os.path.join(self.img_dir, img_name) # 每个图片的位置
        mask_item_path = os.path.join(self.mask_dir, mask_name)
        img = Image.open(img_item_path).convert("RGB")
        gt = Image.open(mask_item_path).convert("L")
        if self.train is not None:
            img, gt = cv_random_flip(img, gt)

        img = self.img_transform(img)
        gt = self.gt_transform(gt)
        gt[gt > 0] = 1.
        
        img_metas = {}
        img_metas['img_path'] = img_item_path
        img_metas['img_name'] = img_name
        img_metas['mask_path'] = mask_item_path
        img_metas['mask_name'] = mask_name

        return img, gt,  is_hardcase, img_metas

    def __len__(self):
        return len(self.img_path_list)


def eval_epoch(model, val_loader, text_features, device):
    model.eval()
    
    # save_mask_path = 'autodl-tmp/text-guided-sam/output_mask/text_guided_sam_ffn/mask'
    # save_visual_path = 'autodl-tmp/text-guided-sam/output_mask/text_guided_sam_ffn/visual'
    
    pbar = tqdm(total=len(val_loader), leave=True, desc='val')
    pixel_TP, pixel_TP_hardcase = 0, 0
    pixel_TP_and_FP, pixel_TP_and_FP_hardcase = 0, 0
    pixel_TP_and_FN, pixel_TP_and_FN_hardcase = 0, 0
    with torch.no_grad():
        for inp, gt, is_hardcase, img_metas in val_loader:
            inp = inp.to(device)
            # gt = gt.to(device)
            pred, _ = model(inp, text_features)
            
            # gt = gt.squeeze().detach().cpu().numpy()
            # current_pixel_TP_and_FN = gt.sum(0).sum(0)
            
            ori_gt= cv2.imread(img_metas['mask_path'][0], 0)
            ori_gt[ori_gt > 0] = 1.
            current_pixel_TP_and_FN = ori_gt.sum(0).sum(0)

            pred = torch.sigmoid(pred)
            # pred = transforms.functional.resize(pred,  ori_gt.shape[:2], transforms.InterpolationMode.BILINEAR)
            pred = pred.squeeze().detach().cpu().numpy()
            pred = np.where(pred > 0.5, 1, 0)
            current_pixel_TP_and_FP = pred.sum(0).sum(0)

            current_pixel_TP = (ori_gt * pred).sum(0).sum(0)
            # current_pixel_TP = (gt * pred).sum(0).sum(0)

            pixel_TP += current_pixel_TP
            pixel_TP_and_FP += current_pixel_TP_and_FP
            pixel_TP_and_FN += current_pixel_TP_and_FN
            
            if is_hardcase:
                pixel_TP_hardcase += current_pixel_TP
                pixel_TP_and_FP_hardcase += current_pixel_TP_and_FP
                pixel_TP_and_FN_hardcase += current_pixel_TP_and_FN
            
            # save_mask(pred, img_metas['img_name'][0], save_mask_path)
            # save_mask(orisize_sinet_mask, mask_name[0], '/media/dl/8t1/zp/eccv/SAM-SINet-Twostage/MASK/sinet/')

            # ori_image = np.array(Image.open(img_metas['img_path'][0]).convert('RGB'))
            # pred = np.array([pred for i in range(3)]).transpose(1, 2, 0)
            # segmented_image = draw_mask(ori_image, pred)
            # name = img_metas['img_name'][0]
            # cv2.imwrite(os.path.join(save_visual_path, name), segmented_image)

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()
        
        if pixel_TP_and_FP == 0:
            pixel_precision = 0
        else:
            pixel_precision = pixel_TP / pixel_TP_and_FP
        pixel_recall = pixel_TP / pixel_TP_and_FN
        mIoU = pixel_TP / (pixel_TP_and_FP + pixel_TP_and_FN - pixel_TP)
        
        if pixel_TP_and_FP_hardcase == 0:
            pixel_precision_hardcase = 0
        else:
            pixel_precision_hardcase = pixel_TP_hardcase / pixel_TP_and_FP_hardcase
        pixel_recall_hardcase = pixel_TP_hardcase / pixel_TP_and_FN_hardcase
        mIoU_hardcase = pixel_TP_hardcase / (pixel_TP_and_FP_hardcase + pixel_TP_and_FN_hardcase - pixel_TP_hardcase)

        val_info = f'precison: {pixel_precision:.4f}, recall: {pixel_recall:.4f}, miou: {mIoU:.4f}' + '\n'
        val_info = val_info + f'precision_hardcase: {pixel_precision_hardcase:.4f}, recall_hardcase: {pixel_recall_hardcase:.4f}, miou_hardcase: {mIoU_hardcase:.4f}'
        print(val_info)

    return mIoU


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

global log_info
device = torch.device(args.device)
print(f'Using device {device}')

val_data = GetData('/media/dl/8t1/datasets/UW-Bench-v2-4/test/JPEGImages',
                   '/media/dl/8t1/datasets/UW-Bench-v2-4/test/SegmentationClass',
                   '/media/dl/8t1/datasets/UW-Bench-v2-4/test/Hardcase', transformsize=1024)
print('test data size:', len(val_data))
val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=8)

text_encoder = text_encoder(embed_dim=512)
text_encoder.init_weights(pretrained='/media/dl/8t1/zp/TVAFE/ViT-B-16.pt')
text_encoder.to(device)
text = ['A photo without water', 'A photo of a wet road', 'A photo of water', 'Turbid brownish-yellow water', 'Semi-transparent bluish-gray water',
        'Pitch-black water with iridescent oil film', 'Crystal-clear water with glass-like transparency',
       'Mottled grayish-green water', 'Opaque milky-white industrial wastewater', 'Noon-highlighted amber turbid water with glinting particulate reflections',
       'Rainstorm-night carlight-illuminated obsidian specular water reflecting crimson brake lights',
       'Morning-haze-dimmed ochre slurry water', 'HDR-backlit indigo specular water']
for name, para in text_encoder.named_parameters():
    para.requires_grad_(False)
text_encoder.eval()
text_embedding = tokenize(text).to(device)
with torch.no_grad():
    text_features = text_encoder(text_embedding).float()
    print('text_features shape:', text_features.shape)

model = build_sam_vit_b(checkpoint='/media/dl/8t1/zp/TVAFE/tvafe_best.pth').to('cuda:0')
for name, para in model.named_parameters():
    para.requires_grad_(False)
model_total_params = sum(p.numel() for p in model.parameters())
model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

threshold = 0.5

mIoU = eval_epoch(model, val_loader, text_features, device)
print('done!')
