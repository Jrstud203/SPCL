# this file is similar as base_dataset.py,
# except that only use ResizeCrop and normalize to data,
# and obtain mrf function

# from .image_folder import make_dataset
import re

from torch.utils.data import Dataset, DataLoader
import os
# from .data_transform import centerAlignAndDataAug, centerAlignAndDataAugVal
import numpy as np
import torch
from ._transforms import build_transforms
from ._samplers import TwoStreamBatchSampler
from .image_folder import make_dataset, walk
from .format_conversion import collate_fn
from .data_transform import centerAlignAndDataAugVal
from copy import deepcopy
from . import _transforms as T
from torchvision.transforms import *


class AcdcalDataset(Dataset):
    def __init__(self, root_dir, phase="train", image_size=256, transforms=None, labeled_num=3, sup=False):
        self.image_size = image_size
        self.pre_transform = self.pre_transform()
        self.augmentation = self.augmentation_transform()
        if phase != 'train':
            self.test_transform = self.test_transform()

        self.dataroot = root_dir
        self.phase = phase
        self.sup = sup
        self.paths = getPath(root_dir, phase, labeled_num, sup=sup)

    def __len__(self):
        return len(self.paths[0]) if self.phase == 'train' and self.sup else len(self.paths[0][0]) + len(self.paths[1][0])

    def __getitem__(self, index):
        # type --- 0表示'ED/'，1表示'ES/'
        index = int(index)
        if index < len(self.paths[0][0]):
            classes_index = 0
        else:
            index = index - len(self.paths[0][0])
            classes_index = 1

        img_path, label_path, types, vendor = self.paths[classes_index][0][index], self.paths[classes_index][1][index], \
                                              self.paths[classes_index][2], self.paths[classes_index][3][index]

        assert self.paths_match(label_path, img_path), "The label_path %s and img_path %s don't match." % (
        label_path, img_path)
        assert types[0] == classes_index, "obtained dataset is incorrect."
        # img_ori, label_ori, index_ori, types, vendors 是一一对应的
        img, label = np.load(img_path), np.load(label_path)  # 3维np数组
        if np.ndim(img) == 2:
            img = img[None, ...]
            label = label[None, ...]

        img, label = centerAlignAndDataAugVal(img, label)
        # 将图片转换为[0,1]
        img = img / 255.0
        img, label = img.numpy(), label.numpy()

        if self.phase == 'train':
            img_tf, label_tf, ignore_tf = [], [], []
            for img_i in range(img.shape[0]):
                # 弱增强: 缩放+rgb
                sample = self.pre_transform({'image': img[img_i], 'label': label[img_i]})
                # 强增强: crop+flip+colorjitter
                sample = self.augmentation(sample)
                ignore_mask = torch.zeros_like(sample['label'])
                ignore_mask[sample['label'] == 254] = 255

                img_tf.append(sample['image'])
                label_tf.append(sample['label'])
                ignore_tf.append(ignore_mask)

            return torch.stack(img_tf, dim=0).float(), torch.stack(label_tf, dim=0).squeeze(1).long(),\
                   torch.stack(ignore_tf,dim=0).squeeze(1).long()
        else:
            img_tf, label_tf = [], []
            for img_i in range(img.shape[0]):
                # 缩放到模型指定输入大小
                sample = self.pre_transform({'image': img[img_i], 'label': label[img_i]})
                img_tf.append(sample['image'])
                sample = self.test_transform({'image': img[img_i], 'label': label[img_i]})
                label_tf.append(sample['label'])

        return torch.stack(img_tf, dim=0).float(), torch.stack(label_tf, dim=0).squeeze(1).long()  # 513*513, 224*224

    def paths_match(self, label_path, img_path):
        filename1_without_ext = os.path.splitext(os.path.basename(label_path))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(img_path))[0]
        return filename1_without_ext == filename2_without_ext

    # @staticmethod
    def augmentation_transform(self):
        return T.Compose([
            T.RandomCrop(size=[self.image_size, self.image_size]),
            T.RandomFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        ])

    def pre_transform(self):
        return T.Compose([
            T.RandomGenerator(output_size=[self.image_size, self.image_size], p_flip=0.0, p_rot=0.0),
            T.ToRGB(),
        ])

    def test_transform(self):
        return T.Compose([
            T.RandomGenerator(output_size=[224, 224], p_flip=0.0, p_rot=0.0),
            T.ToRGB(),
        ])


def getPath(root, phase, labeled_num, sup=False):
    """ get the paths of samples for each phase
        return: list([vendor_img,vendor_label,[index_i],vendors])，特别的把属于同一个index的所有路径放到一起
        vendors -- 厂商
        index -- ED or ES
        vendor_img -- the paths of images
        vendor_label -- the paths of labels
    """
    Vendor = ['gp1/', 'gp2/', 'gp3/', 'gp4/', 'gp5/']
    classes = ['ED/', 'ES/']
    paths = []
    for index in range(2):
        vendor_img, vendor_label, vendors = [], [], []
        for vendorIndex in range(len(Vendor)):
            img_dir = os.path.join(root + Vendor[vendorIndex] + classes[index], '%s_img'%phase)
            if phase == "train":
                img_dir = img_dir+'/slices'
            img_paths = walk(dirname=img_dir)

            label_dir = os.path.join(root + Vendor[vendorIndex] + classes[index], '%s_label'%phase)
            if phase == "train":
                label_dir = label_dir+'/slices'
            label_paths = walk(dirname=label_dir)
            vendor_img.append(img_paths)
            vendor_label.append(label_paths)
            vendors.append(np.array([vendorIndex]).repeat(len(img_paths)))
        vendor_img = np.concatenate(vendor_img, 0)
        vendor_label = np.concatenate(vendor_label, 0)
        vendors = np.concatenate(vendors, 0)
        vendor_img = sorted(vendor_img)
        vendor_label = sorted(vendor_label)

        paths.append([vendor_img, vendor_label, [index], vendors])

    if phase == "train":
        if not os.path.exists(root + f'train_stid_{labeled_num}.npy'):
            get_lbid(root, phase, labeled_num, paths)
        if sup:
            paths_fusion = [paths[0][0] + paths[1][0], paths[0][1] + paths[1][1]]  # [img, label]
            if labeled_num != 70:
                train_edesid = np.load(root + f'train_stid_{labeled_num}.npy').astype(np.uint16)
                labeled_idxs = np.sort(train_edesid, axis=0).tolist()
                paths_fusion = [[paths_fusion[0][i] for i in labeled_idxs], [paths_fusion[1][i] for i in labeled_idxs]]

            print('%d data is created' % (len(paths_fusion[0])))
            return paths_fusion
    print('%d data is created' % (len(paths[0][0]) + len(paths[1][0])))
    return paths


# 确定标签id
def get_lbid(root="../ACDC_aligned/", phase="train", labeled_num=3, paths=None):
    # 获取标签训练数据列表
    names_file = root + f'{phase}.list'
    with open(names_file, 'r') as f:
        sample_list = f.readlines()
    sample_list = [item.replace('\n', '') for item in sample_list][:labeled_num * 2]
    ed_size = len(paths[0][0])

    cl = ['ED', 'ES']
    # 根据原编号确定组号，类别，和对应编号
    target_id = []
    for si in range(len(sample_list)):
        # content: patient099_frame01
        s = [int(s) for s in re.findall(r'-?\d+\.?\d*', sample_list[si])]
        # if s[1] == 2:
        #     continue
        cl_i = s[1] - 1  # 0-ED, 1-ES
        # 确定patient099_frame01在哪个路径上
        totalIndex = s[0]
        if 1 <= totalIndex <= 20:
            # 为了从0编号
            case_no = totalIndex - 1
            gp_id = 1

        elif 21 <= totalIndex <= 40:
            case_no = totalIndex - 21
            gp_id = 2

        elif 41 <= totalIndex <= 60:
            case_no = totalIndex - 41
            gp_id = 3

        elif 61 <= totalIndex <= 80:
            case_no = totalIndex - 61
            gp_id = 4

        else:
            case_no = totalIndex - 81
            gp_id = 5

        # 匹对时，就是对应的编号
        tg_prefix = f'gp{gp_id}_{cl[cl_i]}_{case_no}'
        flag = 0
        for li in range(len(paths[cl_i][0])):
            # gpi_ED/ES_j_k
            if os.path.splitext(os.path.basename(paths[cl_i][0][li]))[0].startswith(tg_prefix):
                target_id.append(li if cl_i == 0 else li+ed_size)
                flag = 1
        if flag == 0:
            print(f"{sample_list[si]} is not found")
    # 保存target_id方便未来实验的调用
    np.save(root + f'train_stid_{labeled_num}.npy', target_id)
    print(f'saved train_stid_{labeled_num}.npy')


def get_acdcal_loaders(root_dir=r'F:/datasets/ACDC/', labeled_num=7, labeled_bs=8, batch_size=24, batch_size_val=16,
                     num_workers=4, worker_init_fn=None, train_transforms=None, val_transforms=None, num_classes=4, sup=False):

    db_train = AcdcalDataset(root_dir=root_dir, phase="train", image_size=513, labeled_num=labeled_num)
    db_val = AcdcalDataset(root_dir=root_dir, phase="val", image_size=513, labeled_num=labeled_num)

    if labeled_bs < batch_size:
        train_edesid = np.load(root_dir + f'train_stid_{labeled_num}.npy').astype(np.uint16)
        labeled_idxs = np.sort(train_edesid, axis=0)
        if labeled_num >= 10:
            labeled_idxs = labeled_idxs.tolist() * 3
        else:
            labeled_idxs = labeled_idxs.tolist() * 4
        print(f'use {labeled_num} labeled data, has {train_edesid.shape[0]} slices')

        unlabeled_idxs = list(range(0, len(db_train)))
        for counter, index in enumerate(labeled_idxs):
            index = index - counter
            if index < 0:
                break
            unlabeled_idxs.pop(int(index))
        batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)
        train_loader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=num_workers,
                                  pin_memory=True, worker_init_fn=worker_init_fn, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(db_train, batch_size=batch_size, num_workers=num_workers,
                                  pin_memory=True, worker_init_fn=worker_init_fn, collate_fn=collate_fn)
    val_loader = DataLoader(db_val, batch_size=batch_size_val, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, val_loader







