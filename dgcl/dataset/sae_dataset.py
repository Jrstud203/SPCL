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
from . import _transforms as T


class SaeDataset(Dataset):
    def __init__(self, root_dir, phase="train", image_size=256, transforms=None, labeled_num=3):
        self.image_size = image_size
        self.pre_transform = self.pre_transform()
        self.augmentation = self.augmentation_transform()
        if phase != 'train':
            self.test_transform = self.test_transform()

        self.dataroot = root_dir
        self.phase = phase
        self.paths = getPath(root_dir, phase, labeled_num)

    def __len__(self):
        return len(self.paths[0][0]) + len(self.paths[1][0])

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

            return torch.stack(img_tf, dim=0).float(), torch.stack(label_tf, dim=0).squeeze(1).long(), \
                   torch.stack(ignore_tf, dim=0).squeeze(1).long()
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


def getPath(root, phase, labeled_num):
    """ get the paths of samples for each phase
        return: list([vendor_img,vendor_label,[index_i],vendors])，特别的把属于同一个index的所有路径放到一起
        vendors -- 厂商
        index -- ED or ES
        vendor_img -- the paths of images
        vendor_label -- the paths of labels
    """
    Vendor = ['A/', 'B/', 'C/', 'D/']
    classes = ['ED/', 'ES/']
    paths = []
    seq_paths = []  # 用于确定哪些位置为标签数据
    for index in range(2):
        vendor_img, vendor_label, vendors, tmp_img = [], [], [], []
        for vendorIndex in range(4):
            img_dir = os.path.join(root + Vendor[vendorIndex] + classes[index], '%s_img'%phase)
            tmp_img_paths = walk(dirname=img_dir)
            if phase == "train":
                img_dir = img_dir + '/slices'
            img_paths = walk(dirname=img_dir)

            label_dir = os.path.join(root + Vendor[vendorIndex] + classes[index], '%s_label'%phase)
            if phase == "train":
                label_dir = label_dir + '/slices'
            label_paths = walk(dirname=label_dir)
            vendor_img.append(img_paths)
            vendor_label.append(label_paths)
            tmp_img.append(tmp_img_paths)
            vendors.append(np.array([vendorIndex]).repeat(len(img_paths)))
        vendor_img = np.concatenate(vendor_img, 0)
        vendor_label = np.concatenate(vendor_label, 0)
        vendors = np.concatenate(vendors, 0)
        vendor_img = sorted(vendor_img)
        vendor_label = sorted(vendor_label)
        tmp_img = np.concatenate(tmp_img, 0)
        tmp_img = sorted(tmp_img)  # 原来以卷为单位的数据排列方式

        paths.append([vendor_img, vendor_label, [index], vendors])
        seq_paths.append(tmp_img)

    if phase == "train":
        if not os.path.exists(root + f'train_stid_{labeled_num}.npy'):
            get_lbid(root, phase, labeled_num, seq_paths, paths)
    print('%d data is created' % (len(paths[0][0]) + len(paths[1][0])))
    return paths


# 确定标签id
def get_lbid(root="../MM/", phase="train", labeled_num=4, seq_paths=None, paths=None):
    # 构造采样列表
    ed_size = len(paths[0][0])
    if labeled_num > 4:
        ab_size = (labeled_num - 4)
        num_a = ab_size // 2
        lb_volid = list(range(1, 1 + num_a)) + list(range(75, 75 + (ab_size - num_a))) + \
                   list(range(150, 150 + 2)) + list(range(160, 160 + 2))
    elif labeled_num == 4:
        lb_volid = list(range(1, 1 + 1)) + list(range(75, 75 + 1)) + \
                   list(range(150, 150 + 1)) + list(range(160, 160 + 1))
    else:
        raise ValueError("labeled_num is at least 4.")

    cl = ['ED', 'ES']
    target_id = []
    for vi in lb_volid:
        img_path = seq_paths[0][vi]
        inner_no = os.path.splitext(os.path.basename(img_path))[0]
        s = os.path.split(img_path)[0].split('/')
        vd_no = s[-3]
        # 根据原编号确定组号，类别，和对应编号
        for index in range(2):
            tg_prefix = f'{vd_no}_{cl[index]}_{inner_no}_'
            flag = 0
            for li in range(len(paths[index][0])):
                # gpi_ED/ES_j_k
                if os.path.splitext(os.path.basename(paths[index][0][li]))[0].startswith(tg_prefix):
                    target_id.append(li if index == 0 else li + ed_size)
                    flag = 1
            if flag == 0:
                print(f"{img_path} is not found")

    # 保存target_id方便未来实验的调用
    np.save(root + f'train_stid_{labeled_num}.npy', target_id)
    print(f'saved train_stid_{labeled_num}.npy')


def get_cardiac_loaders(root_dir=r'F:/datasets/ACDC/', labeled_num=7, labeled_bs=8, batch_size=24, batch_size_val=16,
                     num_workers=4, worker_init_fn=None, train_transforms=None, val_transforms=None, num_classes=4):

    db_train = SaeDataset(root_dir=root_dir, phase="train", image_size=513, labeled_num=labeled_num)
    db_val = SaeDataset(root_dir=root_dir, phase="val", image_size=513, labeled_num=labeled_num)

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
            unlabeled_idxs.pop(index)
        batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)
        train_loader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=num_workers,
                                  pin_memory=True, worker_init_fn=worker_init_fn, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(db_train, batch_size=batch_size, num_workers=num_workers,
                                  pin_memory=True, worker_init_fn=worker_init_fn, collate_fn=collate_fn)
    val_loader = DataLoader(db_val, batch_size=batch_size_val, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, val_loader







