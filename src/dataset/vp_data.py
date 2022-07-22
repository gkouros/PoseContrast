import os
import random
import numpy as np
from PIL import Image, ImageFilter
import pandas as pd

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import cv2

try:
    from .data_utils import process_viewpoint_label, TransLightning, resize_pad, random_crop
except:
    from data_utils import process_viewpoint_label, TransLightning, resize_pad, random_crop

# ImageNet statistics
imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([[-0.5675, 0.7192, 0.4009],
                                [-0.5808, -0.0045, -0.8140],
                                [-0.5836, -0.6948, 0.4203]])
}

categories = [
            'aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair',
            'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']

blacklist = {category: [] for category in categories}
blacklist['motorbike'] += ['n03791053_8626.JPEG', 'n03791053_8438.JPEG', 'n03790512_11192']

# Define normalization and random disturb for input image
disturb = TransLightning(0.1, imagenet_pca['eigval'], imagenet_pca['eigvec'])
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class Pascal3D_NeMo(data.Dataset):
    def __init__(self, root_dir, train=False, input_dim=224, offset=0, rot=0,
                 bs=32, cls_choice=None, occlusion_level=0, pose_batch=None):
        if train:
            self.root_dir = os.path.join(root_dir, 'PASCAL3D_train_NeMo')
            self.data_pendix = ''
        else:
            if occlusion_level > 0:
                self.root_dir = os.path.join(root_dir, 'PASCAL3D_OCC_NeMo')
                self.data_pendix = 'FGL%d_BGL%d' % (occlusion_level, occlusion_level)
            else:
                self.root_dir = os.path.join(root_dir, 'PASCAL3D_NeMo')
                self.data_pendix = ''

        self.input_dim = input_dim
        self.offset = offset
        self.cls_name = cls_choice
        self.train = train
        self.rot = rot

        self.image_path = os.path.join(self.root_dir, 'images')

        if not cls_choice:
            cls_choice = categories

        self.image_names = [{'category': cat, 'name': name}
                            for cat in cls_choice
                            for name in os.listdir(os.path.join(self.image_path, cat + self.data_pendix))
                            if name not in blacklist[cat]]

        self.annotation_path = os.path.join(self.root_dir, 'annotations')

        self.pose_batch = pose_batch
        self.bs = bs
        if pose_batch:
            self.pose_index = {}
            for i in range(12):
                self.pose_index[i] = []
            for i in range(len(self.df)):
                pose_cls = int(self.df.iloc[i]['azimuth'] // 30)
                self.pose_index[pose_cls].append(i)

        if train:
            self.im_transform = transforms.Compose([
                transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
                disturb
            ])
        else:
            self.im_transform = transforms.Compose([transforms.ToTensor(), normalize])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.pose_batch:
            batch_index = int(idx // self.bs)
            cls_index = batch_index % 12
            sample_index = (self.bs * idx // (12 * self.bs) + idx % self.bs) % len(self.pose_index[cls_index])
            idx = self.pose_index[cls_index][sample_index]

        img_name = self.image_names[idx]['name']
        category = self.image_names[idx]['category']
        cls_index = np.array([categories.index(category)])
        cls_index = torch.from_numpy(cls_index).long()

        # TODO read bbox instead
        annotation_file = np.load(os.path.join(
            self.annotation_path, category + self.data_pendix, img_name.split('.')[0] + '.npz'), allow_pickle=True)
        upper, lower, left, right, _, _ = annotation_file['box_obj']
        if left > right or lower < upper:
            raise ValueError('Sample %s[%s] has invalid bbox' % (category, img_name))

        # load gt viewpoint label
        label = np.array([annotation_file['azimuth'], annotation_file['elevation'], annotation_file['theta']])
        label *= 180.0 / np.pi

        # load images in RGB format
        img_path = os.path.join(
            self.root_dir, 'images', category + self.data_pendix, img_name)
        im = Image.open(img_path).convert('RGB')
        im_pos = im.copy()

        if self.train:
            # gaussian blur
            if min(right - left, lower - upper) > 224 and np.random.random() > 0.5:
                blur_size = np.random.randint(low=1, high=5)
                im = im.filter(ImageFilter.GaussianBlur(blur_size))

            # crop the original image with 2D box jittering
            im = random_crop(im, left, upper, right - left, lower - upper)
            im_pos = random_crop(im_pos, left, upper, right - left, lower - upper)
            im_pos = resize_pad(im_pos, self.input_dim)
            im_pos = self.im_transform(im_pos)

            # get rotated image for regularization
            r = random.choice([-self.rot, self.rot])
            im_rot = im.rotate(r)
            im_rot = resize_pad(im_rot, self.input_dim)
            im_rot = self.im_transform(im_rot)
            label_rot = label.copy()
            label_rot[2] = label_rot[2] + r
            label_rot[2] += 360 if label_rot[2] < -180 else (-360 if label_rot[2] > 180 else 0)
            label_rot = process_viewpoint_label(label_rot, self.offset)

            # get flipped image for regularization
            im_flip = im.transpose(Image.FLIP_LEFT_RIGHT)
            im_flip = resize_pad(im_flip, self.input_dim)
            im_flip = self.im_transform(im_flip)
            label_flip = label.copy()
            label_flip[0] = 360 - label_flip[0]
            label_flip[2] = -label_flip[2]
            label_flip = process_viewpoint_label(label_flip, self.offset)

            # image to tensor
            im = resize_pad(im, self.input_dim)
            im = self.im_transform(im)
        else:
            # crop the original image with GT box
            im = im.crop((left, upper, right, lower))
            im_flip = im.transpose(Image.FLIP_LEFT_RIGHT)
            im = resize_pad(im, self.input_dim)
            im_flip = resize_pad(im_flip, self.input_dim)
            im = self.im_transform(im)
            im_flip = self.im_transform(im_flip)

        # change angle labels to values >= 0
        label = process_viewpoint_label(label, self.offset)

        if self.train:
            return cls_index, im, label, im_flip, label_flip, im_rot, label_rot, im_pos
        else:
            return cls_index, im, label, im_flip

class KITTI3D_NeMo(data.Dataset):
    def __init__(self, root_dir, input_dim=224, offset=0, occlusion_level=0):
        assert occlusion_level in [0, 1, 2]

        self.root_dir = os.path.join(root_dir, 'KITTI3D_val_NeMo')
        self.input_dim = input_dim
        self.offset = offset

        self.image_path = os.path.join(self.root_dir, 'images')
        occlusion_levels = {-1: '', 0: '_fully_visible', 1: '_partly_occluded', 2: '_largely_occluded'}
        list_path = os.path.join(self.root_dir, 'lists', 'car%s' % occlusion_levels[occlusion_level], 'car%s.txt' % occlusion_levels[occlusion_level])
        self.image_names = [x.strip() for x in open(os.path.join(list_path)).readlines()]

        self.annotation_path = os.path.join(self.root_dir, 'annotations')
        self.im_transform = transforms.Compose([transforms.ToTensor(), normalize])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        category = 'car'
        cls_index = np.array([categories.index(category)])
        cls_index = torch.from_numpy(cls_index).long()

        # read bbox
        annotation_file = np.load(os.path.join(
            self.annotation_path, category, img_name.split('.')[0] + '.npz'), allow_pickle=True)
        upper, lower, left, right, _, _ = annotation_file['box_obj']

        # load gt viewpoint label
        label = np.array([annotation_file['azimuth'], annotation_file['elevation'], annotation_file['theta']])
        label *= 180.0 / np.pi

        # load images in RGB format
        img_path = os.path.join(self.image_path, category, img_name)
        im = Image.open(img_path).convert('RGB')
        im_pos = im.copy()

        if left > right or lower < upper:
            raise ValueError('Sample %s[%s] has invalid bbox' % ('car', img_name))

        # crop the original image with GT box
        # im = im.crop((left, upper, right, lower))
        im_flip = im.transpose(Image.FLIP_LEFT_RIGHT)
        im = resize_pad(im, self.input_dim)
        im_flip = resize_pad(im_flip, self.input_dim)
        im = self.im_transform(im)
        im_flip = self.im_transform(im_flip)

        # change angle labels to values >= 0
        label = process_viewpoint_label(label, self.offset)

        return cls_index, im, label, im_flip

class Pascal3D(data.Dataset):
    def __init__(self, root_dir, annotation_file, train=True, input_dim=224, offset=0, shot=None, train_feat=False,
                 cls_choice=None, idx_choice=None, rot=0, train_cls=None, pose_batch=False, bs=32):

        self.root_dir = root_dir
        self.input_dim = input_dim
        self.train = train
        self.offset = offset
        self.rot = rot

        # load the data frame for annotations
        frame = pd.read_csv(os.path.join(root_dir, annotation_file))
        frame = frame[frame.elevation != 90]
        frame = frame[frame.difficult == 0]
        self.cls_names = np.unique(frame.cls_name).tolist()

        # align viewpoint annotation in ObjectNet3D to the same format as Pascal3D
        if 'ObjectNet3D' in annotation_file:
            frame.azimuth = (360. + frame.azimuth) % 360

        # filter out occluded or truncated samples in evaluation
        if train or train_feat:
            frame = frame[frame.set == 'train']
        else:
            frame = frame[frame.set == 'val']
            frame = frame[frame.truncated == 0]
            frame = frame[frame.occluded == 0]
            frame = frame[frame.has_keypoints == 1]

        # select classes for zero-shot training and validation
        if cls_choice is not None:
            frame = frame[~frame.cls_name.isin(cls_choice)] if train else frame[frame.cls_name.isin(cls_choice)]

        # select class for cls-specific training
        if train_cls is not None:
            frame = frame[frame.cls_name.isin(train_cls)] if isinstance(train_cls, list) else frame[frame.cls_name == train_cls]

        # randomly select few-shot training samples
        if train and shot is not None:
            classes = np.unique(frame.cls_name)
            fewshot_frame = []
            for cls in classes:
                fewshot_frame.append(frame[frame.cls_name == cls].sample(n=shot))
            frame = pd.concat(fewshot_frame)

        # select training samples according to pre-computed idx
        if train and idx_choice is not None:
            frame = frame.iloc[idx_choice, :]

        self.df = frame

        self.pose_batch = pose_batch
        self.bs = bs
        if pose_batch:
            self.pose_index = {}
            for i in range(12):
                self.pose_index[i] = []
            for i in range(len(self.df)):
                pose_cls = int(self.df.iloc[i]['azimuth'] // 30)
                self.pose_index[pose_cls].append(i)

        if train:
            self.im_transform = transforms.Compose([
                transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
                disturb
            ])
        else:
            self.im_transform = transforms.Compose([transforms.ToTensor(), normalize])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.pose_batch:
            batch_index = int(idx // self.bs)
            cls_index = batch_index % 12
            sample_index = (self.bs * idx // (12 * self.bs) + idx % self.bs) % len(self.pose_index[cls_index])
            idx = self.pose_index[cls_index][sample_index]

        img_name = os.path.join(self.root_dir, self.df.iloc[idx]['im_path'])
        cls_name = self.df.iloc[idx]['cls_name']
        cls_index = np.array([self.cls_names.index(cls_name)])
        cls_index = torch.from_numpy(cls_index).long()

        left = self.df.iloc[idx]['left']
        upper = self.df.iloc[idx]['upper']
        right = self.df.iloc[idx]['right']
        lower = self.df.iloc[idx]['lower']

        # load gt viewpoint label
        label = self.df.iloc[idx, 9:12].values

        # load images in RGB format
        im = Image.open(img_name).convert('RGB')
        im_pos = im.copy()

        if self.train:
            # gaussian blur
            if min(right - left, lower - upper) > 224 and np.random.random() > 0.5:
                blur_size = np.random.randint(low=1, high=5)
                im = im.filter(ImageFilter.GaussianBlur(blur_size))

            # crop the original image with 2D box jittering
            im = random_crop(im, left, upper, right - left, lower - upper)
            im_pos = random_crop(im_pos, left, upper, right - left, lower - upper)
            im_pos = resize_pad(im_pos, self.input_dim)
            im_pos = self.im_transform(im_pos)

            # get rotated image for regularization
            r = random.choice([-self.rot, self.rot])
            im_rot = im.rotate(r)
            im_rot = resize_pad(im_rot, self.input_dim)
            im_rot = self.im_transform(im_rot)
            label_rot = label.copy()
            label_rot[2] = label_rot[2] + r
            label_rot[2] += 360 if label_rot[2] < -180 else (-360 if label_rot[2] > 180 else 0)
            label_rot = process_viewpoint_label(label_rot, self.offset)

            # get flipped image for regularization
            im_flip = im.transpose(Image.FLIP_LEFT_RIGHT)
            im_flip = resize_pad(im_flip, self.input_dim)
            im_flip = self.im_transform(im_flip)
            label_flip = label.copy()
            label_flip[0] = 360 - label_flip[0]
            label_flip[2] = -label_flip[2]
            label_flip = process_viewpoint_label(label_flip, self.offset)

            # image to tensor
            im = resize_pad(im, self.input_dim)
            im = self.im_transform(im)
        else:
            # crop the original image with GT box
            im = im.crop((left, upper, right, lower))
            im_flip = im.transpose(Image.FLIP_LEFT_RIGHT)
            im = resize_pad(im, self.input_dim)
            im_flip = resize_pad(im_flip, self.input_dim)
            im = self.im_transform(im)
            im_flip = self.im_transform(im_flip)

        # chaneg angle labels to values >= 0
        label = process_viewpoint_label(label, self.offset)

        if self.train:
            return cls_index, im, label, im_flip, label_flip, im_rot, label_rot, im_pos
        else:
            return cls_index, im, label, im_flip



if __name__ == '__main__':
    # cls_choice = []
    cls_choice = ['car']

    mode = 2
    if mode == 0:
        bs = 8
        root_dir = '/media/gkouros/Elements SE/Datasets/pascal3d/PASCAL3D+_release1.1'
        annotation_file = 'Pascal3D.txt'
        trainset = Pascal3D(train=False, root_dir=root_dir, annotation_file=annotation_file, pose_batch=True, bs=bs, cls_choice=cls_choice)
    elif mode == 1:
        root_dir = '/media/gkouros/Elements SE/Datasets/pascal3d'
        trainset = Pascal3D_NeMo(root_dir=root_dir, cls_choice=cls_choice)
    elif mode ==2:
        root_dir = '/media/gkouros/Elements SE/Datasets/KITTI3D'
        trainset = KITTI3D_NeMo(root_dir=root_dir, occlusion_level=0)

    idx = np.random.randint(0, len(trainset))
    cls_index, im, label, im_flip = trainset[idx]
    print(label)
    cv2.imshow('', im.detach().permute(1,2,0).cpu().numpy())
    cv2.waitKey(0)

    # dataloader = data.DataLoader(trainset, batch_size=bs, shuffle=False, num_workers=4, drop_last=True)
    # for i, data in enumerate(dataloader):
    #     cls_index, im, label, im_flip, label_flip, im_rot, label_rot, im_pos = data