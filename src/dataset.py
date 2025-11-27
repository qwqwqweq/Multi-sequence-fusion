import os
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import torch
from scipy.ndimage import rotate, zoom, gaussian_filter, map_coordinates
import random
from sklearn.model_selection import StratifiedKFold
import logging
from collections import Counter

class NiiDataAugmentation:
    def __init__(self, p=0.5):
        self.p = p
        self.rotation_range = (-10, 10)
        self.noise_std = 0.01
        self.brightness_range = (0.8, 1.2)
        self.contrast_range = (0.8, 1.2)
        self.random_crop_size = (120, 120, 120)

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        
        try:
            if random.random() < self.p:
                d, h, w = image.shape
                if d >= self.random_crop_size[0] and h >= self.random_crop_size[1] and w >= self.random_crop_size[2]:
                    d_start = random.randint(0, d - self.random_crop_size[0])
                    h_start = random.randint(0, h - self.random_crop_size[1])
                    w_start = random.randint(0, w - self.random_crop_size[2])
                    
                    image = image[d_start:d_start+self.random_crop_size[0],
                                h_start:h_start+self.random_crop_size[1],
                                w_start:w_start+self.random_crop_size[2]]
                    
                    image = zoom(image, (d/self.random_crop_size[0],
                                    h/self.random_crop_size[1],
                                    w/self.random_crop_size[2]))
            
            if random.random() < self.p:
                angle = random.uniform(*self.rotation_range)
                axes = tuple(random.sample([0, 1, 2], 2))
                image = rotate(image, angle, axes=axes, reshape=False)
            
            if random.random() < self.p:
                axis = random.randint(0, 2)
                image = np.flip(image, axis=axis).copy()
            
            if random.random() < self.p:
                noise = np.random.normal(0, self.noise_std, image.shape)
                image = image + noise

            if random.random() < self.p:
                brightness_factor = random.uniform(*self.brightness_range)
                image = image * brightness_factor
            
            if random.random() < self.p:
                contrast_factor = random.uniform(*self.contrast_range)
                mean = np.mean(image)
                image = (image - mean) * contrast_factor + mean
            
            if random.random() < self.p:
                shape = image.shape
                alpha = random.uniform(0, 3)
                sigma = random.uniform(3, 5)
                dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
                dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
                dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
                
                z, y, x = np.meshgrid(np.arange(shape[0]), 
                                    np.arange(shape[1]), 
                                    np.arange(shape[2]),
                                    indexing='ij')
                
                indices = np.reshape(z + dz, (-1, 1)), \
                         np.reshape(y + dy, (-1, 1)), \
                         np.reshape(x + dx, (-1, 1))
                
                image = map_coordinates(image, indices, order=1).reshape(shape)
            
            return image
            
        except Exception as e:
            logging.error(f"Error in data augmentation: {str(e)}")
            return image

    def __repr__(self):
        return f"NiiDataAugmentation(p={self.p})"

class BrainMRIDataset(Dataset):
    def __init__(self, base_path, sequence_type, transform=None, mode='train', target_size=(128, 128, 128)):
        self.transform = transform
        self.sequence_type = sequence_type
        self.target_size = target_size
        self.mode = mode
        self.timestamp = "2025-06"
        self.user = "zhangzs"
        
        self.logger = logging.getLogger(f'{sequence_type}Dataset')
        
        normal_path = os.path.join(base_path, 'nii_output_in_new', sequence_type)
        osteopenia_path = os.path.join(base_path, 'nii_output_im_new', sequence_type)
        osteoporosis_path = os.path.join(base_path, 'nii_output_more_new', sequence_type)
        
        normal_samples = []   
        osteopenia_samples = [] 
        osteoporosis_samples = [] 
        
        if os.path.exists(normal_path):
            normal_samples = self._collect_and_validate_samples(normal_path, 0)
            
        if os.path.exists(osteopenia_path):
            osteopenia_samples = self._collect_and_validate_samples(osteopenia_path, 1)
            
        if os.path.exists(osteoporosis_path):
            osteoporosis_samples = self._collect_and_validate_samples(osteoporosis_path, 2)
        
        self.logger.info(f"Total samples found - Normal: {len(normal_samples)}, "
                        f"Osteopenia: {len(osteopenia_samples)}, "
                        f"Osteoporosis: {len(osteoporosis_samples)}")
        
        if len(normal_samples) == 0 or len(osteopenia_samples) == 0 or len(osteoporosis_samples) == 0:
            raise ValueError(f"Insufficient samples found for {sequence_type}")
        
        np.random.seed(42)
        random.seed(42)
        
        random.shuffle(normal_samples)
        random.shuffle(osteopenia_samples)
        random.shuffle(osteoporosis_samples)
        
        def split_samples(samples):
            n = len(samples)
            train_idx = int(0.7 * n)
            val_idx = int(0.8 * n)
            return samples[:train_idx], samples[train_idx:val_idx], samples[val_idx:]
        
        normal_train, normal_val, normal_test = split_samples(normal_samples)
        osteopenia_train, osteopenia_val, osteopenia_test = split_samples(osteopenia_samples)
        osteoporosis_train, osteoporosis_val, osteoporosis_test = split_samples(osteoporosis_samples)
        
        if mode == 'train':
            self.samples = normal_train + osteopenia_train + osteoporosis_train
        elif mode == 'val':
            self.samples = normal_val + osteopenia_val + osteoporosis_val
        else:  # test
            self.samples = normal_test + osteopenia_test + osteoporosis_test
        
        random.shuffle(self.samples)
        
        labels = [label for _, label in self.samples]
        class_dist = Counter(labels)
        self.logger.info(f"{mode} set size: {len(self.samples)}")
        self.logger.info(f"{mode} set class distribution: "
                        f"Normal: {class_dist[0]}, "
                        f"Osteopenia: {class_dist[1]}, "
                        f"Osteoporosis: {class_dist[2]}")
        
        if len(self.samples) == 0:
            raise ValueError(f"No samples available for {mode} set")

    def _validate_nifti_file(self, file_path):
        try:
            nii_img = nib.load(file_path)
            image = nii_img.get_fdata()
            
            if len(image.shape) > 3:
                self.logger.warning(f"Found {len(image.shape)}D image in {file_path}, attempting to convert to 3D")
                image = image[..., 0]
            
            if len(image.shape) != 3:
                raise ValueError(f"Cannot convert to 3D image, got shape {image.shape}")
            
            if not np.issubdtype(image.dtype, np.floating) and not np.issubdtype(image.dtype, np.integer):
                raise ValueError(f"Unexpected data type: {image.dtype}")
            
            if np.isnan(image).any() or np.isinf(image).any():
                raise ValueError("Image contains NaN or Inf values")
            
            if np.all(image == 0):
                raise ValueError("Image contains all zeros")
            
            if any(s == 0 for s in image.shape):
                raise ValueError(f"Invalid dimensions: {image.shape}")
            
            return True, None
        
        except Exception as e:
            return False, str(e)

    def _collect_and_validate_samples(self, path, label):
        valid_samples = []
        
        if not os.path.exists(path):
            self.logger.warning(f"Path does not exist: {path}")
            return valid_samples
        
        for patient in sorted(os.listdir(path)):
            patient_path = os.path.join(path, patient)
            if os.path.isdir(patient_path):
                nii_files = sorted([f for f in os.listdir(patient_path) if f.endswith('.nii.gz')])
                if nii_files:
                    file_path = os.path.join(patient_path, nii_files[0])
                    try:
                        is_valid, _ = self._validate_nifti_file(file_path)
                        if is_valid:
                            valid_samples.append((file_path, label))
                    except Exception as e:
                        self.logger.error(f"Error processing {file_path}: {str(e)}")
                        continue
        
        return valid_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        try:
            nii_img = nib.load(file_path)
            image = nii_img.get_fdata()
            
            if len(image.shape) > 3:
                image = image[..., 0]
            
            image = np.ascontiguousarray(image)
            
            resize_factor = np.array(self.target_size) / np.array(image.shape)
            image = zoom(image, resize_factor, order=1)
            
            mean = np.mean(image)
            std = np.std(image)
            if std == 0:
                std = 1e-6
            image = (image - mean) / std
            
            if self.transform:
                image = self.transform(image)
            
            image = torch.FloatTensor(image).unsqueeze(0)
            
            return image, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {str(e)}")
            return torch.zeros((1,) + self.target_size), torch.tensor(label, dtype=torch.long)
