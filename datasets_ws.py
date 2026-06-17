# -*- coding: UTF-8 -*-
import os
import torch
import faiss
import logging
import numpy as np
from collections import OrderedDict
from glob import glob
from tqdm import tqdm
from PIL import Image
from os.path import join
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
from sklearn.neighbors import NearestNeighbors
from torch.utils.data.dataloader import DataLoader

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def normalize_dataset_key(dataset_name):
    return dataset_name.lower().replace("-", "_")


GROUPED_DATASETS = {
    "sf_xl": {
        "dataset_name": "SF_XL",
        "path_parts": ("SF_XL",),
        "database_folder": "database",
        "query_folders": OrderedDict([
            ("SF_XL_v1", "queries_v1"),
            ("SF_XL_v2", "queries_v2"),
            ("SF_XL_night", "queries_night"),
            ("SF_XL_occlusion", "queries_occlusion"),
        ]),
    },
    "svox": {
        "dataset_name": "SVOX",
        "path_parts": ("svox", "images"),
        "database_folder": "gallery",
        "query_folders": OrderedDict([
            ("SVOX", "queries"),
            ("SVOX-night", "queries_night"),
            ("SVOX-overcast", "queries_overcast"),
            ("SVOX-rain", "queries_rain"),
            ("SVOX-snow", "queries_snow"),
            ("SVOX-sun", "queries_sun"),
        ]),
        "aliases": {
            "SVOX-base": ("SVOX", "queries"),
            "SVOX_queries": ("SVOX", "queries"),
        },
    },
}

GROUPED_QUERY_ALIASES = {}
for group_key, group_config in GROUPED_DATASETS.items():
    for query_set_name, queries_folder in group_config["query_folders"].items():
        alias_key = normalize_dataset_key(query_set_name)
        if alias_key != group_key:
            GROUPED_QUERY_ALIASES[alias_key] = (group_key, query_set_name, queries_folder)
    for alias_name, query_set_info in group_config.get("aliases", {}).items():
        GROUPED_QUERY_ALIASES[normalize_dataset_key(alias_name)] = (group_key, *query_set_info)

def path_to_pil_img(path):
    return Image.open(path).convert("RGB")

def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (images, 
        triplets_local_indexes, triplets_global_indexes).
        triplets_local_indexes are the indexes referring to each triplet within images.
        triplets_global_indexes are the global indexes of each image.
    Args:
        batch: list of tuple (images, triplets_local_indexes, triplets_global_indexes).
            considering each query to have 10 negatives (negs_num_per_query=10):
            - images: torch tensor of shape (12, 3, h, w).
            - triplets_local_indexes: torch tensor of shape (10, 3).
            - triplets_global_indexes: torch tensor of shape (12).
    Returns:
        images: torch tensor of shape (batch_size*12, 3, h, w).
        triplets_local_indexes: torch tensor of shape (batch_size*10, 3).
        triplets_global_indexes: torch tensor of shape (batch_size, 12).
    """
    images                  = torch.cat([e[0] for e in batch])
    triplets_local_indexes  = torch.cat([e[1][None] for e in batch])
    triplets_global_indexes = torch.cat([e[2][None] for e in batch])
    for i, (local_indexes, global_indexes) in enumerate(zip(triplets_local_indexes, triplets_global_indexes)):
        local_indexes += len(global_indexes) * i  # Increment local indexes by offset (len(global_indexes) is 12)
    return images, torch.cat(tuple(triplets_local_indexes)), triplets_global_indexes


class PCADataset(data.Dataset):
    def __init__(self, args, datasets_folder="dataset", dataset_folder="pitts30k/images/train"):
        dataset_folder_full_path = join(datasets_folder, dataset_folder)
        if not os.path.exists(dataset_folder_full_path) :
            raise FileNotFoundError(f"Folder {dataset_folder_full_path} does not exist")
        self.images_paths = sorted(glob(join(dataset_folder_full_path, "**", "*.jpg"), recursive=True))
        self.resize = args.resize
    def __getitem__(self, index):
        img = base_transform(path_to_pil_img(self.images_paths[index]))
        img = transforms.functional.resize(img, self.resize)
        return img
    def __len__(self):
        return len(self.images_paths)

class BaseDataset(data.Dataset):
    """Dataset with images from database and queries, used for inference (testing and building cache).
    """
    def __init__(self, args, datasets_folder="datasets", dataset_name="pitts30k", split="train"):
        super().__init__()
        self.args = args
        self.dataset_name = dataset_name
        normalized_dataset_name = normalize_dataset_key(dataset_name)
        group_config = GROUPED_DATASETS.get(normalized_dataset_name)
        query_set_config = GROUPED_QUERY_ALIASES.get(normalized_dataset_name)

        if group_config is not None or query_set_config is not None:
            if split != "test":
                raise ValueError(f"{dataset_name} query sets are only available for the test split")
            if query_set_config is not None:
                group_key, query_set_name, queries_folder_name = query_set_config
                group_config = GROUPED_DATASETS[group_key]
                self.dataset_name = query_set_name
            else:
                self.dataset_name = group_config["dataset_name"]
                queries_folder_name = None
            self.dataset_folder = join(datasets_folder, *group_config["path_parts"], split)
            database_folder_name = group_config["database_folder"]
        else:
            self.dataset_folder = join(datasets_folder, dataset_name, "images", split)
            database_folder_name = "database"
        if not os.path.exists(self.dataset_folder): raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")
        
        self.resize = args.resize
        self.test_method = args.test_method
        
        #### Read paths and UTM coordinates for all images.
        database_folder = join(self.dataset_folder, database_folder_name)
        if not os.path.exists(database_folder): raise FileNotFoundError(f"Folder {database_folder} does not exist")
        self.database_paths = sorted(glob(join(database_folder, "**", "*.jpg"), recursive=True))

        self.query_group_slices = None
        if group_config is not None and query_set_config is None:
            self.queries_paths = []
            self.query_group_slices = OrderedDict()
            for query_set_name, queries_folder_name in group_config["query_folders"].items():
                queries_folder = join(self.dataset_folder, queries_folder_name)
                if not os.path.exists(queries_folder):
                    raise FileNotFoundError(f"Folder {queries_folder} does not exist")
                query_paths = sorted(glob(join(queries_folder, "**", "*.jpg"), recursive=True))
                start_index = len(self.queries_paths)
                self.queries_paths.extend(query_paths)
                self.query_group_slices[query_set_name] = (start_index, len(self.queries_paths))
        else:
            if query_set_config is not None:
                queries_folder = join(self.dataset_folder, queries_folder_name)
            else:
                queries_folder = join(self.dataset_folder, "queries")
            if not os.path.exists(queries_folder):
                raise FileNotFoundError(f"Folder {queries_folder} does not exist")
            self.queries_paths = sorted(glob(join(queries_folder, "**", "*.jpg"), recursive=True))
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]).astype(np.float64)
        self.queries_utms  = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]).astype(np.float64)
        
        # Find soft_positives_per_query, which are within val_positive_dist_threshold (deafult 25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.soft_positives_per_query = knn.radius_neighbors(self.queries_utms, 
                                                             radius=args.val_positive_dist_threshold,
                                                             return_distance=False)
        
        self.images_paths = list(self.database_paths) + list(self.queries_paths)
        
        self.database_num = len(self.database_paths)
        self.queries_num  = len(self.queries_paths)
    
    def __getitem__(self, index):
        img = path_to_pil_img(self.images_paths[index])
        img = base_transform(img)
        # With database images self.test_method should always be "hard_resize"
        if self.test_method == "hard_resize":
            # self.test_method=="hard_resize" is the default, resizes all images to the same size.
            img = transforms.functional.resize(img, self.resize)
        else:
            img = self._test_query_transform(img)
        return img, index
    
    def _test_query_transform(self, img):
        """Transform query image according to self.test_method."""
        C, H, W = img.shape
        if self.test_method == "single_query":
            # self.test_method=="single_query" is used when queries have varying sizes, and can't be stacked in a batch.
            processed_img = transforms.functional.resize(img, self.resize) #min(self.resize)
        elif self.test_method == "central_crop":
            # Take the biggest central crop of size self.resize. Preserves ratio.
            scale = max(self.resize[0]/H, self.resize[1]/W)
            processed_img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale).squeeze(0)
            processed_img = transforms.functional.center_crop(processed_img, self.resize)
            assert processed_img.shape[1:] == torch.Size(self.resize), f"{processed_img.shape[1:]} {self.resize}"
        elif self.test_method == "five_crops" or self.test_method == 'nearest_crop' or self.test_method == 'maj_voting':
            # Get 5 square crops with size==shorter_side (usually 480). Preserves ratio and allows batches.
            shorter_side = min(self.resize)
            processed_img = transforms.functional.resize(img, shorter_side)
            processed_img = torch.stack(transforms.functional.five_crop(processed_img, shorter_side))
            assert processed_img.shape == torch.Size([5, 3, shorter_side, shorter_side]), \
                f"{processed_img.shape} {torch.Size([5, 3, shorter_side, shorter_side])}"
        return processed_img
    
    def __len__(self):
        return len(self.images_paths)
    def __repr__(self):
        if self.query_group_slices:
            group_counts = ", ".join(
                f"{name}: {end - start}" for name, (start, end) in self.query_group_slices.items()
            )
            return (f"< {self.__class__.__name__}, {self.dataset_name} - #database: {self.database_num}; "
                    f"#queries: {self.queries_num} ({group_counts}) >")
        return  (f"< {self.__class__.__name__}, {self.dataset_name} - #database: {self.database_num}; #queries: {self.queries_num} >")
    def get_positives(self):
        return self.soft_positives_per_query
