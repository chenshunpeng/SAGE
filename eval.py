# -*- coding: UTF-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import parser
import logging
from os.path import join
from datetime import datetime
import test
import util
import commons
import datasets_ws
import network
import warnings
import pickle
from collections import OrderedDict
from config import apply_config

warnings.filterwarnings("ignore")

######################################### SETUP #########################################
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("test", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)

args.features_dim = 8448

logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

######################################### MODEL #########################################
model = network.SAGE(args)

ckpt = torch.load(args.ckpt_path, map_location=args.device)
state_dict = ckpt['model_state_dict']
if list(state_dict.keys())[0].startswith('module'):
    state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in state_dict.items()})
model.load_state_dict(state_dict, strict=False)
model = model.to(args.device)
model = torch.nn.DataParallel(model)

def load_or_compute_pca(args, model, pca_dataset_folder, full_features_dim):
    if args.pca_dim is not None and os.path.exists(args.pca_path):
        with open(args.pca_path, "rb") as f:
            pca = pickle.load(f)
        logging.info(f"Loaded PCA model from {args.pca_path}")
        return pca
    return util.compute_pca(args, model, pca_dataset_folder, full_features_dim)

# PCA Setup
pca = None
if args.pca_dim is not None:
    full_features_dim = args.features_dim
    args.features_dim = args.pca_dim
    os.makedirs(args.pca_dir, exist_ok=True)
    args.pca_path = os.path.join(args.pca_dir, f"pca_{args.pca_dim}_msls.pkl")
    pca = load_or_compute_pca(args, model, args.pca_dataset_folder, full_features_dim)

######################################### EVALUATION #########################################
logging.info(f"Starting evaluation on {len(args.eval_dataset_names)} datasets: {', '.join(args.eval_dataset_names)}")

for dataset_name in args.eval_dataset_names:
    args = apply_config(args, dataset_name)
    test_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, dataset_name, "test")
    logging.info(f"Test set: {test_ds}")

    recalls, recalls_str = test.test(args, test_ds, model, args.test_method, pca)
    logging.info(f"Recalls on {dataset_name}: {recalls_str}")

logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")