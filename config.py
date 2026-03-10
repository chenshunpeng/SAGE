# -*- coding: UTF-8 -*-

DEFAULT_BATCH_SIZE = 8
DEFAULT_RESIZE = [322, 322]

DATASET_SPECIFIC_CONFIG = {
    "nordland": {"batch_size": 26},
    "eynsham": {"batch_size": 20},
    "msls_740": {"batch_size": 16},
    "tokyo247": {"batch_size": 16},
    "sped": {"batch_size": 2},
    "amstertime": {"batch_size": 2},
}

def apply_config(args, dataset_name):
    name = dataset_name.lower()
    args.resize = DEFAULT_RESIZE
    
    if not getattr(args, "crossimage_encoder", False):
        args.infer_batch_size = 64
    else:
        config = DATASET_SPECIFIC_CONFIG.get(name, {"batch_size": DEFAULT_BATCH_SIZE})
        args.infer_batch_size = config["batch_size"]
    
    return args