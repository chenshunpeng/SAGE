# -*- coding: UTF-8 -*-

DEFAULT_BATCH_SIZE = 8
DEFAULT_RESIZE = [322, 322]

DATASET_BATCH_SIZE_GROUPS = {
    2: ("sped", "amstertime"),
    16: ("msls_740", "tokyo247"),
    20: ("eynsham",),
    26: ("nordland",),
    256: (
        "sf_xl", "sf_xl_v1", "sf_xl_v2", "sf_xl_night", "sf_xl_occlusion",
        "svox", "svox_base", "svox_queries", "svox_night",
        "svox_overcast", "svox_rain", "svox_snow", "svox_sun",
    ),
}

DATASET_SPECIFIC_CONFIG = {
    name: {"batch_size": batch_size}
    for batch_size, dataset_names in DATASET_BATCH_SIZE_GROUPS.items()
    for name in dataset_names
}

def apply_config(args, dataset_name):
    name = dataset_name.lower().replace("-", "_")
    args.resize = DEFAULT_RESIZE

    if getattr(args, "infer_batch_size_was_set", False):
        return args
    
    if not getattr(args, "crossimage_encoder", False):
        args.infer_batch_size = 64
    else:
        config = DATASET_SPECIFIC_CONFIG.get(name, {"batch_size": DEFAULT_BATCH_SIZE})
        args.infer_batch_size = config["batch_size"]
    
    return args
