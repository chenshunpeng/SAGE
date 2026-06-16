# -*- coding: UTF-8 -*-
import faiss
import torch
import logging
import math
import os
import resource
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from prettytable import PrettyTable


MIN_DATABASE_CHUNK_SIZE = 4096
MAX_DATABASE_CHUNK_SIZE = 32768
TARGET_DATABASE_CHUNKS = 100
MEMORY_SAFETY_FACTOR = 1.25


def compute_recalls(args, eval_ds, predictions):
    """Compute recalls for the full query set or for each configured query group."""
    positives_per_query = eval_ds.get_positives()

    def compute_for_slice(start_index, end_index):
        recalls = np.zeros(len(args.recall_values))
        sliced_predictions = predictions[start_index:end_index]
        sliced_positives = positives_per_query[start_index:end_index]
        for pred, positives in zip(sliced_predictions, sliced_positives):
            for i, n in enumerate(args.recall_values):
                if np.any(np.in1d(pred[:n], positives)):
                    recalls[i:] += 1
                    break
        recalls = recalls / len(sliced_predictions) * 100
        recalls_str = ", ".join(
            [f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)]
        )
        return recalls, recalls_str

    query_group_slices = getattr(eval_ds, "query_group_slices", None)
    if not query_group_slices:
        return compute_for_slice(0, eval_ds.queries_num)

    grouped_recalls = OrderedDict()
    grouped_recalls_str = OrderedDict()
    for query_set_name, (start_index, end_index) in query_group_slices.items():
        recalls, recalls_str = compute_for_slice(start_index, end_index)
        grouped_recalls[query_set_name] = recalls
        grouped_recalls_str[query_set_name] = recalls_str
    return grouped_recalls, grouped_recalls_str


def print_recalls_table(args, eval_ds, recalls):
    recall_groups = recalls.items() if isinstance(recalls, dict) else [(eval_ds, recalls)]
    for dataset_label, query_set_recalls in recall_groups:
        table = PrettyTable()
        table.field_names = ['K'] + [str(k) for k in args.recall_values]
        table.add_row(['Recall@K'] + [f'{v:.2f}' for v in query_set_recalls])
        print(table.get_string(title=f"Performances on {dataset_label}"))


def _format_bytes(num_bytes):
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def _get_current_rss_bytes():
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as statm_file:
            resident_pages = int(statm_file.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _get_available_memory_bytes():
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as meminfo_file:
            for line in meminfo_file:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def _get_database_chunk_size(database_num, infer_batch_size):
    if database_num <= MIN_DATABASE_CHUNK_SIZE:
        return database_num
    target_size = math.ceil(database_num / TARGET_DATABASE_CHUNKS)
    target_size = max(MIN_DATABASE_CHUNK_SIZE, min(MAX_DATABASE_CHUNK_SIZE, target_size))
    aligned_size = math.ceil(target_size / infer_batch_size) * infer_batch_size
    return min(aligned_size, database_num)


def _report_efficient_ram_requirements(args, eval_ds, test_method, chunk_size, search_k):
    query_multiplier = 5 if test_method in ["nearest_crop", "maj_voting"] else 1
    query_vectors_num = eval_ds.queries_num * query_multiplier
    float_size = np.dtype(np.float32).itemsize
    index_size = np.dtype(np.int64).itemsize
    query_features_bytes = query_vectors_num * args.features_dim * float_size
    database_chunk_bytes = chunk_size * args.features_dim * float_size
    topk_bytes = query_vectors_num * search_k * (float_size + index_size)
    # Includes global/local results plus NumPy merge candidates, ordering, and outputs.
    minimum_extra_bytes = query_features_bytes + 2 * database_chunk_bytes + 7 * topk_bytes
    recommended_bytes = math.ceil(minimum_extra_bytes * MEMORY_SAFETY_FACTOR)
    current_rss_bytes = _get_current_rss_bytes()
    available_bytes = _get_available_memory_bytes()
    chunks_num = math.ceil(eval_ds.database_num / chunk_size)

    logging.info("Exact low-memory retrieval enabled")
    logging.info(
        f"Retrieval size: database={eval_ds.database_num}, queries={eval_ds.queries_num}, "
        f"feature_dim={args.features_dim}, top_k={search_k}"
    )
    logging.info(f"Database chunk: {chunk_size} images, approximately {chunks_num} chunks")
    logging.info(
        f"Memory: process RSS={_format_bytes(current_rss_bytes)}, "
        f"system available={_format_bytes(available_bytes)}"
    )
    logging.info(
        f"Estimated retrieval memory: minimum extra={_format_bytes(minimum_extra_bytes)}, "
        f"recommended available={_format_bytes(recommended_bytes)}"
    )
    logging.info("The estimate excludes model, CUDA, DataLoader, and other future allocations")

    if available_bytes < minimum_extra_bytes:
        raise MemoryError(
            f"Available memory {_format_bytes(available_bytes)} is below the estimated minimum "
            f"{_format_bytes(minimum_extra_bytes)} for exact low-memory retrieval"
        )
    if available_bytes < recommended_bytes:
        logging.warning(
            f"Available memory {_format_bytes(available_bytes)} is below the recommended "
            f"{_format_bytes(recommended_bytes)} safety threshold"
        )


def _extract_query_features(args, eval_ds, model, test_method, pca):
    query_multiplier = 5 if test_method in ["nearest_crop", "maj_voting"] else 1
    queries_features = np.empty(
        (eval_ds.queries_num * query_multiplier, args.features_dim), dtype=np.float32
    )
    queries_infer_batch_size = 1 if test_method == "single_query" else args.infer_batch_size
    eval_ds.test_method = test_method
    queries_subset_ds = Subset(
        eval_ds, list(range(eval_ds.database_num, eval_ds.database_num + eval_ds.queries_num))
    )
    queries_dataloader = DataLoader(
        dataset=queries_subset_ds,
        num_workers=args.num_workers,
        batch_size=queries_infer_batch_size,
        pin_memory=(args.device == "cuda"),
    )

    logging.debug("Extracting queries features for evaluation/testing")
    for inputs, indices in tqdm(queries_dataloader, ncols=100, desc="Queries"):
        if test_method in ["five_crops", "nearest_crop", "maj_voting"]:
            inputs = torch.cat(tuple(inputs))
        features = model(inputs.to(args.device))
        if test_method == "five_crops":
            features = torch.stack(torch.split(features, 5)).mean(1)
        features = features.cpu().numpy()
        if pca is not None:
            features = pca.transform(features)
        features = np.asarray(features, dtype=np.float32)

        if test_method in ["nearest_crop", "maj_voting"]:
            start_idx = int(indices[0] - eval_ds.database_num) * 5
            end_idx = start_idx + len(indices) * 5
            queries_features[start_idx:end_idx] = features
        else:
            query_indices = indices.numpy() - eval_ds.database_num
            queries_features[query_indices] = features

    return queries_features


def _merge_exact_topk(best_distances, best_predictions, chunk_distances,
                      chunk_predictions, search_k):
    candidate_distances = np.concatenate((best_distances, chunk_distances), axis=1)
    candidate_predictions = np.concatenate((best_predictions, chunk_predictions), axis=1)
    # Distance is primary; the global database index makes exact ties deterministic.
    order = np.lexsort((candidate_predictions, candidate_distances), axis=1)[:, :search_k]
    best_distances = np.take_along_axis(candidate_distances, order, axis=1)
    best_predictions = np.take_along_axis(candidate_predictions, order, axis=1)
    return best_distances, best_predictions


def _refine_crop_predictions(args, eval_ds, test_method, distances, predictions):
    if test_method not in ["nearest_crop", "maj_voting"]:
        return predictions

    search_k = predictions.shape[1]
    distances = distances.reshape(eval_ds.queries_num, 5, search_k)
    predictions = predictions.reshape(eval_ds.queries_num, 5, search_k)
    refined_predictions = np.empty((eval_ds.queries_num, search_k), dtype=predictions.dtype)

    for query_index in range(eval_ds.queries_num):
        if test_method == "maj_voting":
            top_n_voting('top1', predictions[query_index], distances[query_index], args.majority_weight)
            top_n_voting('top5', predictions[query_index], distances[query_index], args.majority_weight)
            top_n_voting('top10', predictions[query_index], distances[query_index], args.majority_weight)

        query_distances = distances[query_index].reshape(-1)
        query_predictions = predictions[query_index].reshape(-1)
        sort_idx = np.lexsort((query_predictions, query_distances))
        sorted_predictions = query_predictions[sort_idx]
        _, unique_idx = np.unique(sorted_predictions, return_index=True)
        refined_predictions[query_index] = sorted_predictions[np.sort(unique_idx)][:search_k]

    return refined_predictions


def test_efficient_ram_usage(args, eval_ds, model, test_method="hard_resize", pca=None):
    """Run exact float32 L2 retrieval while keeping one database chunk in RAM."""
    if eval_ds.database_num == 0 or eval_ds.queries_num == 0:
        raise ValueError("The evaluation dataset must contain database and query images")

    model = model.eval()
    search_k = min(max(args.recall_values), eval_ds.database_num)
    chunk_size = _get_database_chunk_size(eval_ds.database_num, args.infer_batch_size)
    _report_efficient_ram_requirements(args, eval_ds, test_method, chunk_size, search_k)

    with torch.no_grad():
        queries_features = _extract_query_features(args, eval_ds, model, test_method, pca)
        query_vectors_num = len(queries_features)
        invalid_index = np.iinfo(np.int64).max
        best_distances = np.full((query_vectors_num, search_k), np.inf, dtype=np.float32)
        best_predictions = np.full(
            (query_vectors_num, search_k), invalid_index, dtype=np.int64
        )

        eval_ds.test_method = "hard_resize"
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds,
            num_workers=args.num_workers,
            batch_size=args.infer_batch_size,
            pin_memory=(args.device == "cuda"),
        )
        database_chunk = np.empty((chunk_size, args.features_dim), dtype=np.float32)
        chunk_count = 0
        chunk_start_index = 0

        logging.debug("Extracting and searching database features in chunks")
        for inputs, indices in tqdm(database_dataloader, ncols=100, desc="Database"):
            features = model(inputs.to(args.device)).cpu().numpy()
            if pca is not None:
                features = pca.transform(features)
            features = np.asarray(features, dtype=np.float32)
            database_indices = indices.numpy()
            expected_indices = np.arange(
                chunk_start_index + chunk_count,
                chunk_start_index + chunk_count + len(database_indices),
            )
            if not np.array_equal(database_indices, expected_indices):
                raise RuntimeError("Database DataLoader returned non-contiguous indexes")

            end_index = chunk_count + len(features)
            database_chunk[chunk_count:end_index] = features
            chunk_count = end_index

            if chunk_count == chunk_size:
                chunk_index = faiss.IndexFlatL2(args.features_dim)
                chunk_index.add(database_chunk)
                chunk_k = min(search_k, chunk_count)
                chunk_distances, chunk_predictions = chunk_index.search(queries_features, chunk_k)
                chunk_predictions += chunk_start_index
                best_distances, best_predictions = _merge_exact_topk(
                    best_distances, best_predictions, chunk_distances,
                    chunk_predictions, search_k
                )
                chunk_start_index += chunk_count
                chunk_count = 0

        if chunk_count:
            final_chunk = np.ascontiguousarray(database_chunk[:chunk_count])
            chunk_index = faiss.IndexFlatL2(args.features_dim)
            chunk_index.add(final_chunk)
            chunk_k = min(search_k, chunk_count)
            chunk_distances, chunk_predictions = chunk_index.search(queries_features, chunk_k)
            chunk_predictions += chunk_start_index
            if chunk_k < search_k:
                pad_width = search_k - chunk_k
                chunk_distances = np.pad(
                    chunk_distances, ((0, 0), (0, pad_width)), constant_values=np.inf
                )
                chunk_predictions = np.pad(
                    chunk_predictions, ((0, 0), (0, pad_width)), constant_values=invalid_index
                )
            best_distances, best_predictions = _merge_exact_topk(
                best_distances, best_predictions, chunk_distances,
                chunk_predictions, search_k
            )

    predictions = _refine_crop_predictions(
        args, eval_ds, test_method, best_distances, best_predictions
    )
    recalls, recalls_str = compute_recalls(args, eval_ds, predictions)
    print()
    print_recalls_table(args, eval_ds, recalls)
    return recalls, recalls_str


def test(args, eval_ds, model, test_method="hard_resize", pca=None):
    """Compute features of the given dataset and compute the recalls."""
    
    assert test_method in ["hard_resize", "single_query", "central_crop", "five_crops",
                            "nearest_crop", "maj_voting"], f"test_method can't be {test_method}"
    
    if args.efficient_ram_testing:
        return test_efficient_ram_usage(args, eval_ds, model, test_method, pca)
    
    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database features for evaluation/testing")
        # For database use "hard_resize", although it usually has no effect because database images have same resolution
        eval_ds.test_method = "hard_resize"
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device=="cuda"))
        
        if test_method == "nearest_crop" or test_method == 'maj_voting':
            all_features = np.empty((5 * eval_ds.queries_num + eval_ds.database_num, args.features_dim), dtype="float32")
        else:
            all_features = np.empty((len(eval_ds), args.features_dim), dtype="float32")

        for inputs, indices in tqdm(database_dataloader, ncols=100):
            features = model(inputs.to(args.device))
            features = features.cpu().numpy()
            if pca != None:
                features = pca.transform(features)
            all_features[indices.numpy(), :] = features
        
        logging.debug("Extracting queries features for evaluation/testing")
        queries_infer_batch_size = 1 if test_method == "single_query" else args.infer_batch_size
        eval_ds.test_method = test_method
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))
        for inputs, indices in tqdm(queries_dataloader, ncols=100):
            if test_method == "five_crops" or test_method == "nearest_crop" or test_method == 'maj_voting':
                inputs = torch.cat(tuple(inputs))  # shape = 5*bs x 3 x 480 x 480
            features = model(inputs.to(args.device))
            if test_method == "five_crops":  # Compute mean along the 5 crops
                features = torch.stack(torch.split(features, 5)).mean(1)
            features = features.cpu().numpy()
            if pca != None:
                features = pca.transform(features)
            
            if test_method == "nearest_crop" or test_method == 'maj_voting':  # store the features of all 5 crops
                start_idx = eval_ds.database_num + (indices[0] - eval_ds.database_num) * 5
                end_idx   = start_idx + indices.shape[0] * 5
                indices = np.arange(start_idx, end_idx)
                all_features[indices, :] = features
            else:
                all_features[indices.numpy(), :] = features
    
    queries_features = all_features[eval_ds.database_num:]
    database_features = all_features[:eval_ds.database_num]
    
    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features)
    del database_features, all_features
    
    logging.debug("Calculating recalls")
    distances, predictions = faiss_index.search(queries_features, max(args.recall_values))

    if test_method == 'nearest_crop':
        distances = np.reshape(distances, (eval_ds.queries_num, 20 * 5))
        predictions = np.reshape(predictions, (eval_ds.queries_num, 20 * 5))
        for q in range(eval_ds.queries_num):
            # sort predictions by distance
            sort_idx = np.argsort(distances[q])
            predictions[q] = predictions[q, sort_idx]
            # remove duplicated predictions, i.e. keep only the closest ones
            _, unique_idx = np.unique(predictions[q], return_index=True)
            # unique_idx is sorted based on the unique values, sort it again
            predictions[q, :20] = predictions[q, np.sort(unique_idx)][:20]
        predictions = predictions[:, :20]  # keep only the closer 20 predictions for each query
    elif test_method == 'maj_voting':
        distances = np.reshape(distances, (eval_ds.queries_num, 5, 20))
        predictions = np.reshape(predictions, (eval_ds.queries_num, 5, 20))
        for q in range(eval_ds.queries_num):
            # votings, modify distances in-place
            top_n_voting('top1', predictions[q], distances[q], args.majority_weight)
            top_n_voting('top5', predictions[q], distances[q], args.majority_weight)
            top_n_voting('top10', predictions[q], distances[q], args.majority_weight)

            # flatten dist and preds from 5, 20 -> 20*5
            # and then proceed as usual to keep only first 20
            dists = distances[q].flatten()
            preds = predictions[q].flatten()

            # sort predictions by distance
            sort_idx = np.argsort(dists)
            preds = preds[sort_idx]
            # remove duplicated predictions, i.e. keep only the closest ones
            _, unique_idx = np.unique(preds, return_index=True)
            # unique_idx is sorted based on the unique values, sort it again
            # here the row corresponding to the first crop is used as a
            # 'buffer' for each query, and in the end the dimension
            # relative to crops is eliminated
            predictions[q, 0, :20] = preds[np.sort(unique_idx)][:20]
        predictions = predictions[:, 0, :20]  # keep only the closer 20 predictions for each query

    recalls, recalls_str = compute_recalls(args, eval_ds, predictions)

    print()  # print a new line
    print_recalls_table(args, eval_ds, recalls)

    return recalls, recalls_str


def top_n_voting(topn, predictions, distances, maj_weight):
    if topn == 'top1':
        n = 1
        selected = 0
    elif topn == 'top5':
        n = 5
        selected = slice(0, 5)
    elif topn == 'top10':
        n = 10
        selected = slice(0, 10)
    # find predictions that repeat in the first, first five,
    # or fist ten columns for each crop
    vals, counts = np.unique(predictions[:, selected], return_counts=True)
    # for each prediction that repeats more than once,
    # subtract from its score
    for val, count in zip(vals[counts > 1], counts[counts > 1]):
        mask = (predictions[:, selected] == val)
        distances[:, selected][mask] -= maj_weight * count/n
