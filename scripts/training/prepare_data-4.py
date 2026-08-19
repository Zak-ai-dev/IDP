import sys

sys.path.append("/mnt/code")

import hashlib
import json
import logging
import os
import pickle
from collections import Counter

import pandas as pd
from docintel_ml import Alto
from sklearn.model_selection import train_test_split

# The metadata location is owned by load_data, which writes it: importing it here
# guarantees both steps always agree on the same file.
from utils.load_data import get_metadata_path

# Minimum number of members a stratification group must hold for the split to be
# feasible. Below that, sklearn cannot place one member on each side.
MIN_GROUP = 2


def remove_idxs(removed_idxs, *argv):
    """
    Remove the given indexes from every provided list.

    Args:
        removed_idxs (list[int]): Indexes to discard.
        *argv: Lists to filter, all aligned on the same indexing.

    Returns:
        tuple: The filtered lists, in the same order.
    """
    kept_idxs = [i not in removed_idxs for i in range(len(argv[0]))]
    kept_arg = []
    for arg in argv:
        if arg is not None:
            kept_arg.append([elem for i, elem in enumerate(arg) if kept_idxs[i]])
        else:
            kept_arg.append(None)

    return tuple(kept_arg)


def load_metadata(config: dict, logger: logging.Logger) -> pd.DataFrame:
    """
    Load the metadata produced by load_data.

    The metadata is the single source of truth for the train/test membership: the alto
    files are stored on disk under their upload id, so the original filename carrying
    the train/test prefix is only available here.

    Args:
        config (dict): Configuration dictionary.
        logger (logging.Logger): Logger instance.

    Returns:
        pd.DataFrame: Metadata of every downloaded document.
    """
    metadata_path = get_metadata_path(config)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata not found at {metadata_path}. It is written by load_data, which must run "
            f"before preparing the dataset. Check that data_path points to the folder used for loading."
        )

    df_metadata = pd.read_pickle(metadata_path)

    required = {"id", "filename", "label", "set"}
    missing = required - set(df_metadata.columns)
    if missing:
        raise ValueError(
            f"Metadata at {metadata_path} is missing the required columns {sorted(missing)}. "
            f"It was probably written by an older version of load_data: rerun the loading step."
        )

    if df_metadata.empty:
        raise ValueError(f"Metadata at {metadata_path} is empty, nothing to prepare.")

    logger.info(f"Loaded metadata from {metadata_path} ({len(df_metadata)} docs)")
    return df_metadata


def read_altos(df_metadata: pd.DataFrame, config: dict, logger: logging.Logger) -> dict:
    """
    Read the alto files referenced by the metadata and collect the per-document
    information needed downstream.

    Documents listed in the metadata but missing on disk are skipped with a warning,
    so that a partial download does not break the whole preparation.

    Args:
        df_metadata (pd.DataFrame): Metadata of the downloaded documents.
        config (dict): Configuration dictionary.
        logger (logging.Logger): Logger instance.

    Returns:
        dict: Aligned lists of paths, ids, filenames, labels, sets, ocr flags and hashes.
    """
    altos_paths, altos_ids, filenames, labels, sets, ocr_flags, file_hashes = [], [], [], [], [], [], []
    n_missing = 0

    for label, df_label in df_metadata.groupby("label"):
        altos_folder = os.path.join(config["DATA"]["data_path"], label, "altos")

        for row in df_label.itertuples():
            alto_path = os.path.join(altos_folder, f"{row.id}.json")
            if not os.path.exists(alto_path):
                n_missing += 1
                continue

            with open(alto_path) as alto_json:
                loaded_alto = json.load(alto_json)

            altos_paths.append(alto_path)
            altos_ids.append(str(row.id))
            filenames.append(row.filename)
            labels.append(label)
            sets.append(row.set)
            ocr_flags.append(loaded_alto["OCRized"])

            hash = hashlib.sha256(
                json.dumps(
                    {
                        key: value
                        for key, value in loaded_alto.items()
                        if key in ["numPages", "pages", "fonts"]
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).digest()
            file_hashes.append(hash)

        logger.info(f"Loaded {label} data.")

    if n_missing:
        logger.warning(f"{n_missing} documents listed in the metadata are missing on disk, skipped.")

    return {
        "altos_paths": altos_paths,
        "altos_ids": altos_ids,
        "filenames": filenames,
        "labels": labels,
        "sets": sets,
        "ocr_flags": ocr_flags,
        "file_hashes": file_hashes,
    }


def find_duplicates(file_hashes: list, sets: list) -> list:
    """
    Identify duplicate documents from their content hash.

    When a document appears in both splits, the train copy is the one discarded: keeping
    the test copy protects the evaluation set from leaking into training.

    Args:
        file_hashes (list): Content hash of every document.
        sets (list): Train/test membership of every document.

    Returns:
        list[tuple[int, list[int]]]: One entry per duplicated content, holding the index
            kept and the indexes to discard.
    """
    idxs_by_h = {}
    for i, hash in enumerate(file_hashes):
        idxs_by_h.setdefault(hash, []).append(i)

    groups = []
    for idxs in idxs_by_h.values():
        if len(idxs) < 2:
            continue
        # Keep a test occurrence when there is one, otherwise keep the first occurrence.
        test_idxs = [i for i in idxs if sets[i] == "test"]
        kept = test_idxs[0] if test_idxs else idxs[0]
        groups.append((kept, [i for i in idxs if i != kept]))

    return groups


def log_duplicates(groups: list, filenames: list, altos_ids: list, labels: list, sets: list, logger) -> None:
    """
    Log every duplicated document by name and id, so that the discarded files can be
    traced back both to the workspace they came from and to the alto stored on disk.

    Args:
        groups (list): Output of find_duplicates.
        filenames (list): Original filename of every document.
        altos_ids (list): Upload id of every document, i.e. the alto file stem on disk.
        labels (list): Label of every document.
        sets (list): Train/test membership of every document.
        logger (logging.Logger): Logger instance.
    """
    for kept, removed in groups:
        removed_desc = ", ".join(
            f"{filenames[i]} (id={altos_ids[i]}) [{labels[i]}/{sets[i]}]" for i in removed
        )
        logger.info(
            f"Duplicate content: kept {filenames[kept]} (id={altos_ids[kept]}) "
            f"[{labels[kept]}/{sets[kept]}], removed {removed_desc}"
        )

        # A duplicate spanning two labels means the same document is annotated twice
        # with conflicting classes: worth checking before training.
        involved_labels = {labels[i] for i in [kept] + removed}
        if len(involved_labels) > 1:
            logger.warning(
                f"Duplicate {filenames[kept]} (id={altos_ids[kept]}) spans several labels "
                f"{sorted(involved_labels)}, the annotation should be reviewed."
            )


def get_stratify(labels: list, ocr_flags: list, logger: logging.Logger, context: str = ""):
    """
    Build the stratification key, degrading gracefully when it is not feasible.

    Stratifying on (label, ocr) fails as soon as one combination holds a single member,
    which happens whenever a document type is fully OCRized or not OCRized at all. The
    key is therefore validated first, then downgraded to the label alone, then dropped.

    Args:
        labels (list): Label of every document.
        ocr_flags (list): OCR flag of every document.
        logger (logging.Logger): Logger instance.
        context (str): Split being prepared, used for logging.

    Returns:
        list | None: The stratification key, or None when stratification is impossible.
    """
    joint = list(zip(labels, ocr_flags))
    counts = Counter(joint)
    small_groups = {group: n for group, n in counts.items() if n < MIN_GROUP}

    if not small_groups:
        return joint

    logger.warning(
        f"{context}: cannot stratify on (label, ocr), {len(small_groups)} group(s) below "
        f"{MIN_GROUP} members ({small_groups}). Falling back to label only."
    )

    label_counts = Counter(labels)
    small_labels = {label: n for label, n in label_counts.items() if n < MIN_GROUP}
    if small_labels:
        logger.warning(
            f"{context}: cannot stratify on label either ({small_labels}). "
            f"Splitting without stratification."
        )
        return None

    return labels


def prepare_data(config, logger):
    """
    Build the train/validation/test split from the documents downloaded by load_data.

    The test set is not drawn here: it is the one already frozen in the metadata, so
    that evaluation stays comparable across runs. Only the validation set is carved out
    of the train documents, using "validation_size" as a share of the train pool.

    Args:
        config (dict): Configuration dictionary.
        logger (logging.Logger): Logger instance.
    """
    try:
        df_metadata = load_metadata(config, logger)

        # Target classes, read from the metadata rather than the configuration, so that
        # a label declared but never downloaded does not create an empty class.
        classes = sorted(df_metadata["label"].unique())
        logger.info(f"Target classes: {classes}")

        collected = read_altos(df_metadata, config, logger)
        altos_paths = collected["altos_paths"]
        altos_ids = collected["altos_ids"]
        filenames = collected["filenames"]
        labels = collected["labels"]
        sets = collected["sets"]
        ocr_flags = collected["ocr_flags"]
        file_hashes = collected["file_hashes"]

        altos = [Alto(alto_path) for alto_path in altos_paths]

        # CLEAN DATA

        # Remove duplicate documents
        dup_groups = find_duplicates(file_hashes, sets)
        log_duplicates(dup_groups, filenames, altos_ids, labels, sets, logger)

        dup_idxs = sorted(i for _, removed in dup_groups for i in removed)
        dup_labels = [labels[i] for i in dup_idxs]

        altos, altos_paths, altos_ids, filenames, labels, sets, ocr_flags = remove_idxs(
            dup_idxs, altos, altos_paths, altos_ids, filenames, labels, sets, ocr_flags
        )
        logger.info(f"Identified {len(dup_idxs)} duplicates and removed : {Counter(dup_labels)}")

        # Count and display OCR and non OCR per label for sanity check
        ocr_counter = Counter(zip(labels, ocr_flags))
        doc_classes = set([doc_class for (doc_class, ocr) in ocr_counter.keys()])  # Get all labels

        for doc_class in doc_classes:
            ocr_count = ocr_counter[(doc_class, True)]
            no_ocr_count = ocr_counter[(doc_class, False)]
            total = ocr_count + no_ocr_count
            logger.info(
                f" For {doc_class} : remain {total} documents of which "
                f"{format(ocr_count / total, '0.00%')} are from OCR."
            )

        # DATASET SPLIT

        # The test set comes from the metadata: it is already frozen by load_data.
        test_idxs = [i for i, s in enumerate(sets) if s == "test"]
        train_idxs = [i for i, s in enumerate(sets) if s == "train"]

        altos_paths_test = [altos_paths[i] for i in test_idxs]
        y_test = [labels[i] for i in test_idxs]
        altos_ocr_flags_test = [ocr_flags[i] for i in test_idxs]

        altos_paths_pool = [altos_paths[i] for i in train_idxs]
        y_pool = [labels[i] for i in train_idxs]
        altos_ocr_flags_pool = [ocr_flags[i] for i in train_idxs]

        logger.info(f"Frozen split from metadata: {len(y_pool)} train candidates, {len(y_test)} test.")

        # Only the validation set is drawn, out of the train pool.
        seed = config["PARAMS"]["random_state"]
        validation_size = config["PARAMS"]["validation_size"]
        stratify = get_stratify(y_pool, altos_ocr_flags_pool, logger, context="train/val split")

        (
            y_train,
            y_val,
            altos_ocr_flags_train,
            altos_ocr_flags_val,
            altos_paths_train,
            altos_paths_val,
        ) = train_test_split(
            y_pool,
            altos_ocr_flags_pool,
            altos_paths_pool,
            stratify=stratify,
            test_size=validation_size,
            random_state=seed,
        )

        # SAVE DATASET
        dataset_split = {
            "train": {"paths": altos_paths_train, "labels": y_train, "ocr": altos_ocr_flags_train},
            "val": {"paths": altos_paths_val, "labels": y_val, "ocr": altos_ocr_flags_val},
            "test": {"paths": altos_paths_test, "labels": y_test, "ocr": altos_ocr_flags_test},
        }

        for split_name, split_data in dataset_split.items():
            logger.info(f"{split_name}: {len(split_data['labels'])} docs {Counter(split_data['labels'])}")

        os.makedirs(config["MLFLOW"]["folder_output"], exist_ok=True)

        path = os.path.join(config["MLFLOW"]["folder_output"], "dataset_split.pkl")

        if not (os.path.isfile(path)):
            with open(path, "wb") as f:
                pickle.dump(dataset_split, f)
        else:
            raise Exception(
                "Cannot save datasplit as one has already been defined for the current MLFlow "
                "experiments. If you want to change the datasplit, restart with a new data path "
                "in the config .yml file. This is done to ensure reproductibility."
            )

    except Exception as e:
        logger.error(f"Error occured : {e} ")
        logger.debug(f"Error occured : {e} ", stack_info=True, exc_info=True)
