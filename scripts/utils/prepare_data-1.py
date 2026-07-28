import sys

sys.path.append("/mnt/code")

import json

import pickle

import os

import shutil

from copy import deepcopy

from glob import glob

import pandas as pd

from tqdm import tqdm

from pathlib import Path

from sklearn.model_selection import train_test_split

from utils.utils_docintel.dataset_classes.CniNew import CniNew

from docintel_ml import Alto, DatasetCachedOnDisk, DatasetInRam


def prepare_data(config: dict, logger, df_docs: pd.DataFrame = None, export_path: str = None, kept_labels: list = None) -> None:
    """
    Prepare the data by splitting into training, validation, and test sets and saving them.

    Args:
        config (dict): Configuration dictionary containing parameters.
        df_docs (pd.DataFrame): DataFrame containing document metadata.
        export_path (str): Path to export the data.
        logger (logging.Logger): Logger instance for logging messages.
    """
    try:
        logger.info("***** Start Data Preparation ...")

        if df_docs is None:
            metadata_path = config['DATA_PATH']['metadata_path']
            if metadata_path:
                df_docs = pd.read_pickle(metadata_path)
                logger.info(f"Metadata imported from {metadata_path}")
            else:
                logger.error("Metadata path not specified in the configuration")
                raise ValueError("Metadata path not specified in the configuration")

        if export_path is None:
            export_path = config['DATA_PATH']['prepared_data_path']
            logger.info(f"Export path not specified, using default path: {export_path}")

        os.makedirs(export_path, exist_ok=True)

        df_docs_test = split_test_dataset(df_docs, config, logger)
        df_docs_train, df_docs_validation = split_validation_dataset(df_docs, config, logger)

        prepare_and_save_dataset(df_docs_train, os.path.join(export_path, "TRAIN"), kept_labels, logger)
        prepare_and_save_dataset(df_docs_validation, os.path.join(export_path, "VALIDATION"), kept_labels, logger)

        if df_docs_test is not None:
            if kept_labels:
                kept_labels += [label.replace('_anno', '') for label in kept_labels]  # All tags for 2 step of evaluation
                print(kept_labels)
            prepare_and_save_dataset(df_docs_test, os.path.join(export_path, "TEST"), kept_labels, logger)

        logger.info("***** End Data Preparation *****")

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        logger.debug(f"Error occurred: {e}", stack_info=True, exc_info=True)
        raise e


def split_test_dataset(df_docs: pd.DataFrame, config: dict, logger) -> pd.DataFrame:
    """
    Split the test dataset from the main DataFrame.

    config['parcours_test'] is expected to be a list, e.g. ["all"], ["eer"], ["eer", "maps"].
    ["all"] (default) means no filter, every parcours is kept.

    Args:
        df_docs (pd.DataFrame): DataFrame containing document information.
        config (dict): Configuration dictionary containing parameters.
        logger (logging.Logger): Logger instance for logging messages.

    Returns:
        pd.DataFrame: DataFrame containing the test dataset.
    """
    if "test" in df_docs['set'].values:
        df_docs_test = df_docs[df_docs['set'] == 'test']

        parcours_test = config.get('parcours_test', ["all"])

        if parcours_test != ["all"]:
            before = len(df_docs_test)
            df_docs_test = df_docs_test[df_docs_test['parcours'].isin(parcours_test)]
            logger.info(
                f"-- Filtered test on parcours={parcours_test}: "
                f"{before} -> {len(df_docs_test)} rows --"
            )

        logger.info(f"-- Test dataset found and split by percentage of: {round(len(df_docs_test) / len(df_docs), 2)} ...")
        return df_docs_test
    else:
        logger.info("-- No test dataset found --")
        return None


def split_validation_dataset(df_docs: pd.DataFrame, config: dict, logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the training dataset into training and validation sets.

    config['parcours_train'] is expected to be a list, e.g. ["eer"], ["eer", "maps"], ["all"].
    This filter is applied before the train/validation split and has no effect on the test dataset.

    Args:
        df_docs (pd.DataFrame): DataFrame containing document information.
        config (dict): Configuration dictionary containing parameters.
        logger (logging.Logger): Logger instance for logging messages.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tuple containing the training and validation DataFrames.
    """
    logger.info(f"-- Split validation dataset by percentage of: {config['train_validation_ratio']} --")
    if "train" in df_docs['set'].values:
        df_docs_train = df_docs[df_docs['set'] == 'train']

        parcours_train = config.get('parcours_train', ["eer"])

        if parcours_train != ["all"]:
            before = len(df_docs_train)
            df_docs_train = df_docs_train[df_docs_train['parcours'].isin(parcours_train)]
            logger.info(
                f"-- Filtered train/validation on parcours={parcours_train}: "
                f"{before} -> {len(df_docs_train)} rows --"
            )

        df_docs_train, df_docs_validation = train_test_split(
            df_docs_train,
            test_size=config["train_validation_ratio"],
            shuffle=True,
            random_state=config['random_state']
        )
        return df_docs_train, df_docs_validation
    else:
        logger.info("-- No train dataset found --")
        return None, None


def prepare_and_save_dataset(df_docs: pd.DataFrame, path: str, kept_labels: list, logger) -> None:
    """
    Prepare and save the dataset to the specified path.

    Args:
        df_docs (pd.DataFrame): DataFrame containing document information.
        path (str): Path to save the dataset.
        logger (logging.Logger): Logger instance for logging messages.
    """
    if df_docs is not None and not df_docs.empty:
        prepare_dataset(path, df_docs)
        prepare_pickle_dataset(path, kept_labels)
        logger.warning(f"--- Dataset preparation accomplished for {path} ---")


def prepare_dataset(path: str, df_docs: pd.DataFrame) -> None:
    """
    Prepare the dataset by saving alto and tags files.

    Args:
        path (str): Path to save the dataset.
        df_docs (pd.DataFrame): DataFrame containing document information.
    """
    alto_dir = os.path.join(path, "altos")
    tags_dir = os.path.join(path, "tags")

    # Clear any leftover files from a previous run so the folder only
    # ever reflects the current dataset (no stale docs mixed in).
    shutil.rmtree(alto_dir, ignore_errors=True)
    shutil.rmtree(tags_dir, ignore_errors=True)

    os.makedirs(alto_dir, exist_ok=True)
    os.makedirs(tags_dir, exist_ok=True)

    for idx, row in tqdm(df_docs.iterrows(), total=df_docs.shape[0]):
        id = row['id']
        alto = json.load(open(row["alto_path"], "r"))
        annot = json.load(open(row["tags_path"], "r"))

        doc_annot = deepcopy(annot)

        # filter tags by status ( to avoid deleted tags)
        doc_annot["tags"] = [tag for tag in doc_annot["tags"] if tag["status"] == "validated"]

        if any(path.endswith(suffix) for suffix in ['train', 'validation']):
            doc_annot["tags"] = [tag for tag in doc_annot["tags"] if tag["type"].endswith("_anno")]

        with open(f"{alto_dir}/{id}.json", "w") as f:
            json.dump(alto, f, indent=1)

        with open(f"{tags_dir}/{id}.json", "w") as f:
            json.dump(doc_annot, f, indent=1)


def prepare_pickle_dataset(path: str, kept_labels: list) -> None:
    """
    Prepare and save the dataset as a pickle file.

    Args:
        path (str): Path to save the dataset.
    """
    dataset = CniNew(data_path=path, groups_to_aggregate=None, sorted_alto=True, kept_labels=kept_labels)
    dataset = DatasetCachedOnDisk(dataset, path)
    dataset.remove_empty_altos()
    dataset.remove_empty_gts()
    dataset.save(path)


def value_exists(tag: dict) -> bool:
    """
    Check if the tag has valid occurrences.

    Args:
        tag (dict): Tag dictionary containing occurrences.

    Returns:
        bool: True if the tag has valid occurrences, False otherwise.
    """
    occurrences = tag.get("occurrences")
    list_valpos = []
    if not occurrences or not isinstance(occurrences, list):
        return False
    for occ in occurrences:
        valpos = occ.get("valuePositions", [])
        list_valpos += valpos if isinstance(valpos, list) else []
    return bool(list_valpos)
