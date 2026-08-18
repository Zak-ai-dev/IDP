import json
import logging
import os
import re
import warnings
from collections import defaultdict

import pandas as pd
from minio import Minio

from utils.minio.minio_document import MINIODocument

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# Default regex capturing the numeric index at the end of a filename.
# Handles: cni_new_0157.pdf, cni_new_157.pdf, test_cni_new_0157.pdf, TRI_CNI_ETR_0297.pdf
DEFAULT_INDEX_PATTERN = r"_(\d+)\.[^.]+$"


# =============================================================================
# MinIO client
# =============================================================================
def initialize_cos_client(workspace_id: int, environment: str = "PROD") -> MINIODocument:
    """
    Initialize the MinIO client for a given workspace.

    Args:
        workspace_id (int): Workspace id, used to build the MinIO prefix.
        environment (str): "PROD" or "QUAL".

    Returns:
        MINIODocument: Initialized MinIO client.
    """
    env_prefix = "QUAL" if environment == "QUAL" else "PROD"

    cos_client = Minio(
        endpoint=os.getenv(f"{env_prefix}_TRAINDATA_ENDPOINT_URL"),
        access_key=os.getenv(f"{env_prefix}_TRAINDATA_ACCESS_KEY"),
        secret_key=os.getenv(f"{env_prefix}_TRAINDATA_SECRET_KEY"),
        secure=True,
        cert_check=False,
    )
    prefix = f"docfactory_export/workspace/{workspace_id}/"

    return MINIODocument(
        minio_client=cos_client,
        bucket=os.getenv(f"{env_prefix}_TRAINDATA_BUCKET_NAME"),
        path=prefix,
        workspace_id=workspace_id,
        recursive=True,
    )


# =============================================================================
# Document listing / metadata
# =============================================================================
def list_documents_from_minio(
    minio_client: MINIODocument,
    logger: logging.Logger,
    target_status: list[str] = ["validated"],
) -> pd.DataFrame:
    """
    List the documents of a MinIO workspace and filter on the target status.
    Nothing is downloaded here: this only builds the metadata DataFrame.

    Args:
        minio_client (MINIODocument): Initialized MinIO client.
        logger (logging.Logger): Logger instance.
        target_status (list[str]): Statuses to keep.

    Returns:
        pd.DataFrame: DataFrame of documents, deduplicated on filename.
    """
    doc_infos = minio_client.get_documents_info_from_annot()
    df_doc_infos = pd.DataFrame(doc_infos)

    if df_doc_infos.empty:
        logger.warning("--- No document returned by MinIO for this workspace.")
        return df_doc_infos

    df_doc_infos["updatedAt"] = pd.to_datetime(df_doc_infos["updatedAt"])
    df_doc_infos["createdAt"] = pd.to_datetime(df_doc_infos["createdAt"])

    logger.info(f"--- {len(df_doc_infos['filename'].unique())} documents found...")
    logger.info(f"--- {len(df_doc_infos['id'].unique())} uploads...")

    status_found = df_doc_infos["status"].unique().tolist()
    assert all(
        e in status_found for e in target_status
    ), f"Expected status not found among uploads. Status found: {status_found}"

    df_doc_infos = df_doc_infos[df_doc_infos["status"].isin(target_status)].copy()

    # Deduplication: keep the most recent upload per filename
    df_doc_dupp = df_doc_infos[df_doc_infos.duplicated("filename", keep=False)].sort_values("filename").copy()
    logger.info(f"--- {len(df_doc_dupp['filename'].unique())} documents with multiple uploads...")
    logger.info("--- Process duplicates by keeping the most recent...")

    df_doc_infos = pd.concat(
        [
            df_doc_infos[~df_doc_infos["filename"].duplicated(keep=False)],
            df_doc_dupp.loc[df_doc_dupp.groupby("filename")["updatedAt"].idxmax()],
        ]
    )

    logger.info(f"--- Final number of documents listed: {len(df_doc_infos['id'].unique())} ...")
    return df_doc_infos


# =============================================================================
# Filtering by document type (multi-label workspace)
# =============================================================================
def filter_by_filename_regex(df: pd.DataFrame, label_cfg: dict, logger: logging.Logger) -> pd.DataFrame:
    """
    Keep only the documents whose filename matches the label regex.
    Required on a multi-label workspace (exploration workspace 77) where several
    document types coexist under the TRI_<TYPE>_XXXX.pdf naming convention.

    No-op when the label has no "filename_regex" key.

    Args:
        df (pd.DataFrame): Documents listed for the workspace.
        label_cfg (dict): Label configuration entry.
        logger (logging.Logger): Logger instance.

    Returns:
        pd.DataFrame: DataFrame restricted to the label document type.
    """
    pattern = label_cfg.get("filename_regex")
    if not pattern:
        return df

    label = label_cfg["label"]
    before = len(df)
    df = df[df["filename"].str.match(pattern, na=False)].copy()
    logger.info(f"--- [{label}] filename filter '{pattern}': {before} -> {len(df)} docs")
    return df


# =============================================================================
# Train / test split and index exclusions
# =============================================================================
def get_index_from_filename(filename: str, pattern: str = DEFAULT_INDEX_PATTERN):
    """
    Extract the numeric index located at the end of a filename, before the extension.

    Args:
        filename (str): Document filename.
        pattern (str): Capturing regex (group 1 = index).

    Returns:
        int | None: Extracted index, or None when not parsable.
    """
    match = re.search(pattern, filename)
    return int(match.group(1)) if match else None


def assign_set(filename: str, split_mode: str = "strict"):
    """
    Determine train/test membership from the filename prefix.

    Args:
        filename (str): Document filename.
        split_mode (str):
            - "strict": the filename must start with "train" or "test", otherwise
              the document is discarded (None).
            - "test_only": only the "test" prefix is discriminant, everything else
              falls back to "train".

    Returns:
        str | None: "train", "test", or None when the document must be discarded.

    Note:
        The "random" mode is not handled here: it does not depend on the filename
        and is handled directly in sample_train_test.
    """
    fname_lower = filename.lower()

    if split_mode == "test_only":
        return "test" if fname_lower.startswith("test") else "train"

    if fname_lower.startswith("test"):
        return "test"
    if fname_lower.startswith("train"):
        return "train"
    return None


def exclude_index_ranges(df: pd.DataFrame, label_cfg: dict, logger: logging.Logger) -> pd.DataFrame:
    """
    Discard documents whose index falls into an excluded range (out-of-scope journey).
    Ranges are defined separately for "train" and "test", since both lots are
    numbered independently.

    No-op when the label has no "excluded_index_ranges" key in the configuration.

    Args:
        df (pd.DataFrame): DataFrame already holding a "set" column.
        label_cfg (dict): Label configuration entry.
        logger (logging.Logger): Logger instance.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    ranges_by_set = label_cfg.get("excluded_index_ranges") or {}
    if not ranges_by_set:
        return df

    label = label_cfg["label"]
    pattern = label_cfg.get("index_pattern", DEFAULT_INDEX_PATTERN)

    df = df.copy()
    df["doc_index"] = df["filename"].apply(lambda f: get_index_from_filename(f, pattern))

    n_unparsed = int(df["doc_index"].isna().sum())
    if n_unparsed:
        logger.warning(f"--- [{label}] {n_unparsed} filenames without a parsable index, kept by default")

    mask_excluded = pd.Series(False, index=df.index)
    for split, ranges in ranges_by_set.items():
        in_split = df["set"] == split
        for low, high in ranges:
            mask_excluded |= in_split & df["doc_index"].between(low, high)

    before_counts = df["set"].value_counts().to_dict()
    df = df[~mask_excluded].copy()
    logger.info(f"--- [{label}] index exclusion: {before_counts} -> {df['set'].value_counts().to_dict()}")
    return df


def sample_random_split(
    df: pd.DataFrame, label: str, n_train: int, n_test: int, seed: int, logger: logging.Logger
) -> pd.DataFrame:
    """
    Sampling for labels whose filenames carry no train/test information
    (exploration workspace 77, TRI_<TYPE>_XXXX.pdf naming convention).

    The test set is drawn first, then the train set is drawn from the remainder:
    both lots are therefore disjoint by construction, with no leakage between splits.

    Args:
        df (pd.DataFrame): Documents of the given document type.
        label (str): Processed label (used for logging).
        n_train (int): Number of train documents requested.
        n_test (int): Number of test documents requested.
        seed (int): Random seed.
        logger (logging.Logger): Logger instance.

    Returns:
        pd.DataFrame: Sampled documents, with a "set" column.
    """
    df = df.copy()

    n_test_eff = min(n_test, len(df))
    if n_test_eff < n_test:
        logger.warning(f"--- [{label}] only {len(df)} docs available (test requested: {n_test})")
    df_test = df.sample(n=n_test_eff, random_state=seed)

    df_rest = df.drop(index=df_test.index)
    n_train_eff = min(n_train, len(df_rest))
    if n_train_eff < n_train:
        logger.warning(f"--- [{label}] only {len(df_rest)} docs remaining (train requested: {n_train})")
    df_train = df_rest.sample(n=n_train_eff, random_state=seed)

    df_test["set"] = "test"
    df_train["set"] = "train"
    return pd.concat([df_train, df_test], ignore_index=True)


def sample_train_test(
    df: pd.DataFrame, label_cfg: dict, sampling_cfg: dict, logger: logging.Logger
) -> pd.DataFrame:
    """
    Apply the full selection chain for a label:
        1) train/test split according to the label split_mode,
        2) exclusion of the index ranges defined in the configuration (train and test),
        3) random sampling of n_train and n_test documents.

    Args:
        df (pd.DataFrame): Documents listed for the label (already filtered by type).
        label_cfg (dict): Label configuration entry.
        sampling_cfg (dict): "sampling" block of the configuration.
        logger (logging.Logger): Logger instance.

    Returns:
        pd.DataFrame: Sampled documents, with a "set" column.
    """
    label = label_cfg["label"]
    split_mode = label_cfg.get("split_mode", "strict")
    n_train = sampling_cfg.get("n_train", 200)
    n_test = sampling_cfg.get("n_test", 100)
    seed = sampling_cfg.get("seed", 42)

    df = df.copy()

    # No train/test hint in the filename: the split is drawn at random.
    if split_mode == "random":
        df_sampled = sample_random_split(df, label, n_train, n_test, seed, logger)
        logger.info(f"--- [{label}] final sample: {df_sampled['set'].value_counts().to_dict()}")
        return df_sampled

    df["set"] = df["filename"].apply(lambda f: assign_set(f, split_mode))

    before = len(df)
    df = df[df["set"].notna()].copy()
    logger.info(f"--- [{label}] {before} -> {len(df)} docs after split (mode={split_mode})")

    df = exclude_index_ranges(df, label_cfg, logger)

    sampled = []
    for split, n in [("train", n_train), ("test", n_test)]:
        pool = df[df["set"] == split]
        if len(pool) < n:
            logger.warning(f"--- [{label}] only {len(pool)} '{split}' docs available (requested: {n})")
            sampled.append(pool)
        else:
            sampled.append(pool.sample(n=n, random_state=seed))

    df_sampled = pd.concat(sampled, ignore_index=True)
    logger.info(f"--- [{label}] final sample: {df_sampled['set'].value_counts().to_dict()}")
    return df_sampled


# =============================================================================
# Alto download
# =============================================================================
def get_alto_path(id: str, export_path: str) -> str:
    """Build the local path of a document alto file."""
    return os.path.join(export_path, "altos", f"{id}.json")


def save_json(data: dict, file_path: str) -> None:
    """
    Save data as JSON to a file.

    Args:
        data (dict): Data to be saved.
        file_path (str): Path to the file where data will be saved.
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=1)


def download_dataset(
    minio_client: MINIODocument,
    df_doc_infos: pd.DataFrame,
    export_path: str,
    logger: logging.Logger,
) -> None:
    """
    Download the alto files of the selected documents into export_path/altos.
    export_path is dedicated to a single label, hence one folder per document type.

    Args:
        minio_client (MINIODocument): Initialized MinIO client.
        df_doc_infos (pd.DataFrame): Documents to download (already sampled).
        export_path (str): Destination folder of the label.
        logger (logging.Logger): Logger instance.
    """
    path_dir_alto = os.path.join(export_path, "altos")
    os.makedirs(path_dir_alto, exist_ok=True)

    n_ok, n_ko = 0, 0
    for id in df_doc_infos["id"].to_numpy():
        try:
            doc = minio_client.get_document(id)
            save_json(doc["alto"], f"{path_dir_alto}/{id}.json")
            n_ok += 1
        except Exception as e:
            n_ko += 1
            logger.warning(f"--- Failed to download alto {id}: {e}")

    logger.info(f"--- {n_ok} altos downloaded into {path_dir_alto} ({n_ko} failures)")


# =============================================================================
# Metadata
# =============================================================================
def save_metadata(df_doc_infos: pd.DataFrame, config: dict, logger: logging.Logger) -> None:
    """
    Save the global metadata (all labels combined) to a single pickle file.

    Args:
        df_doc_infos (pd.DataFrame): Cumulated DataFrame of every downloaded document.
        config (dict): Configuration dictionary containing metadata path.
        logger (logging.Logger): Logger instance.
    """
    metadata_path = os.path.join(config["DATA"]["data_path"], "metadata.pkl")
    os.makedirs(config["DATA"]["data_path"], exist_ok=True)
    df_doc_infos.to_pickle(metadata_path)
    logger.info(f"***** Metadata saved successfully into {metadata_path} ({len(df_doc_infos)} docs)")


# =============================================================================
# Workspace loading
# =============================================================================
def load_data_from_wks(
    config: dict,
    logger: logging.Logger,
    workspace_id: int,
    label_cfgs: list[dict],
    environment: str = "PROD",
) -> list[pd.DataFrame]:
    """
    Load the data of a workspace for one or several labels.

    The MinIO listing is performed only once, even when several labels share the
    workspace (exploration workspace 77): each label then isolates its own document
    type through its "filename_regex", samples it and downloads it into its own folder.

    The download happens AFTER sampling, so that only the selected documents are
    retrieved instead of the whole workspace.

    Args:
        config (dict): Configuration dictionary.
        logger (logging.Logger): Logger instance.
        workspace_id (int): Workspace to load.
        label_cfgs (list[dict]): Configuration entries of the labels bound to this workspace.
        environment (str): "PROD" or "QUAL".

    Returns:
        list[pd.DataFrame]: One metadata DataFrame per loaded label.
    """
    labels = [cfg["label"] for cfg in label_cfgs]
    logger.info(f"***** Start Data Loading - workspace {workspace_id} - labels {labels} ...")

    dfs = []
    try:
        sampling_cfg = config["DATA"].get("sampling", {})
        default_status = config["DATA"].get("default_target_status", ["validated"])

        # The status is a workspace-level property: take the one of the first label
        # and warn when labels of a same workspace disagree.
        target_status = label_cfgs[0].get("target_status", default_status)
        if any(cfg.get("target_status", default_status) != target_status for cfg in label_cfgs):
            logger.warning(
                f"--- Workspace {workspace_id}: diverging statuses between labels, using {target_status}"
            )

        minio_client = initialize_cos_client(workspace_id=workspace_id, environment=environment)
        df_wks = list_documents_from_minio(minio_client, logger, target_status=target_status)

        if df_wks.empty:
            return dfs

        for label_cfg in label_cfgs:
            label = label_cfg["label"]
            export_path = os.path.join(config["DATA"]["data_path"], label)
            os.makedirs(export_path, exist_ok=True)

            df_label = filter_by_filename_regex(df_wks, label_cfg, logger)
            if df_label.empty:
                logger.warning(f"--- [{label}] no document left after the filename filter.")
                continue

            df_sampled = sample_train_test(df_label, label_cfg, sampling_cfg, logger)
            if df_sampled.empty:
                logger.warning(f"--- [{label}] no document left after sampling.")
                continue

            download_dataset(minio_client, df_sampled, export_path, logger)

            df_sampled["label"] = label
            df_sampled["document_type"] = label
            df_sampled["workspace_id"] = workspace_id
            df_sampled["alto_path"] = df_sampled["id"].apply(lambda id_: get_alto_path(id_, export_path))
            dfs.append(df_sampled)

        logger.info(f"***** End Data Loading - workspace {workspace_id} ...")
        return dfs

    except Exception as e:
        logger.error(f"Error occurred on workspace {workspace_id} (labels {labels}): {e}")
        logger.debug(f"Error occurred on workspace {workspace_id}: {e}", stack_info=True, exc_info=True)
        return dfs


# =============================================================================
# Orchestrator
# =============================================================================
def load_data(config: dict, logger: logging.Logger, environment: str = "PROD") -> pd.DataFrame:
    """
    Orchestrate the loading of every enabled label declared in the configuration.

    - A single MinIO listing per workspace, even when several labels share it.
    - Alto files are stored in a dedicated folder per document type.
    - A single metadata.pkl summarizes every downloaded document.

    Args:
        config (dict): Configuration dictionary.
        logger (logging.Logger): Logger instance.
        environment (str): "PROD" or "QUAL".

    Returns:
        pd.DataFrame: Cumulated metadata of every loaded label.
    """
    logger.info("***** Start global Data Loading ...")

    entries_by_wks = defaultdict(list)
    for entry in config["DATA"]["labels"]:
        if not entry.get("enabled", True):
            logger.info(f"--- Label '{entry['label']}' disabled in configuration, skipped.")
            continue
        entries_by_wks[entry["workspaceId"]].append(entry)

    all_dfs = []
    for wks_id, label_cfgs in entries_by_wks.items():
        all_dfs.extend(
            load_data_from_wks(
                config=config,
                logger=logger,
                workspace_id=wks_id,
                label_cfgs=label_cfgs,
                environment=environment,
            )
        )

    if not all_dfs:
        logger.error("***** No document loaded, metadata not saved.")
        return pd.DataFrame()

    df_all = pd.concat(all_dfs, ignore_index=True)
    save_metadata(df_all, config, logger)

    logger.info("***** Loading summary:")
    recap = df_all.groupby(["document_type", "set"]).size().unstack(fill_value=0)
    for line in recap.to_string().split("\n"):
        logger.info(f"    {line}")
    logger.info("***** End global Data Loading ...")

    return df_all
