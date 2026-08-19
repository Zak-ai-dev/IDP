import hashlib
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

# Column used as the stable identity of a document across runs. The filename is
# preferred over the upload id, which changes whenever a document is re-uploaded.
STABLE_KEY_COLUMN = "filename"

# Hash scores are normalized into this range, so that a fraction of the space can
# be compared against a threshold (used by the random train/test split).
HASH_SPACE = 10_000

# Name of the metadata file, always written under data_path.
METADATA_FILENAME = "metadata.pkl"


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


def stable_hash_score(key: str, seed: int, namespace: str = "") -> int:
    """
    Compute a deterministic score in [0, HASH_SPACE) for a given document key.

    The score is an intrinsic property of the document: it depends only on the key,
    the seed and the namespace, never on the size or the ordering of the pool it
    belongs to. Two runs therefore yield the same score for the same document, even
    when documents have been added to or removed from the workspace in between.

    Args:
        key (str): Stable document key (filename).
        seed (int): Seed shared by the whole selection.
        namespace (str): Optional discriminator, used to decorrelate the split
            decision from the sampling decision for a same document.

    Returns:
        int: Score in [0, HASH_SPACE).
    """
    digest = hashlib.md5(f"{namespace}|{seed}|{key}".encode("utf-8")).hexdigest()
    return int(digest, 16) % HASH_SPACE


def assign_set_by_hash(df: pd.DataFrame, label: str, n_train: int, n_test: int, seed: int) -> pd.DataFrame:
    """
    Assign train/test membership by hash, for labels whose filenames carry no
    train/test information (exploration workspace 77, TRI_<TYPE>_XXXX.pdf).

    The test share is derived from the requested volumes, then applied as a threshold
    on the hash space. A given document always falls on the same side of the threshold,
    so it can never drift from train to test between two runs.

    Args:
        df (pd.DataFrame): Documents of the given document type.
        label (str): Processed label, used as hash namespace.
        n_train (int): Number of train documents requested.
        n_test (int): Number of test documents requested.
        seed (int): Seed shared by the whole selection.

    Returns:
        pd.DataFrame: DataFrame with a "set" column.
    """
    df = df.copy()
    test_share = n_test / (n_train + n_test)
    threshold = test_share * HASH_SPACE

    split_score = df[STABLE_KEY_COLUMN].apply(lambda k: stable_hash_score(k, seed, f"split|{label}"))
    df["set"] = pd.Series(["test" if s < threshold else "train" for s in split_score], index=df.index)
    return df


def select_by_hash(df: pd.DataFrame, n: int, label: str, seed: int) -> pd.DataFrame:
    """
    Select n documents by keeping the lowest hash scores.

    Because the score of a document never changes, growing the pool does not reshuffle
    the previous selection: already selected documents whose score stays within the
    lowest n are kept. This gives incremental stability instead of a full redraw.

    Args:
        df (pd.DataFrame): Pool to select from.
        n (int): Number of documents to keep.
        label (str): Processed label, used as hash namespace.
        seed (int): Seed shared by the whole selection.

    Returns:
        pd.DataFrame: Selected documents.
    """
    if len(df) <= n:
        return df.copy()

    df = df.copy()
    df["_hash_score"] = df[STABLE_KEY_COLUMN].apply(lambda k: stable_hash_score(k, seed, f"select|{label}"))
    return df.nsmallest(n, "_hash_score", keep="first").drop(columns=["_hash_score"])


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

    if split_mode == "random":
        # No train/test hint in the filename: the split is derived from the hash.
        df = assign_set_by_hash(df, label, n_train, n_test, seed)
        logger.info(f"--- [{label}] hash split: {df['set'].value_counts().to_dict()}")
    else:
        df["set"] = df["filename"].apply(lambda f: assign_set(f, split_mode))
        before = len(df)
        df = df[df["set"].notna()].copy()
        logger.info(f"--- [{label}] {before} -> {len(df)} docs after split (mode={split_mode})")

    df = exclude_index_ranges(df, label_cfg, logger)

    selected = []
    for split, n in [("train", n_train), ("test", n_test)]:
        pool = df[df["set"] == split]
        if len(pool) < n:
            logger.warning(f"--- [{label}] only {len(pool)} '{split}' docs available (requested: {n})")
        selected.append(select_by_hash(pool, n, label, seed))

    df_sampled = pd.concat(selected, ignore_index=True)
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
# Previous selection (replayed from the existing metadata)
# =============================================================================
def get_metadata_path(config: dict) -> str:
    """Build the path of the global metadata file."""
    return os.path.join(config["DATA"]["data_path"], METADATA_FILENAME)


def load_previous_selection(config: dict, logger: logging.Logger) -> pd.DataFrame:
    """
    Load the selection of a previous run from the existing metadata.

    The metadata already records, for every document, its label, its stable key and
    its split: it is therefore self-sufficient to replay a frozen lot, and no
    separate manifest file is needed.

    Args:
        config (dict): Configuration dictionary.
        logger (logging.Logger): Logger instance.

    Returns:
        pd.DataFrame: Previous metadata, empty when no previous run exists.
    """
    metadata_path = get_metadata_path(config)
    if not os.path.exists(metadata_path):
        logger.info(f"--- No previous metadata at {metadata_path}, the selection will be computed.")
        return pd.DataFrame()

    df_previous = pd.read_pickle(metadata_path)
    logger.info(f"--- Previous selection loaded from {metadata_path} ({len(df_previous)} docs)")
    return df_previous


def apply_previous_selection(
    df: pd.DataFrame, df_previous: pd.DataFrame, label: str, logger: logging.Logger
) -> pd.DataFrame:
    """
    Restrict the pool to the exact documents selected by a previous run for this label.

    This bypasses the hash selection entirely: the frozen lot is reproduced as is,
    which is what makes a training dataset auditable across runs.

    Args:
        df (pd.DataFrame): Documents listed for the label.
        df_previous (pd.DataFrame): Metadata of the previous run.
        label (str): Processed label.
        logger (logging.Logger): Logger instance.

    Returns:
        pd.DataFrame: Documents of the frozen lot, with their recorded "set".
    """
    if df_previous.empty or "label" not in df_previous.columns:
        return pd.DataFrame()

    entries = df_previous[df_previous["label"] == label]
    if entries.empty:
        return pd.DataFrame()

    set_by_key = dict(zip(entries[STABLE_KEY_COLUMN], entries["set"]))
    df_frozen = df[df[STABLE_KEY_COLUMN].isin(set_by_key)].copy()
    df_frozen["set"] = df_frozen[STABLE_KEY_COLUMN].map(set_by_key)

    n_missing = len(set_by_key) - len(df_frozen)
    if n_missing:
        logger.warning(
            f"--- [{label}] {n_missing} documents of the previous run are no longer available "
            f"in the workspace"
        )

    logger.info(f"--- [{label}] previous selection replayed: {df_frozen['set'].value_counts().to_dict()}")
    return df_frozen


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
    metadata_path = get_metadata_path(config)
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
    df_previous: pd.DataFrame = None,
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
        df_previous (pd.DataFrame): Metadata of a previous run. When it holds entries
            for a label, that lot is replayed instead of being recomputed.

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

            # A frozen lot takes precedence over the hash selection, so that an
            # already published dataset is reproduced exactly.
            df_sampled = pd.DataFrame()
            if df_previous is not None and not df_previous.empty:
                df_sampled = apply_previous_selection(df_label, df_previous, label, logger)

            if df_sampled.empty:
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
def load_data(
    config: dict,
    logger: logging.Logger,
    environment: str = "PROD",
) -> pd.DataFrame:
    """
    Orchestrate the loading of every enabled label declared in the configuration.

    - A single MinIO listing per workspace, even when several labels share it.
    - Alto files are stored in a dedicated folder per document type.
    - A single metadata file summarizes every downloaded document.

    Reproducibility relies on two complementary mechanisms. Selection is driven by a
    per-document hash rather than a global draw, so a document keeps its score and its
    split whatever happens to the rest of the workspace. On top of that, the metadata
    of the previous run is replayed when "reuse_previous" is enabled: the very same
    documents are reloaded, even if the workspace of a document type changes.

    Args:
        config (dict): Configuration dictionary.
        logger (logging.Logger): Logger instance.
        environment (str): "PROD" or "QUAL".

    Returns:
        pd.DataFrame: Cumulated metadata of every loaded label.
    """
    logger.info("***** Start global Data Loading ...")

    reuse_previous = config["DATA"].get("reuse_previous", True)
    df_previous = load_previous_selection(config, logger) if reuse_previous else None
    if not reuse_previous:
        logger.info("--- Previous selection ignored, the lot is recomputed from scratch.")

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
                df_previous=df_previous,
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
