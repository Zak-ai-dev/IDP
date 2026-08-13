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

# Regex par défaut de capture de l'index numérique en fin de nom de fichier.
# Gère : cni_new_0157.pdf, cni_new_157.pdf, test_cni_new_0157.pdf, cni_old_0157.pdf
DEFAULT_INDEX_PATTERN = r"_(\d+)\.[^.]+$"


# =============================================================================
# Client MinIO
# =============================================================================
def initialize_cos_client(workspace_id: int, environment: str = "PROD") -> MINIODocument:
    """
    Initialise le client MinIO pour un workspace donné.

    Args:
        workspace_id (int): Identifiant du workspace, utilisé pour construire le prefix MinIO.
        environment (str): "PROD" ou "QUAL".

    Returns:
        MINIODocument: Client MinIO initialisé.
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
# Listing / métadonnées documents
# =============================================================================
def list_documents_from_minio(
    minio_client: MINIODocument,
    logger: logging.Logger,
    target_status: list[str] = ["validated"],
) -> pd.DataFrame:
    """
    Liste les documents d'un workspace MinIO et filtre sur le statut cible.
    Ne télécharge rien : ne fait que construire le DataFrame de métadonnées.

    Args:
        minio_client (MINIODocument): Client MinIO initialisé.
        logger (logging.Logger): Logger.
        target_status (list[str]): Statuts à conserver.

    Returns:
        pd.DataFrame: DataFrame des documents (dédoublonné sur filename).
    """
    doc_infos = minio_client.get_documents_info_from_annot()
    df_doc_infos = pd.DataFrame(doc_infos)

    if df_doc_infos.empty:
        logger.warning("--- Aucun document retourné par MinIO pour ce workspace.")
        return df_doc_infos

    df_doc_infos["updatedAt"] = pd.to_datetime(df_doc_infos["updatedAt"])
    df_doc_infos["createdAt"] = pd.to_datetime(df_doc_infos["createdAt"])

    logger.info(f"--- {len(df_doc_infos['filename'].unique())} documents trouvés...")
    logger.info(f"--- {len(df_doc_infos['id'].unique())} uploads...")

    status_found = df_doc_infos["status"].unique().tolist()
    assert all(
        e in status_found for e in target_status
    ), f"Statut attendu introuvable parmi les uploads. Statuts trouvés: {status_found}"

    df_doc_infos = df_doc_infos[df_doc_infos["status"].isin(target_status)].copy()

    # Déduplication : on garde l'upload le plus récent par filename
    df_doc_dupp = df_doc_infos[df_doc_infos.duplicated("filename", keep=False)].sort_values("filename").copy()
    logger.info(f"--- {len(df_doc_dupp['filename'].unique())} documents avec uploads multiples...")
    logger.info("--- Traitement des doublons : conservation du plus récent...")

    df_doc_infos = pd.concat(
        [
            df_doc_infos[~df_doc_infos["filename"].duplicated(keep=False)],
            df_doc_dupp.loc[df_doc_dupp.groupby("filename")["updatedAt"].idxmax()],
        ]
    )

    logger.info(f"--- Nombre final de documents listés : {len(df_doc_infos['id'].unique())} ...")
    return df_doc_infos


# =============================================================================
# Split train / test et exclusions d'index
# =============================================================================
def get_index_from_filename(filename: str, pattern: str = DEFAULT_INDEX_PATTERN):
    """
    Extrait l'index numérique situé en fin de nom de fichier, avant l'extension.

    Args:
        filename (str): Nom du fichier.
        pattern (str): Regex de capture (groupe 1 = index).

    Returns:
        int | None: Index extrait, ou None si non parsable.
    """
    match = re.search(pattern, filename)
    return int(match.group(1)) if match else None


def assign_set(filename: str, split_mode: str = "strict"):
    """
    Détermine l'appartenance train/test à partir du préfixe du nom de fichier.

    Args:
        filename (str): Nom du fichier.
        split_mode (str):
            - "strict"    : le nom doit commencer par "train" ou "test", sinon exclu (None).
            - "test_only" : seul le préfixe "test" est discriminant, tout le reste -> "train".

    Returns:
        str | None: "train", "test", ou None si à exclure.
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
    Écarte les documents dont l'index tombe dans une plage exclue (parcours hors périmètre).
    Les plages sont définies séparément pour "train" et "test", car les numérotations
    des deux lots sont indépendantes.

    No-op si le label n'a pas de clé "excluded_index_ranges" dans la config.

    Args:
        df (pd.DataFrame): DataFrame contenant déjà une colonne "set".
        label_cfg (dict): Entrée de config du label.
        logger (logging.Logger): Logger.

    Returns:
        pd.DataFrame: DataFrame filtré.
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
        logger.warning(f"--- [{label}] {n_unparsed} filenames sans index parsable, conservés par défaut")

    mask_excluded = pd.Series(False, index=df.index)
    for split, ranges in ranges_by_set.items():
        in_split = df["set"] == split
        for low, high in ranges:
            mask_excluded |= in_split & df["doc_index"].between(low, high)

    before_counts = df["set"].value_counts().to_dict()
    df = df[~mask_excluded].copy()
    logger.info(f"--- [{label}] exclusion index : {before_counts} -> {df['set'].value_counts().to_dict()}")
    return df


def sample_train_test(
    df: pd.DataFrame, label_cfg: dict, sampling_cfg: dict, logger: logging.Logger
) -> pd.DataFrame:
    """
    Applique la chaîne complète de sélection pour un label :
        1) split train/test selon le split_mode du label,
        2) exclusion des plages d'index définies en config (train et test),
        3) échantillonnage aléatoire de n_train et n_test documents.

    Args:
        df (pd.DataFrame): Documents listés pour le label.
        label_cfg (dict): Entrée de config du label.
        sampling_cfg (dict): Bloc "sampling" de la config.
        logger (logging.Logger): Logger.

    Returns:
        pd.DataFrame: Échantillon retenu, avec colonne "set".
    """
    label = label_cfg["label"]
    split_mode = label_cfg.get("split_mode", "strict")
    n_train = sampling_cfg.get("n_train", 200)
    n_test = sampling_cfg.get("n_test", 100)
    seed = sampling_cfg.get("seed", 42)

    df = df.copy()
    df["set"] = df["filename"].apply(lambda f: assign_set(f, split_mode))

    before = len(df)
    df = df[df["set"].notna()].copy()
    logger.info(f"--- [{label}] {before} -> {len(df)} docs après split (mode={split_mode})")

    df = exclude_index_ranges(df, label_cfg, logger)

    sampled = []
    for split, n in [("train", n_train), ("test", n_test)]:
        pool = df[df["set"] == split]
        if len(pool) < n:
            logger.warning(f"--- [{label}] seulement {len(pool)} docs '{split}' disponibles (demandé : {n})")
            sampled.append(pool)
        else:
            sampled.append(pool.sample(n=n, random_state=seed))

    df_sampled = pd.concat(sampled, ignore_index=True)
    logger.info(f"--- [{label}] échantillon final : {df_sampled['set'].value_counts().to_dict()}")
    return df_sampled


# =============================================================================
# Téléchargement des altos
# =============================================================================
def get_alto_path(id: str, export_path: str) -> str:
    """Construit le chemin local de l'alto d'un document."""
    return os.path.join(export_path, "altos", f"{id}.json")


def save_json(data: dict, file_path: str) -> None:
    """Sauvegarde un dictionnaire au format JSON."""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=1)


def download_dataset(
    minio_client: MINIODocument,
    df_doc_infos: pd.DataFrame,
    export_path: str,
    logger: logging.Logger,
) -> None:
    """
    Télécharge les altos des documents retenus dans export_path/altos.
    export_path est déjà dédié à un label, donc un dossier par typologie de document.

    Args:
        minio_client (MINIODocument): Client MinIO initialisé.
        df_doc_infos (pd.DataFrame): Documents à télécharger (déjà échantillonnés).
        export_path (str): Dossier de destination du label.
        logger (logging.Logger): Logger.
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
            logger.warning(f"--- Échec du téléchargement de l'alto {id} : {e}")

    logger.info(f"--- {n_ok} altos téléchargés dans {path_dir_alto} ({n_ko} échecs)")


# =============================================================================
# Metadata
# =============================================================================
def save_metadata(df_doc_infos: pd.DataFrame, config: dict, logger: logging.Logger) -> None:
    """
    Sauvegarde le metadata global (tous labels confondus) dans un unique pickle.

    Args:
        df_doc_infos (pd.DataFrame): DataFrame cumulé de tous les documents téléchargés.
        config (dict): Configuration.
        logger (logging.Logger): Logger.
    """
    metadata_path = os.path.join(config["DATA"]["data_path"], "metadata.pkl")
    os.makedirs(config["DATA"]["data_path"], exist_ok=True)
    df_doc_infos.to_pickle(metadata_path)
    logger.info(f"***** Metadata sauvegardé dans {metadata_path} ({len(df_doc_infos)} docs)")


# =============================================================================
# Chargement d'un workspace
# =============================================================================
def load_data(
    config: dict,
    logger: logging.Logger,
    workspace_id: int,
    label_cfg: dict,
    export_path: str = None,
    environment: str = None,
) -> pd.DataFrame:
    """
    Charge les données d'un workspace mono-label :
    listing MinIO -> split/exclusions/échantillonnage -> téléchargement des altos retenus.

    Le téléchargement intervient APRÈS l'échantillonnage, afin de ne récupérer
    que les documents sélectionnés et non l'intégralité du workspace.

    Args:
        config (dict): Configuration.
        logger (logging.Logger): Logger.
        workspace_id (int): Workspace à charger.
        label_cfg (dict): Entrée de config du label.
        export_path (str): Dossier de destination. Défaut : prepared_data_path/<label>.
        environment (str): "PROD" ou "QUAL".

    Returns:
        pd.DataFrame: Métadonnées des documents effectivement téléchargés.
    """
    label = label_cfg["label"]

    try:
        logger.info(f"***** Début du chargement - workspace {workspace_id} / label {label} ...")

        if export_path is None:
            export_path = os.path.join(config["DATA"]["prepared_data_path"], label)
            logger.info(f"Export path non spécifié, utilisation du chemin par défaut : {export_path}")

        os.makedirs(export_path, exist_ok=True)

        target_status = label_cfg.get(
            "target_status", config["DATA"].get("default_target_status", ["validated"])
        )
        sampling_cfg = config["DATA"].get("sampling", {})

        minio_client = initialize_cos_client(workspace_id=workspace_id, environment=environment)

        df_doc_infos = list_documents_from_minio(minio_client, logger, target_status=target_status)
        if df_doc_infos.empty:
            return pd.DataFrame()

        df_sampled = sample_train_test(df_doc_infos, label_cfg, sampling_cfg, logger)
        if df_sampled.empty:
            logger.warning(f"--- [{label}] aucun document retenu après échantillonnage.")
            return pd.DataFrame()

        download_dataset(minio_client, df_sampled, export_path, logger)

        df_sampled["label"] = label
        df_sampled["document_type"] = label
        df_sampled["workspace_id"] = workspace_id
        df_sampled["alto_path"] = df_sampled["id"].apply(lambda id_: get_alto_path(id_, export_path))

        logger.info(f"***** Fin du chargement - workspace {workspace_id} / label {label} ...")
        return df_sampled

    except Exception as e:
        logger.error(f"Erreur sur le workspace {workspace_id} / label {label} : {e}")
        logger.debug(f"Erreur sur le workspace {workspace_id} : {e}", stack_info=True, exc_info=True)
        return pd.DataFrame()


# =============================================================================
# Orchestrateur
# =============================================================================
def load_data_general(config: dict, logger: logging.Logger, environment: str = None) -> pd.DataFrame:
    """
    Orchestre le chargement de tous les labels actifs déclarés en config.

    - Un seul appel MinIO par workspace_id unique.
    - Les workspaces multi-labels (ex : workspace explo 77) sont ignorés pour le moment.
    - Les altos sont stockés dans un dossier dédié par typologie de document.
    - Un unique metadata.pkl récapitule l'ensemble des documents téléchargés.

    Args:
        config (dict): Configuration.
        logger (logging.Logger): Logger.
        environment (str): "PROD" ou "QUAL".

    Returns:
        pd.DataFrame: Metadata cumulé de tous les labels chargés.
    """
    logger.info("***** Début du chargement général des données ...")

    entries_by_wks = defaultdict(list)
    for entry in config["DATA"]["labels"]:
        if not entry.get("enabled", True):
            logger.info(f"--- Label '{entry['label']}' désactivé en config, ignoré.")
            continue
        entries_by_wks[entry["workspaceId"]].append(entry)

    all_dfs = []
    for wks_id, entries in entries_by_wks.items():
        if len(entries) > 1:
            labels = [e["label"] for e in entries]
            logger.info(f"--- Workspace {wks_id} multi-labels {labels} : déprioritisé pour le moment, ignoré.")
            continue

        label_cfg = entries[0]
        df = load_data(
            config=config,
            logger=logger,
            workspace_id=wks_id,
            label_cfg=label_cfg,
            environment=environment,
        )
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        logger.error("***** Aucun document chargé, metadata non sauvegardé.")
        return pd.DataFrame()

    df_all = pd.concat(all_dfs, ignore_index=True)
    save_metadata(df_all, config, logger)

    logger.info("***** Récapitulatif du chargement :")
    recap = df_all.groupby(["document_type", "set"]).size().unstack(fill_value=0)
    for line in recap.to_string().split("\n"):
        logger.info(f"    {line}")
    logger.info("***** Fin du chargement général des données ...")

    return df_all
