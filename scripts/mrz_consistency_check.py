"""
Détection de décalages MRZ vs valeurs prédites sur la carte.
Consomme un DataFrame dont les colonnes sont nommées avec suffixe _anno.
Retourne les champs en désaccord pour chaque document -> signal de fraude potentielle.

Formats supportés :
  - CNI nouveau format (TD1, 3 x 30 chars)
  - CNI ancien format (propriétaire français, 2 x 36 chars)
"""

import re
import unicodedata
import pandas as pd

from mrz_cni_validator import (
    parse_and_validate_line1_cni as _parse_cni_new_line1,
    parse_and_validate_line2_cni as _parse_cni_new_line2,
    parse_line3_cni as _parse_cni_new_line3,
)
from postprocess_cni_old_with_mrz import (
    parse_line1_cni_old as _parse_cni_old_line1,
    parse_and_validate_line2_cni_old as _parse_cni_old_line2,
    parse_firstname_cni_old as _parse_cni_old_firstname,
    mrz_date_to_ddmmyyyy,
    compute_validity_date,
)


# =============================================================================
# Helpers de normalisation et lecture DataFrame
# =============================================================================

def _normalize_str(s: str) -> str:
    """Majuscules + suppression des accents (ICAO ne les encode pas)."""
    if not s or not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.upper().strip()


def _normalize_date(d: str) -> str:
    """Ramène DD/MM/YY, DD/MM/YYYY et YYYY-MM-DD au format DD/MM/YYYY."""
    if not d or not isinstance(d, str):
        return ""
    d = d.strip()
    m = re.fullmatch(r"(\d{2})/(\d{2})/(\d{2})", d)
    if m:
        dd, mm, yy = m.groups()
        century = "19" if int(yy) > int(str(pd.Timestamp.now().year)[2:]) else "20"
        return f"{dd}/{mm}/{century}{yy}"
    if re.fullmatch(r"\d{2}/\d{2}/\d{4}", d):
        return d
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", d)
    if m:
        yyyy, mm, dd = m.groups()
        return f"{dd}/{mm}/{yyyy}"
    return d


def _get(row, col: str) -> str:
    val = row.get(col, "")
    if pd.isna(val):
        return ""
    return str(val).strip()


def _compare(field: str, mrz_val: str, card_val: str, discrepancies: dict):
    """Flag uniquement si les deux valeurs sont présentes et divergent."""
    if not mrz_val or not card_val:
        return
    if mrz_val != card_val:
        discrepancies[field] = {"mrz": mrz_val, "card": card_val}


# =============================================================================
# CNI nouveau format (TD1 - 3 x 30)
# =============================================================================

def check_cni_new_consistency(row) -> dict:
    """
    Détecte les décalages MRZ / valeurs carte sur une CNI nouveau format (TD1).

    Args:
        row: ligne d'un DataFrame avec colonnes _anno.

    Returns:
        dict {champ: {"mrz": val_mrz, "card": val_card}} pour chaque désaccord.
        Dict vide = pas de décalage détecté.
    """
    discrepancies = {}
    mrz1 = _get(row, "MrzStrip1_anno")
    mrz2 = _get(row, "MrzStrip2_anno")
    mrz3 = _get(row, "MrzStrip3_anno")

    if not mrz1 or not mrz2:
        return discrepancies

    l1_fields, _ = _parse_cni_new_line1(mrz1)
    l2_fields, _ = _parse_cni_new_line2(mrz1, mrz2)

    _compare(
        "DocumentNumber",
        l1_fields["document_number"],
        _normalize_str(_get(row, "DocumentNumber_anno")),
        discrepancies,
    )
    mrz_dob = mrz_date_to_ddmmyyyy(l2_fields["birth_date_raw"], is_birth_date=True)
    _compare(
        "DateOfBirth",
        _normalize_date(mrz_dob or ""),
        _normalize_date(_get(row, "DateOfBirth_anno")),
        discrepancies,
    )
    mrz_exp = mrz_date_to_ddmmyyyy(l2_fields["expiry_date_raw"], is_birth_date=False)
    _compare(
        "ValidityDate",
        _normalize_date(mrz_exp or ""),
        _normalize_date(_get(row, "ValidityDate_anno")),
        discrepancies,
    )
    _compare(
        "Sex",
        l2_fields["sex"],
        _normalize_str(_get(row, "Sex_anno")),
        discrepancies,
    )

    if mrz3:
        l3 = _parse_cni_new_line3(mrz3)
        _compare(
            "Surname",
            _normalize_str(l3["surname"]),
            _normalize_str(_get(row, "Surname_anno")),
            discrepancies,
        )
        _compare(
            "FirstName",
            _normalize_str(l3["given_names"]),
            _normalize_str(_get(row, "FirstName_anno")),
            discrepancies,
        )

    return discrepancies


# =============================================================================
# CNI ancien format (propriétaire français - 2 x 36)
# =============================================================================

def check_cni_old_consistency(row) -> dict:
    """
    Détecte les décalages MRZ / valeurs carte sur une CNI ancien format.

    Args:
        row: ligne d'un DataFrame avec colonnes _anno.

    Returns:
        dict {champ: {"mrz": val_mrz, "card": val_card}} pour chaque désaccord.
        Dict vide = pas de décalage détecté.
    """
    discrepancies = {}
    mrz1 = _get(row, "MrzStrip1_anno")
    mrz2 = _get(row, "MrzStrip2_anno")

    if not mrz1 or not mrz2:
        return discrepancies

    l1_fields = _parse_cni_old_line1(mrz1)
    l2_fields, _ = _parse_cni_old_line2(mrz1, mrz2)

    _compare(
        "Surname",
        _normalize_str(l1_fields["surname"]),
        _normalize_str(_get(row, "Surname_anno")),
        discrepancies,
    )
    fn = _parse_cni_old_firstname(l2_fields.get("firstname_field", ""))
    _compare(
        "FirstName",
        _normalize_str(fn["given_names"]),
        _normalize_str(_get(row, "FirstName_anno")),
        discrepancies,
    )
    mrz_dob = mrz_date_to_ddmmyyyy(l2_fields["birth_date_raw"], is_birth_date=True)
    _compare(
        "DateOfBirth",
        _normalize_date(mrz_dob or ""),
        _normalize_date(_get(row, "DateOfBirth_anno")),
        discrepancies,
    )
    validity = compute_validity_date(l2_fields["issue_date_raw"])
    _compare(
        "ValidityDate",
        _normalize_date(validity or ""),
        _normalize_date(_get(row, "ValidityDate_anno")),
        discrepancies,
    )
    _compare(
        "Sex",
        l2_fields["sex"],
        _normalize_str(_get(row, "Sex_anno")),
        discrepancies,
    )

    return discrepancies


# =============================================================================
# Application sur DataFrame
# =============================================================================

def apply_mrz_consistency_check(df: pd.DataFrame, doc_type: str) -> pd.DataFrame:
    """
    Applique la vérification de cohérence MRZ sur tout le DataFrame.

    Args:
        df       : DataFrame avec colonnes _anno.
        doc_type : "cni_new" ou "cni_old".

    Returns:
        df enrichi de deux colonnes :
          mrz_discrepancies : dict des champs en désaccord (vide = OK)
          mrz_fraud_flag    : bool, True si au moins un désaccord détecté
    """
    checkers = {
        "cni_new": check_cni_new_consistency,
        "cni_old": check_cni_old_consistency,
    }

    if doc_type not in checkers:
        raise ValueError(f"doc_type inconnu : {doc_type!r}. Valeurs acceptées : {list(checkers)}")

    checker = checkers[doc_type]
    df = df.copy()
    df["mrz_discrepancies"] = df.apply(checker, axis=1)
    df["mrz_fraud_flag"] = df["mrz_discrepancies"].apply(bool)
    return df
