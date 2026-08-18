"""
Détection de décalages MRZ vs valeurs prédites sur la carte.
Consomme un DataFrame dont les colonnes sont nommées avec suffixe _anno.
Retourne les champs en désaccord pour chaque document -> signal de fraude potentielle.

Formats supportés :
  - CNI nouveau format (TD1, 3 x 30 chars)
  - CNI ancien format (propriétaire français, 2 x 36 chars)

Script indépendant, sans validation des check digits MRZ.

=== Champs comparables par format ===

CNI nouveau (TD1) — ligne 1 : DocumentNumber
                  — ligne 2 : DateOfBirth, Sex, ValidityDate
                  — ligne 3 : Surname, FirstName

CNI ancien        — ligne 1 : Surname
                  — ligne 2 : FirstName, DateOfBirth, Sex, IssuanceDate
                  NB: ValidityDate absente de la MRZ ancienne CNI (Wikipedia/ICAO confirmé)
                      Nationality absente de la ligne 2
"""

import re
import unicodedata
import datetime
import pandas as pd


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
        century = "19" if int(yy) > int(str(datetime.date.today().year)[2:]) else "20"
        return f"{dd}/{mm}/{century}{yy}"
    if re.fullmatch(r"\d{2}/\d{2}/\d{4}", d):
        return d
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", d)
    if m:
        yyyy, mm, dd = m.groups()
        return f"{dd}/{mm}/{yyyy}"
    return d


def _yymmdd_to_ddmmyyyy(yymmdd: str, is_birth_date: bool) -> str:
    """Convertit une date MRZ YYMMDD en DD/MM/YYYY avec résolution du siècle."""
    if not yymmdd or not yymmdd.isdigit() or len(yymmdd) != 6:
        return ""
    yy, mm, dd = int(yymmdd[0:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
    current_yy = datetime.date.today().year % 100
    century = (1900 if yy > current_yy else 2000) if is_birth_date else 2000
    try:
        return datetime.date(century + yy, mm, dd).strftime("%d/%m/%Y")
    except ValueError:
        return ""


def _aamm_to_mmyyyy(aamm: str) -> str:
    """Convertit une date d'émission MRZ AAMM en MM/YYYY pour comparaison.
    La date d'émission n'a pas de jour dans la MRZ -> on compare sur MM/YYYY."""
    if not aamm or not aamm.isdigit() or len(aamm) != 4:
        return ""
    yy, mm = int(aamm[0:2]), int(aamm[2:4])
    current_yy = datetime.date.today().year % 100
    century = 1900 if yy > current_yy else 2000
    try:
        return datetime.date(century + yy, mm, 1).strftime("%m/%Y")
    except ValueError:
        return ""


def _normalize_issuance_date(d: str) -> str:
    """Extrait MM/YYYY depuis une date d'émission carte (DD/MM/YYYY ou MM/YYYY)."""
    if not d or not isinstance(d, str):
        return ""
    d = d.strip()
    # DD/MM/YYYY -> MM/YYYY
    m = re.fullmatch(r"\d{2}/(\d{2})/(\d{4})", d)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    # MM/YYYY direct
    if re.fullmatch(r"\d{2}/\d{4}", d):
        return d
    return d


def _parse_names(names_field: str) -> tuple[str, str]:
    """Parse un champ MRZ 'NOM<<PRENOM1<PRENOM2' -> (nom, prenoms)."""
    surname_part, _, given_part = names_field.partition("<<")
    surname = surname_part.replace("<", " ").strip()
    given_names = " ".join(p for p in re.split("<+", given_part) if p)
    return surname, given_names


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
# CNI nouveau format (TD1 - 3 x 30 chars)
#
# Ligne 1 : [0:2] type | [2:5] pays | [5:14] n° doc (9) | [14] clé | [15:30] opt
# Ligne 2 : [0:6] DDN  | [6] clé    | [7] sexe          | [8:14] expiration
#           [14] clé   | [15:29] opt | [29] clé composite
# Ligne 3 : NOM<<PRENOM (30 chars, pas de clé)
# =============================================================================

def check_cni_new_consistency(row) -> dict:
    """
    Détecte les décalages MRZ / valeurs carte sur une CNI nouveau format (TD1).
    Champs comparés : DocumentNumber, DateOfBirth, Sex, ValidityDate, Surname, FirstName.
    """
    discrepancies = {}
    mrz1 = _get(row, "MrzStrip1_anno").upper().strip()
    mrz2 = _get(row, "MrzStrip2_anno").upper().strip()
    mrz3 = _get(row, "MrzStrip3_anno").upper().strip()

    if len(mrz1) != 30 or len(mrz2) != 30:
        return discrepancies

    # Ligne 1 — numéro de document
    _compare(
        "DocumentNumber",
        mrz1[5:14].replace("<", ""),
        _normalize_str(_get(row, "DocumentNumber_anno")),
        discrepancies,
    )

    # Ligne 2 — date de naissance, sexe, date de validité
    _compare(
        "DateOfBirth",
        _normalize_date(_yymmdd_to_ddmmyyyy(mrz2[0:6], is_birth_date=True)),
        _normalize_date(_get(row, "DateOfBirth_anno")),
        discrepancies,
    )
    _compare(
        "Sex",
        mrz2[7],
        _normalize_str(_get(row, "Sex_anno")),
        discrepancies,
    )
    _compare(
        "ValidityDate",
        _normalize_date(_yymmdd_to_ddmmyyyy(mrz2[8:14], is_birth_date=False)),
        _normalize_date(_get(row, "ValidityDate_anno")),
        discrepancies,
    )

    # Ligne 3 — nom et prénom
    if len(mrz3) == 30:
        surname, given_names = _parse_names(mrz3)
        _compare(
            "Surname",
            _normalize_str(surname),
            _normalize_str(_get(row, "Surname_anno")),
            discrepancies,
        )
        _compare(
            "FirstName",
            _normalize_str(given_names),
            _normalize_str(_get(row, "FirstName_anno")),
            discrepancies,
        )

    return discrepancies


# =============================================================================
# CNI ancien format (propriétaire français - 2 x 36 chars)
#
# Ligne 1 : [0:2] type | [2:5] pays | [5:30] nom (25) | [30:36] dept+code admin
# Ligne 2 : [0:4]  date émission AAMM
#           [4:6]  département
#           [6:12] code administratif
#           [12]   clé de contrôle de line2[0:12]
#           [13:27] prénom(s) (14 chars fixes)
#           [27:33] date de naissance YYMMDD
#           [33]   clé de contrôle DDN
#           [34]   sexe ("M" ou "F")
#           [35]   clé de contrôle globale
#
# NB : PAS de date de validité dans la MRZ (Wikipedia/ICAO confirmé).
#      PAS de nationalité dans la ligne 2.
#      Date d'émission présente ([0:4] AAMM) -> comparable si colonne dispo.
# =============================================================================

def check_cni_old_consistency(row) -> dict:
    """
    Détecte les décalages MRZ / valeurs carte sur une CNI ancien format.
    Champs comparés : Surname, FirstName, DateOfBirth, Sex, IssuanceDate (si dispo).
    """
    discrepancies = {}
    mrz1 = _get(row, "MrzStrip1_anno").upper().strip()
    mrz2 = _get(row, "MrzStrip2_anno").upper().strip()

    if len(mrz1) != 36 or len(mrz2) != 36:
        return discrepancies

    # Ligne 1 — nom (25 chars, [5:30])
    surname = mrz1[5:30].replace("<", " ").strip()
    _compare(
        "Surname",
        _normalize_str(surname),
        _normalize_str(_get(row, "Surname_anno")),
        discrepancies,
    )

    # Ligne 2 — date d'émission AAMM ([0:4])
    # Comparaison sur MM/YYYY car la MRZ ne contient pas le jour
    mrz_issuance = _aamm_to_mmyyyy(mrz2[0:4])
    card_issuance = _normalize_issuance_date(_get(row, "IssuanceDate_anno"))
    _compare("IssuanceDate", mrz_issuance, card_issuance, discrepancies)

    # Ligne 2 — prénom(s) ([13:27], 14 chars fixes)
    given_names = " ".join(p for p in re.split("<+", mrz2[13:27]) if p)
    _compare(
        "FirstName",
        _normalize_str(given_names),
        _normalize_str(_get(row, "FirstName_anno")),
        discrepancies,
    )

    # Ligne 2 — date de naissance YYMMDD ([27:33])
    _compare(
        "DateOfBirth",
        _normalize_date(_yymmdd_to_ddmmyyyy(mrz2[27:33], is_birth_date=True)),
        _normalize_date(_get(row, "DateOfBirth_anno")),
        discrepancies,
    )

    # Ligne 2 — sexe ([34])
    _compare(
        "Sex",
        mrz2[34],
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

    df = df.copy()
    df["mrz_discrepancies"] = df.apply(checkers[doc_type], axis=1)
    df["mrz_fraud_flag"] = df["mrz_discrepancies"].apply(bool)
    return df
