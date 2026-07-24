import copy
import re
import datetime

# =============================================================================
# Helpers check digit ICAO 9303 (identiques passeport / CNI nouveau format)
# =============================================================================

_WEIGHTS = [7, 3, 1]


def _char_value(c: str) -> int:
    if c == "<":
        return 0
    if c.isdigit():
        return int(c)
    if c.isalpha():
        return ord(c.upper()) - ord("A") + 10
    raise ValueError(f"Caractère MRZ invalide : {c!r}")


def compute_check_digit(data: str) -> int:
    total = 0
    for i, c in enumerate(data):
        total += _char_value(c) * _WEIGHTS[i % 3]
    return total % 10


def check_digit_is_valid(data: str, check_char: str) -> bool:
    if not check_char.isdigit():
        return data.strip("<") == ""
    return compute_check_digit(data) == int(check_char)


# =============================================================================
# Parsing et validation MRZ ancienne CNI française (format propriétaire)
# =============================================================================
#
# Format NON standard (pas ICAO TD2) — 2 lignes x 36 caractères.
#
# Ligne 1 (0-indexed) — aucun check digit :
#   [0:2]   type doc ("ID")
#   [2:5]   pays ("FRA")
#   [5:30]  nom de famille (25 chars, padding "<")
#   [30:36] département + code administratif (6 chars)
#
# Ligne 2 (0-indexed) — 3 check digits :
#   [0:4]   date d'émission AAMM (année + mois, pas jour)
#   [4:6]   département (2 chars)
#   [6:12]  code administratif (6 chars)
#   [12]    clé de contrôle de line2[0:12]
#   [13:27] prénom(s) (14 chars, séparés par "<<")
#   [27:33] date de naissance YYMMDD (6 chars)
#   [33]    clé de contrôle de la date de naissance
#   [34]    sexe ("M" ou "F")
#   [35]    clé de contrôle GLOBALE (couvre données des deux lignes)
#
# Composite (clé globale) :
#   line2[0:13] + line1[5:30] + line2[27:35]
#   = (date émission + dept + code admin + clé1) + nom + (DDN + clé DDN + sexe)


def parse_line1_cni_old(line1: str) -> dict:
    """Ligne 1 : extraction directe, aucun check digit."""
    line1 = line1.upper().ljust(36, "<")[:36]
    surname_raw = line1[5:30]
    surname = surname_raw.replace("<", " ").strip()
    return {
        "surname": surname,
        "surname_field": surname_raw,   # gardé pour heuristique troncature
    }


def parse_and_validate_line2_cni_old(line1: str, line2: str) -> tuple[dict, list]:
    """Ligne 2 : extraction + validation des 3 check digits.
    line1 est nécessaire pour le check digit global (couvre les deux lignes)."""
    line1 = line1.upper().ljust(36, "<")[:36]
    line2 = line2.upper().ljust(36, "<")[:36]
    errors = []

    issue_date = line2[0:4]     # AAMM (pas de jour)
    birth_date = line2[27:33]   # YYMMDD
    birth_date_check = line2[33]
    sex = line2[34]
    composite_check = line2[35]

    # Clé intermédiaire : line2[0:12]
    if not check_digit_is_valid(line2[0:12], line2[12]):
        errors.append("code_admin")

    # Clé date de naissance
    if not check_digit_is_valid(birth_date, birth_date_check):
        errors.append("birth_date")

    # Clé globale : line2[0:13] + line1[5:30] + line2[27:35]
    composite_data = line2[0:13] + line1[5:30] + line2[27:35]
    if not check_digit_is_valid(composite_data, composite_check):
        errors.append("composite")

    fields = {
        "issue_date_raw": issue_date,   # AAMM
        "birth_date_raw": birth_date,   # YYMMDD
        "sex": sex,
        "firstname_field": line2[13:27],  # 14 chars bruts, gardé pour parsing prénom
    }
    return fields, errors


# =============================================================================
# Parsing du prénom (ligne 2, positions fixes)
# =============================================================================

def parse_firstname_cni_old(firstname_field: str) -> dict:
    """Le prénom est sur 14 caractères fixes (line2[13:27]).
    Troncature détectée si le dernier caractère n'est pas '<' (zone pleine, pas de padding).
    Prénoms multiples séparés par '<<' ou '<'."""
    given_names_list = [p for p in re.split("<+", firstname_field) if p]
    given_names = " ".join(given_names_list)
    is_truncated = not firstname_field.endswith("<")
    return {
        "given_names": given_names,
        "is_truncated": is_truncated,
    }


# =============================================================================
# Fonctions d'extraction publiques
# =============================================================================

def extract_mrz_cni_old_fields(mrz1_value: str, mrz2_value: str) -> dict | None:
    """Retourne les champs validés si TOUS les check digits sont corrects, sinon None."""
    if not mrz1_value or len(mrz1_value.strip()) != 36:
        return None
    if not mrz2_value or len(mrz2_value.strip()) != 36:
        return None

    line1 = mrz1_value.strip().upper()
    line2 = mrz2_value.strip().upper()

    fields2, errors = parse_and_validate_line2_cni_old(line1, line2)
    return fields2 if not errors else None


def extract_mrz_cni_old_names(mrz1_value: str, mrz2_fields: dict) -> dict | None:
    """Retourne nom/prénom depuis les deux lignes.
    Surname (ligne 1, 25 chars) : toujours extrait si non vide.
    FirstName (ligne 2, 14 chars) : extrait seulement si pas tronqué (padding '<' final)."""
    result = {}

    if mrz1_value and len(mrz1_value.strip()) == 36:
        fields1 = parse_line1_cni_old(mrz1_value.strip().upper())
        if fields1["surname"]:
            result["surname"] = fields1["surname"]
            result["surname_truncated"] = not fields1["surname_field"].endswith("<")

    if mrz2_fields and "firstname_field" in mrz2_fields:
        fn = parse_firstname_cni_old(mrz2_fields["firstname_field"])
        if fn["given_names"] and not fn["is_truncated"]:
            result["given_names"] = fn["given_names"]

    return result or None


# =============================================================================
# Helpers date
# =============================================================================

def mrz_date_to_ddmmyyyy(yymmdd: str, is_birth_date: bool) -> str | None:
    if not yymmdd.isdigit() or len(yymmdd) != 6:
        return None
    yy, mm, dd = int(yymmdd[0:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
    current_yy = datetime.date.today().year % 100
    century = (1900 if yy > current_yy else 2000) if is_birth_date else 2000
    try:
        return datetime.date(century + yy, mm, dd).strftime("%d/%m/%Y")
    except ValueError:
        return None


def compute_validity_date(issue_date_aamm: str) -> str | None:
    """Calcule la date de validité depuis la date d'émission AAMM.
    Règle CNI ancienne génération :
      - émise entre 2004 et 2013 inclus : valide 15 ans (extension automatique)
      - émise avant 2004 ou après 2013  : valide 10 ans"""
    if not issue_date_aamm or len(issue_date_aamm) != 4 or not issue_date_aamm.isdigit():
        return None
    yy, mm = int(issue_date_aamm[0:2]), int(issue_date_aamm[2:4])
    current_yy = datetime.date.today().year % 100
    century = 1900 if yy > current_yy else 2000
    issue_year = century + yy
    years_valid = 15 if 2004 <= issue_year <= 2013 else 10
    try:
        d = datetime.date(issue_year, mm, 1)
        # expiration = même jour, même mois, + N années
        expiry = d.replace(year=d.year + years_valid)
        return expiry.strftime("%d/%m/%Y")
    except ValueError:
        return None


# =============================================================================
# Resolver + helper tag
# =============================================================================

MRZ_DERIVED_KEYS = ["rank", "empty", "groups", "suggestions", "occurences"]


def _make_mrz_tag(tag_type, value, source_tag):
    new_tag = {"type": tag_type, "value": value}
    for k in MRZ_DERIVED_KEYS:
        if k in source_tag:
            new_tag[k] = source_tag.get(k)
    new_tag["empty"] = False
    return new_tag


def _build_mrz_field_resolver_cni_old(mrz_fields, mrz_names, mrz1_raw_tag, mrz2_raw_tag):
    """Retourne un dict {datapoint_type: (valeur, tag_source)}."""
    resolved = {}

    if mrz_fields is not None:
        resolved["Sex"] = (mrz_fields["sex"], mrz2_raw_tag)

        raw_birth = mrz_date_to_ddmmyyyy(mrz_fields["birth_date_raw"], is_birth_date=True)
        if raw_birth:
            resolved["DateOfBirth"] = (postprocess_date(raw_birth), mrz2_raw_tag)

        validity = compute_validity_date(mrz_fields["issue_date_raw"])
        if validity:
            resolved["ValidityDate"] = (postprocess_date(validity), mrz2_raw_tag)

    if mrz_names is not None:
        if "surname" in mrz_names and not mrz_names.get("surname_truncated"):
            resolved["Surname"] = (mrz_names["surname"], mrz1_raw_tag)
        if "given_names" in mrz_names:
            resolved["FirstName"] = (mrz_names["given_names"], mrz2_raw_tag)

    return {k: v for k, v in resolved.items() if v[0]}


# =============================================================================
# postprocess_all CNI ancien format
# =============================================================================

def postprocess_all(tags: dict, alto):
    """
    Post-processes all tags CNI ancien format (format propriétaire français, 2 x 36 chars).

    MRZ ancien format — champs extraits si MRZ validée :
      DateOfBirth  : line2[27:33] YYMMDD, validé par clé [33]
      ValidityDate : calculé depuis date d'émission line2[0:4] + 10 ou 15 ans
      Sex          : line2[34]
      Surname      : line1[5:30] (25 chars, pas de check digit)
      FirstName    : line2[13:27] (14 chars, seulement si pas tronqué)

    Note : DocumentNumber non extrait — le numéro CNI 12 chars est encodé de façon
    complexe sur les codes admin des deux lignes, pas exploitable directement.

    Args:
        tags (dict): The tags to post-process.

    Returns:
        list: A list of post-processed tags.
    """
    try:
        processing_functions = {
            "Surname": post_process_name,
            "FirstName": post_process_firstname,
            "Sex": postprocess_sex,
            "Nationality": post_process_nationality,
            "DateOfBirth": post_process_dateofbirth,
            "PlaceOfBirth": post_process_placeofbirth,
            "UsageName": post_process_name,
            "DocumentNumber": post_process_document_number,
            "ValidityDate": post_process_validitydate,
            "MrzStrip1": post_process_mrz1,
            "MrzStrip2": post_process_mrz2,
        }

        tags_list = copy.deepcopy(tags)
        tags_pp = []

        # Address
        address_preds = [tag for tag in tags if tag["type"] == 'Address_anno']
        address_pp_dict = post_process_address(address_preds, alto)
        tags_pp.append(address_pp_dict)

        tags_set = remove_duplicated_datapoint(tags_list, exclude_datapoints=["Address_anno"])

        # --- Extraction MRZ ---
        mrz1_raw_tag = next((t for t in tags_set if t['type'] == "MrzStrip1_anno"), None)
        mrz2_raw_tag = next((t for t in tags_set if t['type'] == "MrzStrip2_anno"), None)

        mrz_fields = None
        if mrz1_raw_tag is not None and mrz2_raw_tag is not None:
            mrz1_cleaned = post_process_mrz1(mrz1_raw_tag['value'])
            mrz2_cleaned = post_process_mrz2(mrz2_raw_tag['value'])
            mrz_fields = extract_mrz_cni_old_fields(mrz1_cleaned, mrz2_cleaned)

        mrz_names = None
        if mrz1_raw_tag is not None:
            mrz1_cleaned = post_process_mrz1(mrz1_raw_tag['value'])
            mrz_names = extract_mrz_cni_old_names(mrz1_cleaned, mrz_fields)

        mrz_resolved = _build_mrz_field_resolver_cni_old(
            mrz_fields, mrz_names,
            mrz1_raw_tag, mrz2_raw_tag
        )

        # --- Boucle principale ---
        for tag in tags_set:
            if tag['type'] != "Address_anno":
                tag['type'] = tag['type'].replace('_anno', '')

                if tag['type'] in mrz_resolved:
                    tag['value'] = mrz_resolved[tag['type']][0]
                    tag['empty'] = False

                elif tag['type'] in processing_functions:
                    if tag['type'] == "FirstName":
                        tag['value'] = processing_functions[tag['type']](tag, alto)
                    else:
                        tag['value'] = processing_functions[tag['type']](tag['value'])

                tags_pp.append(tag)

        # --- Rattrapage : datapoints absents des prédictions mais validés en MRZ ---
        existing_types = {tag_pp['type'] for tag_pp in tags_pp}
        for mrz_type, (mrz_value, source_tag) in mrz_resolved.items():
            if mrz_type not in existing_types:
                tags_pp.append(_make_mrz_tag(mrz_type, mrz_value, source_tag))
                existing_types.add(mrz_type)

        if "Sex" not in existing_types:
            tags_pp.append({
                "type": "Sex",
                "value": postprocess_sex("", tags_set),
                "rank": None,
                "empty": True,
                "groups": [],
                "suggestions": [],
                "occurences": 0,
            })

        return tags_pp

    except Exception as e:
        logger.error(e, exc_info=True)
