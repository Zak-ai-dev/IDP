import copy
import re
import datetime

# =============================================================================
# Helpers check digit ICAO 9303 (identiques passeport)
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
# Parsing et validation MRZ TD1 (3 lignes x 30 caractères)
# =============================================================================

def parse_and_validate_line1_cni(line1: str) -> tuple[dict, list]:
    line1 = line1.upper().ljust(30, "<")[:30]
    errors = []

    doc_number = line1[5:14]
    doc_number_check = line1[14]

    if not check_digit_is_valid(doc_number, doc_number_check):
        errors.append("doc_number")

    fields = {
        "document_number": doc_number.replace("<", ""),
    }
    return fields, errors


def parse_and_validate_line2_cni(line1: str, line2: str) -> tuple[dict, list]:
    """Le check composite TD1 couvre les deux lignes, donc line1 est requis."""
    line1 = line1.upper().ljust(30, "<")[:30]
    line2 = line2.upper().ljust(30, "<")[:30]
    errors = []

    birth_date = line2[0:6]
    birth_date_check = line2[6]
    sex = line2[7]
    expiry_date = line2[8:14]
    expiry_date_check = line2[14]
    composite_check = line2[29]

    if not check_digit_is_valid(birth_date, birth_date_check):
        errors.append("birth_date")

    if not check_digit_is_valid(expiry_date, expiry_date_check):
        errors.append("expiry_date")

    # Composite TD1 : line1[5:30] + line2[0:7] + line2[8:15] + line2[15:29]
    composite_data = (
        line1[5:30] +
        line2[0:7] +
        line2[8:15] +
        line2[15:29]
    )
    if not check_digit_is_valid(composite_data, composite_check):
        errors.append("composite")

    fields = {
        "birth_date_raw": birth_date,
        "sex": sex,
        "expiry_date_raw": expiry_date,
    }
    return fields, errors


def parse_line3_cni(line3: str) -> dict:
    """Ligne 3 : noms uniquement, pas de check digit.
    Même logique de troncature que ligne 1 TD3 passeport."""
    line3 = line3.upper().ljust(30, "<")[:30]
    surname_part, _, given_names_part = line3.partition("<<")
    surname = surname_part.replace("<", " ").strip()
    given_names_list = [p for p in re.split("<+", given_names_part) if p]
    return {
        "surname": surname,
        "given_names": " ".join(given_names_list),
        "names_field": line3,
    }


# =============================================================================
# Fonctions d'extraction publiques
# =============================================================================

def extract_mrz_cni_line12_fields(mrz1_value: str, mrz2_value: str) -> dict | None:
    """Retourne les champs lignes 1+2 si TOUS les check digits sont valides, sinon None."""
    if not mrz1_value or len(mrz1_value.strip()) != 30:
        return None
    if not mrz2_value or len(mrz2_value.strip()) != 30:
        return None

    line1 = mrz1_value.strip().upper()
    line2 = mrz2_value.strip().upper()

    fields1, errors1 = parse_and_validate_line1_cni(line1)
    fields2, errors2 = parse_and_validate_line2_cni(line1, line2)

    return {**fields1, **fields2} if not errors1 + errors2 else None


def extract_mrz_cni_names(mrz3_value: str) -> dict | None:
    """Retourne nom/prénom depuis la ligne 3.
    Surname : toujours extrait si non vide.
    Given names : seulement si pas de signe de troncature (padding '<' final présent)."""
    if not mrz3_value or len(mrz3_value.strip()) != 30:
        return None

    fields = parse_line3_cni(mrz3_value.strip().upper())
    result = {}

    if fields["surname"]:
        result["surname"] = fields["surname"]

    if fields["given_names"] and fields["names_field"].endswith("<"):
        result["given_names"] = fields["given_names"]

    return result or None


# =============================================================================
# Helpers date (identiques passeport)
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


# =============================================================================
# Resolver générique (même pattern que passeport)
# =============================================================================

MRZ_DERIVED_KEYS = ["rank", "empty", "groups", "suggestions", "occurences"]


def _make_mrz_tag(tag_type, value, source_tag):
    new_tag = {"type": tag_type, "value": value}
    for k in MRZ_DERIVED_KEYS:
        if k in source_tag:
            new_tag[k] = source_tag.get(k)
    new_tag["empty"] = False
    return new_tag


def _build_mrz_field_resolver_cni(mrz12_fields, mrz3_names, mrz1_raw_tag, mrz2_raw_tag, mrz3_raw_tag):
    """Retourne un dict {datapoint_type: (valeur, tag_source)} pour tous les champs
    extractibles depuis les 3 bandes MRZ CNI validées."""
    resolved = {}

    if mrz12_fields is not None:
        resolved["DocumentNumber"] = (mrz12_fields["document_number"], mrz1_raw_tag)
        resolved["Sex"] = (mrz12_fields["sex"], mrz2_raw_tag)

        raw_birth = mrz_date_to_ddmmyyyy(mrz12_fields["birth_date_raw"], is_birth_date=True)
        if raw_birth:
            resolved["DateOfBirth"] = (postprocess_date(raw_birth), mrz2_raw_tag)

        raw_expiry = mrz_date_to_ddmmyyyy(mrz12_fields["expiry_date_raw"], is_birth_date=False)
        if raw_expiry:
            resolved["ValidityDate"] = (postprocess_date(raw_expiry), mrz2_raw_tag)

    if mrz3_names is not None:
        if "surname" in mrz3_names:
            resolved["Surname"] = (mrz3_names["surname"], mrz3_raw_tag)
        if "given_names" in mrz3_names:
            resolved["FirstName"] = (mrz3_names["given_names"], mrz3_raw_tag)

    return {k: v for k, v in resolved.items() if v[0]}


# =============================================================================
# postprocess_all CNI
# =============================================================================

def postprocess_all(tags: dict, alto):
    """
    Post-processes all tags CNI en appliquant les fonctions de postprocessing classiques.
    Si les bandes MRZ sont validées (check digits ICAO 9303 TD1), leurs valeurs alimentent
    en priorité : DocumentNumber, DateOfBirth, Sex, ValidityDate, Surname, FirstName.
    Les datapoints non prédits par le modèle mais présents en MRZ sont rattrapés en fin.

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
            "MrzStrip3": post_process_mrz3,
        }

        tags_list = copy.deepcopy(tags)
        tags_pp = []

        # Address
        address_preds = [tag for tag in tags if tag["type"] == 'Address_anno']
        address_pp_dict = post_process_address(address_preds, alto)
        tags_pp.append(address_pp_dict)

        tags_set = remove_duplicated_datapoint(tags_list, exclude_datapoints=["Address_anno"])

        # --- Extraction MRZ (sur valeurs nettoyées, avant traitement classique) ---
        mrz1_raw_tag = next((t for t in tags_set if t['type'] == "MrzStrip1_anno"), None)
        mrz2_raw_tag = next((t for t in tags_set if t['type'] == "MrzStrip2_anno"), None)
        mrz3_raw_tag = next((t for t in tags_set if t['type'] == "MrzStrip3_anno"), None)

        mrz12_fields = None
        if mrz1_raw_tag is not None and mrz2_raw_tag is not None:
            mrz1_cleaned = post_process_mrz1(mrz1_raw_tag['value'])
            mrz2_cleaned = post_process_mrz2(mrz2_raw_tag['value'])
            mrz12_fields = extract_mrz_cni_line12_fields(mrz1_cleaned, mrz2_cleaned)

        mrz3_names = None
        if mrz3_raw_tag is not None:
            mrz3_cleaned = post_process_mrz3(mrz3_raw_tag['value'])
            mrz3_names = extract_mrz_cni_names(mrz3_cleaned)

        mrz_resolved = _build_mrz_field_resolver_cni(
            mrz12_fields, mrz3_names,
            mrz1_raw_tag, mrz2_raw_tag, mrz3_raw_tag
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
