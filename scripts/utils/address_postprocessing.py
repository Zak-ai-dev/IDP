import re
import copy
import numpy as np

# NOTE: `logger`, `tag_to_proper_noun` et `FRANCE_CITIES` sont supposés définis/importés ailleurs
# dans le module d'origine (inchangés, non fournis dans cet extrait).

# ---------------------------------------------------------------------------
# Regex précompilées
# ---------------------------------------------------------------------------
NON_ALPHANUM_PATTERN = re.compile(r'[^a-z0-9àâçéèêëîïôûùüÿñæœ\'\- ]')
MULTI_SPACE_PATTERN = re.compile(r'\s+')
CP_PATTERN = re.compile(r'\b\d{5}\b')
NUM_PATTERN = re.compile(r'^(\d+[a-z]?)\s+')
DIGITS_PATTERN = re.compile(r'\b\d+\b')

# Utilisées dans clean_address_text (comportement inchangé, y compris le `\d{}`
# d'origine qui matche littéralement "{}" et non un quantificateur — conservé
# à l'identique pour ne pas modifier le traitement en aval).
_LETTERS_DIGITS_PATTERN = re.compile(r'([a-z]{2,})(\d{})')
_DIGITS_LETTERS_PATTERN = re.compile(r'(\d)([a-z]{2,})')

# Utilisées dans la normalisation du segment avant/après le code postal
_EME_PATTERN = re.compile(r'(\d+) (EME)')
_ISOLATED_LETTER_PATTERN = re.compile(r'\b[a-z]\b')
_ISOLATED_LETTER_DIGIT_AFTER_CP_PATTERN = re.compile(r'\b(?![LlDd])[A-Za-z]\b|\b\d\b')

# Utilisées dans separate_rue
_RUE_BEFORE_PATTERN = re.compile(r'(\w)RUE')
_RUE_AFTER_PATTERN = re.compile(r'RUE(\w)')
_RUE_PATTERN = re.compile(r'RUE')


def clean_address_text(text: str) -> str:
    """Nettoie le texte : passage en minuscules, séparation lettres/chiffres,
    suppression des caractères non alphanumériques.

    Args:
        text (str): Le texte à nettoyer.

    Returns:
        str: Le texte nettoyé.
    """
    try:
        text = text.lower()
        text = _LETTERS_DIGITS_PATTERN.sub(r'\1 \2', text)
        text = _DIGITS_LETTERS_PATTERN.sub(r'\1 \2', text)
        text = NON_ALPHANUM_PATTERN.sub(' ', text)
        text = MULTI_SPACE_PATTERN.sub(' ', text).strip()
        return text
    except Exception as e:
        logger.debug(e, exc_info=True)
        return text


def remove_isolated_characters(string: str) -> str:
    """Supprime les caractères isolés en fin de chaîne.

    Args:
        string (str): La chaîne à traiter.

    Returns:
        str: La chaîne sans les caractères isolés en fin de chaîne.
    """
    try:
        while True:
            new_string = re.sub(r'\b([a-zA-Z0-9])\s*$|\s*$', '', string)
            if new_string == string:
                break
            string = new_string
        return string
    except Exception as e:
        logger.debug(e, exc_info=True)
        return string


def separate_rue(string: str) -> str:
    """Sépare le mot "RUE" du reste de l'adresse.

    Args:
        string (str): La chaîne à traiter.

    Returns:
        str: La chaîne avec "RUE" correctement séparé.
    """
    try:
        string = _RUE_BEFORE_PATTERN.sub(r'\1 RUE', string)
        string = _RUE_AFTER_PATTERN.sub(r'RUE \1', string)
        string = _RUE_PATTERN.sub(r' RUE ', string)
        string = MULTI_SPACE_PATTERN.sub(' ', string).strip()
        return string
    except Exception as e:
        logger.debug(e, exc_info=True)
        return string


def is_french_city(city: str) -> bool:
    """Vérifie si la ville est française.

    Args:
        city (str): La ville à vérifier.

    Returns:
        bool: True si la ville est française, False sinon.
    """
    try:
        return city.upper() in FRANCE_CITIES
    except Exception as e:
        logger.debug(e, exc_info=True)
        return city


def _address_from_parts(parts: dict) -> str:
    """Reconstruit une chaîne d'adresse à partir de ses composants.

    Args:
        parts (dict): Les composants de l'adresse.

    Returns:
        str: L'adresse complète.
    """
    try:
        pred_address = f'{parts["number"]} {parts["street"]} {parts["postal_code"]} {parts["city"]} {parts["country"]}'.strip()
        pred_address = MULTI_SPACE_PATTERN.sub(' ', pred_address)
        return pred_address
    except Exception as e:
        logger.debug(e, exc_info=True)


def _normalize_street_segment(before_cp: str) -> str:
    """Normalise le segment situé avant le code postal (numéro + rue) :
    fusionne les suffixes ordinaux (ex: "3 EME" -> "3EME"), supprime les lettres
    isolées, les caractères isolés en fin de chaîne, puis sépare le mot "RUE".

    Args:
        before_cp (str): Segment brut situé avant le code postal.

    Returns:
        str: Le segment normalisé.
    """
    try:
        before_cp = _EME_PATTERN.sub(r'\1\2', before_cp)
        before_cp = _ISOLATED_LETTER_PATTERN.sub('', before_cp)
        before_cp = remove_isolated_characters(before_cp)
        before_cp = separate_rue(before_cp)
        return before_cp
    except Exception as e:
        logger.debug(e, exc_info=True)
        return before_cp


def _normalize_city_segment(after_cp: str) -> str:
    """Normalise le segment situé après le code postal (ville + pays) :
    supprime les chiffres ainsi que les lettres/chiffres isolés.

    Args:
        after_cp (str): Segment brut situé après le code postal.

    Returns:
        str: Le segment normalisé.
    """
    try:
        after_cp = DIGITS_PATTERN.sub('', after_cp).strip()
        after_cp = _ISOLATED_LETTER_DIGIT_AFTER_CP_PATTERN.sub('', after_cp)
        return after_cp
    except Exception as e:
        logger.debug(e, exc_info=True)
        return after_cp


def post_process_address(address_preds: list, alto_json: dict) -> dict:
    """
    Reconstruit une adresse à partir des prédictions Address_anno (ordonnées
    spatialement via leur position Y dans la page), puis en extrait les
    composants structurés (number, street, postal_code, city, country).

    Args:
        address_preds (list): Liste des prédictions Address_anno.
        alto_json (dict): ALTO JSON contenant les coordonnées du texte OCR.

    Returns:
        dict: Le dictionnaire d'adresse post-traité.
    """
    try:
        if len(address_preds) == 0:
            return {
                "type": "Address",
                "value": tag_to_proper_noun(" ".join([occ["value"] for occ in address_preds])),
                "rank": [occ["rank"] for occ in address_preds],
                "empty": False,
                "groups": [],
                "suggestions": [],
                "occurences": [occ["occurences"] for occ in address_preds],
            }

        # --- 1. Reconstruction : ordonne les prédictions par position Y et les concatène ---
        predictions_with_y, ranks, occurences = [], [], []
        for pred in address_preds:
            pred_copy = copy.deepcopy(pred)
            occurrence = pred_copy["occurences"][0]
            text_position = occurrence["textPositions"][0]
            pred_copy["_y"] = alto_json["pages"][text_position["page"]]["texts"][text_position["text"]]["y"]
            predictions_with_y.append(pred_copy)
            ranks.append(pred_copy["rank"])
            occurences.append(occurrence)

        predictions_with_y.sort(key=lambda x: x["_y"])
        pred_address = " ".join(pred["value"].strip() for pred in predictions_with_y)

        # --- 2. Extraction des composants structurés ---
        # Ce bloc a son propre try/except (comme l'ancien "step1"), volontairement
        # distinct du try/except englobant : si le parsing échoue, on garde la
        # reconstruction spatiale (rank/occurences) déjà calculée ci-dessus, et
        # on renvoie seulement l'adresse brute/partiellement nettoyée en "value" —
        # exactement le comportement d'origine.
        try:
            pred_address = clean_address_text(pred_address)
            pred_address = tag_to_proper_noun(pred_address)

            cp_match = CP_PATTERN.search(pred_address)
            if cp_match:
                parts = {'number': '', 'street': '', 'postal_code': cp_match.group(), 'city': '', 'country': ''}

                before_cp = _normalize_street_segment(pred_address[:cp_match.start()].strip())
                num_match = NUM_PATTERN.match(before_cp)
                if num_match:
                    parts['number'] = num_match.group(1)
                    # Exclut tous les chiffres entre le numéro et la rue
                    parts['street'] = DIGITS_PATTERN.sub('', before_cp[num_match.end():]).strip()
                else:
                    parts['street'] = DIGITS_PATTERN.sub('', before_cp).strip()

                after_cp_cleaned = _normalize_city_segment(pred_address[cp_match.end():].strip())
                words = after_cp_cleaned.split()
                parts['city'] = after_cp_cleaned if len(words) == 1 else ' '.join(words[:-1]).strip()

                if is_french_city(parts['city']):
                    parts['country'] = "FRANCE"
                else:
                    parts['country'] = words[-1] if len(words) > 1 else ""

                pred_address = _address_from_parts(parts)
            # Si pas de code postal trouvé, pred_address reste la chaîne nettoyée/taguée telle quelle.

        except Exception as e:
            logger.debug(e)
            # pred_address garde sa dernière valeur assignée avant l'échec

        return {
            "type": "Address",
            "value": pred_address,
            "rank": np.mean(ranks),
            "empty": False,
            "groups": [],
            "suggestions": [],
            "occurences": occurences,
        }

    except Exception as e:
        logger.debug(e)
        return {
            "type": "Address",
            "value": tag_to_proper_noun(" ".join([occ["value"] for occ in address_preds])),
            "rank": [occ["rank"] for occ in address_preds],
            "empty": False,
            "groups": [],
            "suggestions": [],
            "occurences": [occ["occurences"] for occ in address_preds],
        }
