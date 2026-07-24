import ast


def _parse_boxes(value):
    """Convertit une string en liste de dicts, [] si invalide."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    return parsed if isinstance(parsed, list) else []


def signature_match(preds, gts, iou_threshold=0.5):
    """
    Évalue si les prédictions de signature correspondent à la vérité terrain.

    Returns:
        bool: True si match, False sinon (y compris en cas d'erreur).
    """
    try:
        preds = _parse_boxes(preds)
        gts = _parse_boxes(gts)

        if not preds and not gts:
            return True

        if not preds or not gts:
            return False

        pred = max(preds, key=lambda x: x.get("score", 1.0))
        gt = gts[0]

        pred_box = [pred["x1"], pred["y1"], pred["x2"], pred["y2"]]
        gt_box = [gt["x1"], gt["y1"], gt["x2"], gt["y2"]]

        return compute_iou_xyxy(pred_box, gt_box) >= iou_threshold

    except Exception as e:
        print(f"[signature_match] erreur : {type(e).__name__} : {e} | preds={preds!r} gts={gts!r}")
        return False
