#!/usr/bin/env python3
"""
Grid search de tuning (libdmtx ou zxing-cpp) directement sur un DataFrame contenant
une colonne de chemins ALTO (ex: "alto_path"), en testant l'extraction sur les deux
pages de chaque ALTO (page 1 et page 0 par défaut).

Contrairement à un dossier de crops pré-extraits, ce script appelle directement ta
fonction get_image_from_alto(alto_path, page) (importée ou collée ci-dessous), met le
résultat en cache sur disque (extraction ALTO coûteuse, ~2s/page vu ton pipeline), puis
fait tourner la grille de prétraitement + paramètres de décodage sur les images cachées.

Un alto est compté en succès si AU MOINS UNE des deux pages testées est décodée.

=== A ADAPTER AVANT UTILISATION (section juste en dessous des imports) ===
Remplace le bloc "df = ..." et "get_image_from_alto" par ton propre chargement du
DataFrame et ta propre fonction (import direct, ou copier/coller son code).

Usage:
    pip install pylibdmtx zxing-cpp opencv-python numpy pandas tqdm

    python tune_from_alto.py --engine libdmtx --pages 1 0 --sample_n 20

Le premier run est lent (extraction ALTO + cache sur disque dans --cache_dir).
Les runs suivants (autre grille, autre engine) réutilisent le cache et sont rapides.
"""
import argparse
import csv
import itertools
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

MORPH_KERNEL = np.ones((3, 3), np.uint8)


# ============================================================================
# === A ADAPTER : ton DataFrame et ta fonction get_image_from_alto ==========
# ============================================================================

# Option A : import direct depuis ton module/notebook exporté en .py
# from mon_module import df, get_image_from_alto

# Option B : colle ici le code existant, par exemple :
# df = pd.read_pickle("/mnt/data/doc_cni_new_format/df.pkl")
#
# def get_image_from_alto(alto_path, page):
#     ...  # ton implémentation actuelle

df = None                   # <-- remplace par ton DataFrame (contient la colonne alto_col)
get_image_from_alto = None  # <-- remplace par ta fonction (alto_path, page) -> image

# ============================================================================


def to_gray_array(img) -> np.ndarray:
    """Convertit le retour de get_image_from_alto (PIL.Image ou numpy array) en niveaux de gris."""
    if isinstance(img, np.ndarray):
        if img.ndim == 3:
            if img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    return np.array(img.convert("L"))  # PIL.Image


# --------------------------------------------------------------------------
# Cache disque des pages ALTO (évite de rappeler get_image_from_alto à chaque combo)
# --------------------------------------------------------------------------

def build_page_cache(df: pd.DataFrame, alto_col: str, pages: list,
                      get_image_from_alto, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    page_images = {}  # idx -> {page: np.ndarray or None}
    for idx, alto_path in tqdm(list(enumerate(df[alto_col])), desc="Extraction/cache ALTO"):
        page_images[idx] = {}
        for page in pages:
            cache_file = cache_dir / f"{idx:05d}_p{page}.png"
            if cache_file.exists():
                img = cv2.imread(str(cache_file), cv2.IMREAD_GRAYSCALE)
                page_images[idx][page] = img
                continue
            try:
                raw = get_image_from_alto(alto_path=alto_path, page=page)
                gray = to_gray_array(raw)
                cv2.imwrite(str(cache_file), gray)
                page_images[idx][page] = gray
            except Exception:
                page_images[idx][page] = None
    return page_images


# --------------------------------------------------------------------------
# Prétraitement (commun aux deux engines)
# --------------------------------------------------------------------------

def preprocess(gray: np.ndarray, upscale: int, median_ksize: int,
                morph_op: str, threshold_mode: str) -> np.ndarray:
    img = gray
    if upscale != 1:
        img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    if median_ksize > 0:
        img = cv2.medianBlur(img, median_ksize)
    if morph_op == "close":
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, MORPH_KERNEL)
    elif morph_op == "open":
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, MORPH_KERNEL)
    elif morph_op == "close_open":
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, MORPH_KERNEL)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, MORPH_KERNEL)
    if threshold_mode == "otsu":
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_mode == "adaptive":
        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 7
        )
    return img


# --------------------------------------------------------------------------
# Decoders
# --------------------------------------------------------------------------

def try_decode_libdmtx(img: np.ndarray, threshold: int, shrink: int, deviation: int, timeout_ms: int):
    from pylibdmtx.pylibdmtx import decode as dmtx_decode
    try:
        results = dmtx_decode(img, timeout=timeout_ms, shrink=shrink,
                               threshold=threshold, deviation=deviation, max_count=1)
    except Exception:
        return None
    return results[0].data.decode("utf-8", errors="replace") if results else None


def try_decode_zxingcpp(img: np.ndarray, try_rotate: bool, try_downscale: bool,
                         try_invert: bool, binarizer_name: str):
    import zxingcpp
    binarizers = {
        "local_average": zxingcpp.Binarizer.LocalAverage,
        "global_histogram": zxingcpp.Binarizer.GlobalHistogram,
        "fixed_threshold": zxingcpp.Binarizer.FixedThreshold,
    }
    try:
        results = zxingcpp.read_barcodes(
            img, formats=zxingcpp.BarcodeFormat.DataMatrix,
            try_rotate=try_rotate, try_downscale=try_downscale,
            try_invert=try_invert, binarizer=binarizers[binarizer_name],
        )
    except Exception:
        return None
    return results[0].text if results else None


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alto_col", default="alto_path")
    ap.add_argument("--pages", type=int, nargs="+", default=[1, 0],
                     help="Ordre d'essai des pages ALTO. Le premier succès arrête la recherche pour cette ligne.")
    ap.add_argument("--cache_dir", type=Path, default=Path("alto_page_cache"))
    ap.add_argument("--engine", choices=["libdmtx", "zxingcpp"], default="libdmtx")
    ap.add_argument("--output_csv", type=Path, default=None)
    ap.add_argument("--sample_n", type=int, default=None,
                     help="Limiter aux N premières lignes du DataFrame (pour itérer vite)")
    ap.add_argument("--max_combos", type=int, default=200)

    # prétraitement (commun)
    ap.add_argument("--upscales", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--median_ksizes", type=int, nargs="+", default=[0, 3, 5, 7])
    ap.add_argument("--morph_ops", nargs="+", default=["none", "close", "open", "close_open"])
    ap.add_argument("--threshold_modes", nargs="+", default=["none", "otsu", "adaptive"])

    # libdmtx
    ap.add_argument("--thresholds", type=int, nargs="+", default=[25, 35, 45, 55])
    ap.add_argument("--shrinks", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--deviations", type=int, nargs="+", default=[0, 10, 20])
    ap.add_argument("--timeout_ms", type=int, default=3000)

    # zxing-cpp
    ap.add_argument("--try_rotate_options", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--try_downscale_options", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--try_invert_options", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--binarizers", nargs="+",
                     default=["local_average", "global_histogram", "fixed_threshold"])

    args = ap.parse_args()

    if args.output_csv is None:
        args.output_csv = Path(f"results_{args.engine}_from_alto.csv")

    if df is None or get_image_from_alto is None:
        raise SystemExit(
            "df et/ou get_image_from_alto ne sont pas définis. "
            "Édite la section 'A ADAPTER' en haut du script pour les renseigner."
        )

    work_df = df.head(args.sample_n) if args.sample_n else df

    print(f"{len(work_df)} lignes, extraction/cache des pages {args.pages} depuis {args.alto_col}...")
    page_images = build_page_cache(work_df, args.alto_col, args.pages, get_image_from_alto, args.cache_dir)

    preproc_combos = list(itertools.product(
        args.upscales, args.median_ksizes, args.morph_ops, args.threshold_modes
    ))
    if args.engine == "libdmtx":
        engine_combos = list(itertools.product(args.thresholds, args.shrinks, args.deviations))
    else:
        engine_combos = list(itertools.product(
            args.try_rotate_options, args.try_downscale_options, args.try_invert_options, args.binarizers
        ))
    all_combos = list(itertools.product(preproc_combos, engine_combos))

    if len(all_combos) > args.max_combos:
        print(f"{len(all_combos)} combinaisons possibles, échantillonnage de {args.max_combos}")
        all_combos = random.sample(all_combos, args.max_combos)
    else:
        print(f"Test de {len(all_combos)} combinaisons sur {len(work_df)} altos")

    rows = []
    for preproc_params, engine_params in all_combos:
        upscale, median_ksize, morph_op, threshold_mode = preproc_params
        n_success = 0
        t0 = time.perf_counter()
        for idx in range(len(work_df)):
            found = False
            for page in args.pages:
                gray = page_images.get(idx, {}).get(page)
                if gray is None:
                    continue
                proc = preprocess(gray, upscale, median_ksize, morph_op, threshold_mode)
                if args.engine == "libdmtx":
                    threshold, shrink, deviation = engine_params
                    data = try_decode_libdmtx(proc, threshold, shrink, deviation, args.timeout_ms)
                else:
                    try_rotate, try_downscale, try_invert, binarizer_name = engine_params
                    data = try_decode_zxingcpp(proc, bool(try_rotate), bool(try_downscale),
                                               bool(try_invert), binarizer_name)
                if data is not None:
                    found = True
                    break  # une page a marché, inutile de tester l'autre pour cet alto
            if found:
                n_success += 1
        elapsed = time.perf_counter() - t0

        row = {
            "upscale": upscale, "median_ksize": median_ksize,
            "morph_op": morph_op, "threshold_mode": threshold_mode,
            "n_altos": len(work_df), "n_success": n_success,
            "success_rate": round(n_success / len(work_df), 4),
            "avg_time_per_alto_s": round(elapsed / len(work_df), 4),
        }
        if args.engine == "libdmtx":
            row.update({"threshold": engine_params[0], "shrink": engine_params[1], "deviation": engine_params[2]})
        else:
            row.update({"try_rotate": bool(engine_params[0]), "try_downscale": bool(engine_params[1]),
                        "try_invert": bool(engine_params[2]), "binarizer": engine_params[3]})
        rows.append(row)

    rows.sort(key=lambda r: r["success_rate"], reverse=True)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nRésultats sauvegardés dans {args.output_csv}\n")
    print(f"Top 10 combinaisons ({args.engine}, succès = au moins une page décodée) :")
    for r in rows[:10]:
        extra = (f"thr={r['threshold']} shr={r['shrink']} dev={r['deviation']}" if args.engine == "libdmtx"
                 else f"rot={r['try_rotate']} dwn={r['try_downscale']} inv={r['try_invert']} bin={r['binarizer']}")
        print(f"{r['success_rate']*100:5.1f}%  up={r['upscale']} med={r['median_ksize']} "
              f"morph={r['morph_op']} thrmode={r['threshold_mode']}  {extra}  "
              f"({r['avg_time_per_alto_s']:.3f}s/alto)")


if __name__ == "__main__":
    main()
