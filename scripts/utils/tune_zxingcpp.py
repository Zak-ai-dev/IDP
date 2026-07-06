#!/usr/bin/env python3
"""
Grid search de tuning pour zxing-cpp sur un dossier d'images de DataMatrix (2D-Doc CNI).

Teste des combinaisons de prétraitement (upscale, median blur, morphologie,
mode de seuillage) x options read_barcodes() de zxingcpp (try_rotate,
try_downscale, try_invert, binarizer), et classe les combinaisons par taux
de succès sur le dossier fourni.

Le median blur + la morphologie (close/open) sont utiles quand l'image du
DataMatrix est grainée/moucheté à l'intérieur des modules (bruit de papier ou
d'impression), un cas fréquent sur les CNI photographiées/scannées en basse
qualité : le seuillage échoue non pas par manque de résolution mais parce que
le bruit intra-module empêche de trancher noir/blanc proprement.

Usage:
    pip install zxing-cpp opencv-python numpy
    python tune_zxingcpp.py --images_dir /chemin/vers/crops --output_csv results_zxingcpp.csv

Optionnel, pour mesurer l'exactitude (pas juste le taux d'extraction) :
    --gt_csv ground_truth.csv   (colonnes: filename,expected)

Si le nombre de combinaisons est trop grand, utiliser --max_combos pour échantillonner
aléatoirement un sous-ensemble de la grille plutôt que de tout tester.

Note: l'API python de zxing-cpp a légèrement évolué selon les versions. Ce script
cible la version courante (pip install zxing-cpp) exposant zxingcpp.read_barcodes(...)
avec les paramètres formats/try_rotate/try_downscale/try_invert/binarizer. Si ta version
diffère, adapte la fonction try_decode() en conséquence (cf. message d'erreur affiché).
"""
import argparse
import csv
import itertools
import random
import time
from pathlib import Path

import cv2
import numpy as np
import zxingcpp

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MORPH_KERNEL = np.ones((3, 3), np.uint8)

BINARIZERS = {
    "local_average": zxingcpp.Binarizer.LocalAverage,
    "global_histogram": zxingcpp.Binarizer.GlobalHistogram,
    "fixed_threshold": zxingcpp.Binarizer.FixedThreshold,
}


def load_images(images_dir: Path):
    images = []
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append((p.name, img))
    return images


def preprocess(gray: np.ndarray, upscale: int, median_ksize: int,
                morph_op: str, threshold_mode: str) -> np.ndarray:
    img = gray
    if upscale != 1:
        img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    if median_ksize > 0:
        # élimine le bruit grainé intra-module sans flouter les frontières
        # entre modules (contrairement à un flou gaussien/bilateral)
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


def try_decode(img: np.ndarray, try_rotate: bool, try_downscale: bool,
                try_invert: bool, binarizer_name: str):
    try:
        results = zxingcpp.read_barcodes(
            img,
            formats=zxingcpp.BarcodeFormat.DataMatrix,
            try_rotate=try_rotate,
            try_downscale=try_downscale,
            try_invert=try_invert,
            binarizer=BINARIZERS[binarizer_name],
        )
    except TypeError as e:
        raise SystemExit(
            "L'API zxingcpp installée ne correspond pas à celle attendue par ce script "
            f"(erreur: {e}). Vérifie 'pip show zxing-cpp' et adapte try_decode()."
        )
    except Exception:
        return None
    if results:
        return results[0].text
    return None


def load_ground_truth(gt_csv: Path):
    if gt_csv is None:
        return {}
    gt = {}
    with open(gt_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt[row["filename"]] = row["expected"]
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, type=Path)
    ap.add_argument("--gt_csv", type=Path, default=None)
    ap.add_argument("--output_csv", type=Path, default=Path("results_zxingcpp.csv"))
    ap.add_argument("--upscales", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--median_ksizes", type=int, nargs="+", default=[0, 3, 5, 7],
                     help="0 = pas de median blur. Doit rester petit devant la taille d'un module.")
    ap.add_argument("--morph_ops", nargs="+", default=["none", "close", "open", "close_open"],
                     choices=["none", "close", "open", "close_open"])
    ap.add_argument("--threshold_modes", nargs="+", default=["none", "otsu", "adaptive"],
                     choices=["none", "otsu", "adaptive"])
    ap.add_argument("--try_rotate_options", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--try_downscale_options", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--try_invert_options", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--binarizers", nargs="+",
                     default=["local_average", "global_histogram", "fixed_threshold"])
    ap.add_argument("--max_combos", type=int, default=200,
                     help="Echantillonne aléatoirement si le nombre total de combos dépasse ce seuil")
    args = ap.parse_args()

    images = load_images(args.images_dir)
    if not images:
        print(f"Aucune image trouvée dans {args.images_dir}")
        return
    gt = load_ground_truth(args.gt_csv)

    preproc_combos = list(itertools.product(
        args.upscales, args.median_ksizes, args.morph_ops, args.threshold_modes
    ))
    zx_combos = list(itertools.product(
        args.try_rotate_options, args.try_downscale_options, args.try_invert_options, args.binarizers
    ))
    all_combos = list(itertools.product(preproc_combos, zx_combos))

    if len(all_combos) > args.max_combos:
        print(f"{len(all_combos)} combinaisons possibles, échantillonnage de {args.max_combos}")
        all_combos = random.sample(all_combos, args.max_combos)
    else:
        print(f"Test de {len(all_combos)} combinaisons sur {len(images)} images")

    rows = []
    for (upscale, median_ksize, morph_op, threshold_mode), (try_rotate, try_downscale, try_invert, binarizer_name) in all_combos:
        n_success = 0
        n_correct = 0
        t0 = time.perf_counter()
        for filename, gray in images:
            proc = preprocess(gray, upscale, median_ksize, morph_op, threshold_mode)
            data = try_decode(proc, bool(try_rotate), bool(try_downscale), bool(try_invert), binarizer_name)
            if data is not None:
                n_success += 1
                if filename in gt and data == gt[filename]:
                    n_correct += 1
        elapsed = time.perf_counter() - t0
        rows.append({
            "upscale": upscale,
            "median_ksize": median_ksize,
            "morph_op": morph_op,
            "threshold_mode": threshold_mode,
            "try_rotate": bool(try_rotate),
            "try_downscale": bool(try_downscale),
            "try_invert": bool(try_invert),
            "binarizer": binarizer_name,
            "n_images": len(images),
            "n_success": n_success,
            "success_rate": round(n_success / len(images), 4),
            "n_correct": n_correct if gt else "",
            "avg_time_per_image_s": round(elapsed / len(images), 4),
        })

    rows.sort(key=lambda r: r["success_rate"], reverse=True)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nRésultats sauvegardés dans {args.output_csv}\n")
    print("Top 10 combinaisons (par taux de succès) :")
    header = (f"{'succ.':>6} {'up':>3} {'med':>4} {'morph':>10} {'thrmode':>8} "
              f"{'rot':>4} {'dwn':>4} {'inv':>4} {'binarizer':>16} {'t/img':>7}")
    print(header)
    for r in rows[:10]:
        print(f"{r['success_rate']*100:5.1f}% {r['upscale']:>3} {r['median_ksize']:>4} "
              f"{r['morph_op']:>10} {r['threshold_mode']:>8} {str(r['try_rotate']):>4} "
              f"{str(r['try_downscale']):>4} {str(r['try_invert']):>4} "
              f"{r['binarizer']:>16} {r['avg_time_per_image_s']:>6.3f}s")


if __name__ == "__main__":
    main()
