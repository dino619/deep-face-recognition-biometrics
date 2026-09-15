#!/usr/bin/env python
"""
Eksperimenti za cevovod prepoznavanja obrazov na zbirki CelebA-HQ-small za nalogo 3.
Koda skrbi za tri kljucne korake: (1) uglasitev Viola-Jones detektorja, da dobimo
stabilne izreze tudi pri zahtevnih slikah, (2) ekstrakcijo znacilk (LBP, HOG,
gosti RootSIFT), ker razlicne teksturne/gradientne predstavitve razlicno dobro
lovijo identitete, in (3) ovrednotenje prepoznave na celih slikah ter na
detektiranih obrazih, da vidimo, kako detekcija vpliva na koncni rezultat.
"""
import argparse  # Parsiranje CLI argumentov
import json  # Branje/pisanje JSON rezultatov
from dataclasses import dataclass  # Za strukturirane zapise (Record)
from pathlib import Path  # Delo s potmi
from typing import Callable, Dict, Iterable, List, Optional, Tuple  # Tip napovedovanja

import cv2  # OpenCV za detekcijo in obdelavo slik
import matplotlib.pyplot as plt  # Risanje grafov CMC
import numpy as np  # Matrike in numericne operacije
import pandas as pd  # Branje CSV anotacij
import torch  # Za DL modele
from insightface import model_zoo  # InsightFace ArcFace embedding
from insightface.app import FaceAnalysis  # InsightFace detektor
from skimage.feature import hog, local_binary_pattern  # Ekstrakcija HOG in LBP
from sklearn.metrics import pairwise_distances  # Izracun razdalj med vektorji
from tqdm import tqdm  # Napredna vrstica
from ultralytics import YOLO  # YOLOv8 detektor

try:
    from facenet_pytorch import InceptionResnetV1  # DL prepoznavanje
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False


@dataclass
class Record:
    """Zapis o posamezni sliki: indeks, identiteta osebe, bounding box obraza in razdelitev (train/test)."""
    idx: str  # Unikatni identifikator slike
    identity: str  # Oznaka osebe (oseba je ime osebe)
    # Pravilni okvir (x, y, sirina, visina) obraza
    bbox: Tuple[int, int, int, int]
    split: str  # ali 'train' ali 'test'


def load_metadata(csv_path: Path) -> List[Record]:
    """Prebere CSV z anotacijami in vrne zapise; branje tu centraliziramo, da imamo enoten izvor resnice."""
    df = pd.read_csv(csv_path)  # Branje CSV datoteke v DataFrame
    records: List[Record] = []  # Akumulator za zapise
    for row in df.itertuples():  # itertuples je hitrejsi kot iterrows
        records.append(
            Record(
                idx=str(row.idx),  # id slike kot niz
                identity=str(row.identity),  # oznaka osebe
                bbox=(int(row.x_1), int(row.y_1), int(
                    # bounding box: x, y, sirina, visina
                    row.width), int(row.height)),
                split=str(row.split),  # train ali test
            )
        )
    return records


def split_records(records: List[Record]) -> Tuple[List[Record], List[Record]]:
    """Razdeli zapise na ucenje/test glede na split, kar poenostavi kasnejsi nadzor nad eksperimentom."""
    train = [r for r in records if r.split ==
             "train"]  # Filteriramo ucne zapise
    test = [r for r in records if r.split ==
            "test"]  # Filteriramo testne zapise
    return train, test


def ensure_dir(path: Path) -> None:
    """Ustvari mapo, ce se ne obstaja, da shranjevanje rezultatov nikoli ne odpove."""
    path.mkdir(
        parents=True, exist_ok=True)  # parents=True ustvari vso hierarhijo direktorijev


def read_image(img_root: Path, idx: str) -> np.ndarray:
    """Nalozi sliko po indeksu in jo pretvori v RGB; standardizacija barvnega prostora je potrebna."""
    path = img_root / f"{idx}.jpg"  # Konstruiramo pot: root + id + .jpg
    img = cv2.imread(str(path))  # OpenCV bere slike v BGR
    if img is None:
        raise FileNotFoundError(path)  # Napaka, ce slike ni
    # Pretvorba v RGB za skladnost z matplotlib/skimage
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def crop_face(img: np.ndarray, bbox: Optional[Tuple[int, int, int, int]], target_size: int = 160) -> np.ndarray:
    """Izreze obraz glede na bbox (z robom) in spremeni velikost; homogena velikost olajsa znacilke."""
    h, w = img.shape[:2]  # Dimenzije izvorne slike
    if bbox is not None:
        x, y, bw, bh = bbox  # Rozpakiranje bounding boxa
        # Okvir malo razsirimo (10%), da zajamemo kontekst (lase, brado)
        margin = 0.1
        mx, my = int(bw * margin), int(bh * margin)
        x1, y1 = max(0, x - mx), max(0, y - my)  # Zacetni vogal z zavarovalko
        # Koncni vogal z zavarovalko
        x2, y2 = min(w, x + bw + mx), min(h, y + bh + my)
        if x2 <= x1 or y2 <= y1:
            face = img  # Ce je bbox neveljaven, uporabimo celo sliko
        else:
            face = img[y1:y2, x1:x2]  # Izrez obraza
    else:
        face = img  # Ce detekcije ni, vrnemo celo sliko
    face = cv2.resize(face, (target_size, target_size),
                      interpolation=cv2.INTER_LINEAR)  # Poravnava na fixno velikost
    return face


def compute_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Izracuna Intersection-over-Union med dvema pravokotnikoma (x, y, w, h); standardna metrika za detekcijo."""
    ax, ay, aw, ah = a  # Prvi pravokotnik
    bx, by, bw, bh = b  # Drugi pravokotnik
    ax2, ay2 = ax + aw, ay + ah  # Desni in spodnji kota prvega
    bx2, by2 = bx + bw, by + bh  # Desni in spodnji kota drugega
    inter_x1, inter_y1 = max(ax, bx), max(ay, by)  # Zacetek preseka
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)  # Konec preseka
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0,
                                                        inter_y2 - inter_y1)  # Dimenzije preseka
    inter_area = inter_w * inter_h  # Povrsina preseka
    union = aw * ah + bw * bh - inter_area  # Povrsina unije
    # IoU = presek / unija
    return float(inter_area / union) if union > 0 else 0.0


# ---------------------- Detekcija ----------------------

def evaluate_detection_dl(records: List[Record], img_root: Path, detector: Callable[[np.ndarray], Optional[Tuple[int, int, int, int]]], max_images: Optional[int] = None) -> Dict:
    """Oceni poljuben detektor (YOLO, InsightFace) na podmnozici slik in vrne IoU statistiko."""
    ious: List[float] = []
    iterator = records if max_images is None else records[:max_images]
    details = []
    for rec in tqdm(iterator, desc="detect"):
        img = read_image(img_root, rec.idx)
        pred = detector(img)
        if pred is None:
            ious.append(0.0)
            details.append({"idx": rec.idx, "iou": 0.0,
                           "detected": False, "gt": rec.bbox, "pred": None})
            continue
        iou = compute_iou(rec.bbox, pred)
        ious.append(iou)
        details.append({"idx": rec.idx, "iou": iou,
                       "detected": True, "gt": rec.bbox, "pred": pred})
    mean_iou = float(np.mean(ious)) if ious else 0.0
    detection_rate = float(np.mean([iou > 0 for iou in ious])
                           ) if ious else 0.0
    return {"mean_iou": mean_iou, "detection_rate": detection_rate, "details": details}


def detect_dataset_dl(records: List[Record], img_root: Path, detector: Callable[[np.ndarray], Optional[Tuple[int, int, int, int]]]) -> Dict[str, Optional[Tuple[int, int, int, int]]]:
    """Za vsak zapis izvede detekcijo z poljubnim detektorjem in vrne slovar idx -> bbox."""
    boxes: Dict[str, Optional[Tuple[int, int, int, int]]] = {}
    for rec in tqdm(records, desc="detect_all"):
        img = read_image(img_root, rec.idx)
        boxes[rec.idx] = detector(img)
    return boxes


def _yolo_detector(model: YOLO):
    """Vrne funkcijo za detekcijo ene slike z YOLO modelom."""
    def detect(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        results = model.predict(img, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return None
        boxes = results[0].boxes
        best_idx = int(torch.argmax(boxes.conf))
        xyxy = boxes.xyxy[best_idx].cpu().numpy()
        x1, y1, x2, y2 = xyxy
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return None
        return int(x1), int(y1), int(w), int(h)
    return detect


def _insightface_detector(app: FaceAnalysis):
    """Vrne funkcijo za detekcijo ene slike z InsightFace (FaceAnalysis)."""
    def detect(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        faces = app.get(bgr)
        if faces is None or len(faces) == 0:
            return None
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0])
                   * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = best.bbox.astype(int)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return None
        return int(x1), int(y1), int(w), int(h)
    return detect


def _iter_grid(grid: Dict[str, List]) -> Iterable[Dict]:
    """Generator vseh kombinacij parametrov za Viola-Jones, da sistematicno preletimo prostor nastavitev."""
    for sf in grid.get("scaleFactor", [1.1]):  # Faktor skaliranja: kako hitro povecujemo okno detekcije
        # Min sosedi: koliko sosednih okvirov so potrebni za potrditev
        for mn in grid.get("minNeighbors", [5]):
            # Min velikost: najmanjsa zaznavna velikost objekta
            for ms in grid.get("minSize", [(30, 30)]):
                yield {"scaleFactor": sf, "minNeighbors": mn, "minSize": tuple(ms)}


def detect_one(img: np.ndarray, cascade: cv2.CascadeClassifier, params: Dict) -> Optional[Tuple[int, int, int, int]]:
    """Detektira en obraz in vrne najvecji bbox; izbiramo najvecjega, ker model pogosto vrne vec okvirov."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Viola-Jones dela na sivinah
    # Detekcija z danimi parametri
    faces = cascade.detectMultiScale(gray, **params)
    if len(faces) == 0:
        return None  # Brez detekcije
    # Izberi najvecji obraz (po povrsini)
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    return int(x), int(y), int(w), int(h)


def evaluate_detection(records: List[Record], img_root: Path, cascade: cv2.CascadeClassifier, params: Dict, max_images: Optional[int] = None) -> Dict:
    """Oceni detekcijo na podmnozici slik, da hitro preverimo, ali parametri zajamejo obraze in ohranijo poravnavo."""
    ious: List[float] = []  # Shranimo IoU za statistiko
    detected = 0  # Stevec uspehov (IoU > 0)
    total = 0  # Stevec vseh primerov
    details = []  # Podrobnosti za analizo posameznih primerov
    # Omejitev zaradi hitrosti
    iterator = records if max_images is None else records[:max_images]
    for rec in tqdm(iterator, desc="detect"):
        img = read_image(img_root, rec.idx)  # Nalozi sliko
        pred = detect_one(img, cascade, params)  # Napoved bbox
        total += 1
        if pred is None:
            ious.append(0.0)  # Brez detekcije -> IoU 0
            details.append({"idx": rec.idx, "iou": 0.0,
                           "detected": False, "gt": rec.bbox, "pred": None})
            continue
        iou = compute_iou(rec.bbox, pred)  # Ujemanje z GT (ground truth)
        detected += 1 if iou > 0 else 0  # Uspeh, ce se prekrivata
        ious.append(iou)  # Shrani IoU
        details.append({"idx": rec.idx, "iou": iou,
                       "detected": True, "gt": rec.bbox, "pred": pred})
    mean_iou = float(np.mean(ious)) if ious else 0.0  # Povprecni IoU
    detection_rate = float(
        np.mean([iou > 0 for iou in ious])) if ious else 0.0  # Delež uspehov
    return {"mean_iou": mean_iou, "detection_rate": detection_rate, "details": details}


def grid_search_detection(train_records: List[Record], img_root: Path, param_grid: Dict, max_images: Optional[int] = 100) -> Tuple[Dict, List[Dict]]:
    """Preizkusi mrezo parametrov za Viola-Jones in vrne najboljse; grid search uporabljamo, ker ima malo gumbov."""
    cascade = cv2.CascadeClassifier(
        # Privzet frontalni model
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    best_params: Dict = {}  # Paramet res z najboljsim IoU
    best_score = -1.0  # Inicijalizacija na -1
    logs: List[Dict] = []  # Dnevnik vseh poskusov
    for params in _iter_grid(param_grid):
        stats = evaluate_detection(
            # Oceni trenutno kombinacijo
            train_records, img_root, cascade, params, max_images=max_images)
        score = stats["mean_iou"]
        logs.append(
            {"params": params, "mean_iou": stats["mean_iou"], "detection_rate": stats["detection_rate"]})
        if score > best_score:
            best_score = score  # Hranimo najboljso kombinacijo
            best_params = params
    return best_params, logs


def detect_dataset(records: List[Record], img_root: Path, params: Dict) -> Dict[str, Optional[Tuple[int, int, int, int]]]:
    """Za vsak zapis izvede detekcijo in vrne slovar idx -> bbox; enak model kot pri uglasitvi."""
    cascade = cv2.CascadeClassifier(
        # Enak model kot pri uglasitvi
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # Slovar za shranjene bboxe
    boxes: Dict[str, Optional[Tuple[int, int, int, int]]] = {}
    for rec in tqdm(records, desc="detect_all"):
        img = read_image(img_root, rec.idx)  # Nalozi sliko
        boxes[rec.idx] = detect_one(
            img, cascade, params)  # Shrani bbox ali None
    return boxes


# ---------------------- Gradnja galerije in probe ----------------------

def make_gallery_probe(
    records: List[Record],
    split: str = "test",
    gallery_per_id: int = 1,
) -> Tuple[List[Record], List[Record]]:
    """Zgradi galerijo in probe znotraj danega splita; ohranimo prekrivanje identitet, ker identifikacija zahteva reference."""
    by_id: Dict[str, List[Record]] = {}  # Grupiraj zapise po identiteti
    for r in records:
        if r.split != split:
            continue  # Filtriramo po zelenem splitu
        by_id.setdefault(r.identity, []).append(r)  # Grupiraj po identiteti
    gallery: List[Record] = []  # Referencne slike
    probe: List[Record] = []  # Slike za prepoznavo
    for identity, recs in by_id.items():
        # Deterministicni vrstni red (sortiranje po indeksu)
        recs_sorted = sorted(recs, key=lambda r: int(r.idx))
        if len(recs_sorted) <= gallery_per_id:
            # Potrebujemo vsaj en probe za oceno identitete
            continue
        gallery.extend(recs_sorted[:gallery_per_id])  # Prve slike v galerijo
        probe.extend(recs_sorted[gallery_per_id:])  # Ostale v probe
    return gallery, probe


# ---------------------- Ekstrakcija znacilk ----------------------

def _prep_gray(img: np.ndarray, size: int = 160) -> np.ndarray:
    """Pripravi sivinsko sliko (spremeni velikost + CLAHE) za ekstrakcijo znacilk; enotna velikost zmanjsa varianco."""
    if img.shape[0] != size or img.shape[1] != size:
        # Poravnava na fixno velikost
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Pretvorba v sivine
    # Izboljsa kontrast za gradientne znacilke
    # Lokalni kontrast izpopolnjitev
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Normalizacija vektorja na L2; s tem metrike (kosinus/evklid) niso odvisne od absolutne energije slike."""
    norm = np.linalg.norm(vec) + 1e-8  # Varovalo proti deljenju z 0
    return (vec / norm).astype(np.float32)


def _lbp_spatial_hist(gray: np.ndarray, radius: int, n_points: int, grid: Tuple[int, int]) -> np.ndarray:
    """Izracun LBP in ga razporedimo v prostorsko mrezo; ohrani lokalno razporeditev vzorcev."""
    lbp = local_binary_pattern(
        gray, n_points, radius, method="uniform")  # Izracun LBP slike
    # riu2 -> n_points + 2 kosa (uniformni LBP ima manj kategorij)
    bins = np.arange(0, n_points + 3)
    gh, gw = grid  # Dimenzije prostorske mreze
    h, w = lbp.shape  # Dimenzije slike
    cell_h, cell_w = h // gh, w // gw  # Velikost ene celice v mrezi
    hists: List[np.ndarray] = []  # Akumulator za histograme
    for gy in range(gh):  # Iteriramo po vrsticah mreze
        for gx in range(gw):  # Iteriramo po stolpcih mreze
            y0, y1 = gy * cell_h, (gy + 1) * \
                cell_h if gy < gh - 1 else h  # y intervalni
            x0, x1 = gx * cell_w, (gx + 1) * \
                cell_w if gx < gw - 1 else w  # x intervalni
            patch = lbp[y0:y1, x0:x1]  # Izrez LBP matrike za to celico
            hist, _ = np.histogram(patch, bins=bins, range=(
                0, n_points + 2), density=True)  # Histogram z normalizacijo
            hists.append(hist.astype(np.float32))  # Dodaj v akumulator
    return np.concatenate(hists)  # Sestavi vse histograme v en vektor


def feat_lbp_multiscale(img: np.ndarray) -> np.ndarray:
    """Ekstrakcija LBP v dveh skalah; dve skali zajameta mikro in makro vzorce obraza."""
    gray = _prep_gray(img)  # Pripravi sivinsko sliko
    # Dve skali zajameta mikro in makro vzorce; prostorska mresa ohrani razporeditev
    grid = (8, 8)  # Prostorska mresa (8x8 celic)
    hists = []  # Akumulator za histograme
    for radius in (2, 3):  # Dve skali: majhni in vecji radii
        n_points = 8 * radius  # Stevilo orientacij (8 orientacij na radij)
        hists.append(_lbp_spatial_hist(
            gray, radius=radius, n_points=n_points, grid=grid))  # Izracun prostorskega LBP
    return l2_normalize(np.concatenate(hists))  # Sestavi in normaliziraj


def feat_hog(img: np.ndarray) -> np.ndarray:
    """Ekstrakcija HOG (Histogram of Oriented Gradients); zajema направљена gradiente, ki so robustni na osvetlitev."""
    gray = _prep_gray(img)  # Pripravi sivinsko sliko
    features = hog(
        gray,
        orientations=9,  # 9 orientacij gradientov (0-180 stopinj)
        pixels_per_cell=(8, 8),  # Velikost celice (8x8 pikslov)
        cells_per_block=(2, 2),  # Velikost bloka (2x2 celic) za normalizacijo
        block_norm="L2-Hys",  # Normalizacija po blokih (L2-Hys je robusten)
        transform_sqrt=True,  # Kvadratni koren za manjso senzitivnost
        feature_vector=True,  # Vrni ravno vektor
    )
    return l2_normalize(features.astype(np.float32))


def feat_dense_sift(img: np.ndarray, step: int = 16, kp_size: int = 16) -> np.ndarray:
    """Ekstrakcija gostega SIFT z napredovanjem po mrezi; zajema lokalne detajle brez zaznavanja kljucnih tock."""
    gray = _prep_gray(img)  # Pripravi sivinsko sliko
    h, w = gray.shape  # Dimenzije slike
    xs = list(range(step // 2, w, step))  # x koordinate mreze
    ys = list(range(step // 2, h, step))  # y koordinate mreze
    keypoints = [cv2.KeyPoint(float(x), float(y), kp_size)
                 # Konstruiraj seznam pseudo-kljucnih tock na mrezi
                 for y in ys for x in xs]
    sift = cv2.SIFT_create()  # Instan ciira SIFT ekstraktor
    # Izracun deskriptorjev na danem lokacijah
    _, desc = sift.compute(gray, keypoints)
    # Pricakovan dolzina (stevilo tock * 128)
    expected_len = len(keypoints) * 128
    if desc is None:
        # Ce deskriptorjev ni (redko), vrni nule
        return np.zeros(expected_len, dtype=np.float32)
    flat = desc.reshape(-1)  # Razprostremi deskriptorje v vektor
    if flat.size < expected_len:
        # Padding, ce je premalo deskriptorjev
        pad = np.zeros(expected_len - flat.size, dtype=flat.dtype)
        flat = np.concatenate([flat, pad])  # Dodaj ničle na konec
    return l2_normalize(flat)  # Normaliziraj vektor


def make_insightface_embedder(model):
    def extractor(img: np.ndarray) -> np.ndarray:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (112, 112), interpolation=cv2.INTER_LINEAR)

        try:
            if not hasattr(model, "get_feat"):
                return np.zeros(512, dtype=np.float32)
            emb = model.get_feat(bgr)
        except Exception:
            return np.zeros(512, dtype=np.float32)

        vec = np.asarray(emb, dtype=np.float32).reshape(-1)
        return l2_normalize(vec)
    return extractor


if FACENET_AVAILABLE:
    def make_facenet_embedder(model, device: torch.device) -> Callable[[np.ndarray], np.ndarray]:
        """Zgradi funkcijo, ki vrne Facenet (VGGFace2) embedding za dan obraz."""
        model.eval()

        def extractor(img: np.ndarray) -> np.ndarray:
            face = cv2.resize(img, (160, 160), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(face.astype(np.float32) / 255.0)
            tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
            tensor = (tensor - 0.5) / 0.5  # Normalizacija na [-1, 1]
            tensor = tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(tensor)
            vec = emb.cpu().numpy().reshape(-1).astype(np.float32)
            return l2_normalize(vec)
        return extractor


FEATURES = {
    "lbp_ms": (feat_lbp_multiscale, "euclidean"),  # LBP, evklidska razdalja
    "hog": (feat_hog, "euclidean"),  # HOG, evklidska razdalja
    # Gost SIFT, evklidska razdalja
    "dense_sift": (feat_dense_sift, "euclidean"),
}


# ---------------------- Prepoznavanje ----------------------

def build_features(records: List[Record], img_root: Path, extractor, det_boxes: Optional[Dict[str, Optional[Tuple[int, int, int, int]]]] = None, target_size: int = 160) -> Tuple[np.ndarray, List[str], List[str]]:
    """Izracuna znacilke za podane zapise; det_boxes omogoca primerjavo med scenariji (cele slike vs detektirani izrezi)."""
    features: List[np.ndarray] = []  # Akumulator za vektorje znacilk
    identities: List[str] = []  # Ohrani identitete za ujemanje
    # Ohrani indekse slik, ki so bile dejansko uporabljene
    used_idx: List[str] = []
    for rec in tqdm(records, desc=f"feat_{extractor.__name__}"):
        # Uporabi detekcijo, ce je dana
        bbox = det_boxes.get(rec.idx) if det_boxes is not None else None
        if det_boxes is not None and bbox is None:
            continue  # Preskoci, ce detekcija ni uspela (in jo pricakujemo)
        img = read_image(img_root, rec.idx)  # Nalozi sliko
        # Pripravi obraz (izrez ali cela slika)
        face = crop_face(img, bbox, target_size=target_size)
        vec = extractor(face)  # Izracun znacilk
        features.append(vec)  # Shrani vektor
        identities.append(rec.identity)  # Shrani identiteto
        used_idx.append(rec.idx)  # Shrani indeks
    if not features:
        return np.empty((0,)), [], []  # Vrni prazne nize, ce ni znacilk
    feats = np.stack(features)  # Sestavi v matriko (N x D)
    return feats, identities, used_idx


def recognition_eval(gallery: List[Record], probe: List[Record], img_root: Path, extractor, det_gallery: Optional[Dict[str, Optional[Tuple[int, int, int, int]]]] = None, det_probe: Optional[Dict[str, Optional[Tuple[int, int, int, int]]]] = None, metric: str = "cosine") -> Dict:
    """Izvede ujemanje med probe in galerijo in izracuna CMC metriko; je osnova za oceno prepoznave."""
    gallery_feats, gallery_ids, gallery_idx = build_features(
        # Izracun znacilk za galerijo (referencne slike)
        gallery, img_root, extractor, det_boxes=det_gallery)
    probe_feats, probe_ids, probe_idx = build_features(
        # Izracun znacilk za probe (slike za prepoznavo)
        probe, img_root, extractor, det_boxes=det_probe)
    if len(probe_feats) == 0 or len(gallery_feats) == 0:
        # Vrni ničle, ce ni dovolj podatkov
        return {"rank1": 0.0, "rank5": 0.0, "cmc": [], "probe_count": len(probe_feats), "gallery_count": len(gallery_feats)}
    # Izracun razdalj med vsemi pari (M x N matrika)
    dists = pairwise_distances(probe_feats, gallery_feats, metric=metric)
    gallery_ids_np = np.array(gallery_ids)  # Pretvorba v numpy
    probe_ids_np = np.array(probe_ids)  # Pretvorka v numpy
    # Akumulator za range (rang pravilnega ujemanja za vsak probe)
    ranks: List[int] = []
    for i in range(dists.shape[0]):  # Za vsak probe
        # Sortiraj galerisjke slike po razdalji (najmanjsa razdalja = najmanj podobna)
        order = np.argsort(dists[i])
        match_positions = np.where(gallery_ids_np[order] == probe_ids_np[i])[
            0]  # Poisci pozicije pravilnih ujemanj (iste osebe)
        if len(match_positions) == 0:
            # Neuspeh: rang nad velikostjo galerije
            ranks.append(len(gallery_ids_np) + 1)
        else:
            # Rang prvega pravilnega ujemanja (+1 ker je rang 1-indexed)
            ranks.append(int(match_positions[0]) + 1)
    ranks_np = np.array(ranks)  # Pretvorka v numpy
    # Delež primerov, ko je pravilno ujemanje na prvem mestu
    rank1 = float(np.mean(ranks_np <= 1))
    # Delež primerov, ko je pravilno ujemanje v prvih 5 mestih
    rank5 = float(np.mean(ranks_np <= 5))
    # Najvecji rang za CMC krivuljo (do 20 ali velikost galerije)
    max_rank = min(20, len(gallery_ids_np))
    cmc = []  # Cumulative Match Characteristic
    for r in range(1, max_rank + 1):
        cmc.append(float(np.mean(ranks_np <= r)))  # Delež uspehov pri rangu r
    return {
        "rank1": rank1,
        "rank5": rank5,
        "cmc": cmc,
        "ranks": ranks,
        "probe_count": len(probe_ids_np),
        "gallery_count": len(gallery_ids_np),
        "probe_idx": probe_idx,
        "gallery_idx": gallery_idx,
    }


def plot_cmc(cmc: List[float], out_path: Path, label: str) -> None:
    """Narise CMC krivuljo in jo shrani; vizualizacija pomaga hitro videti, ali se izboljsave odrazajo v zgodnjih rangih."""
    ensure_dir(out_path.parent)  # Zagotovi, da obstaja mapa
    fig, ax = plt.subplots(figsize=(4, 3))  # Ustvari figuro
    ax.plot(range(1, len(cmc) + 1), cmc, marker="o",
            label=label)  # Nari si CMC krivuljo
    ax.set_xlabel("Rank")  # Oznaka osi x
    ax.set_ylabel("Match rate")  # Oznaka osi y
    ax.set_title(f"CMC: {label}")  # Naslov grafa
    ax.set_ylim(0, 1.05)  # Omejitve osi y
    ax.grid(True, linestyle="--", alpha=0.5)  # Dodaj mreo
    ax.legend()  # Legenda
    fig.tight_layout()  # Avtomatsko prilagodi razporeditev
    fig.savefig(out_path)  # Shrani sliko
    plt.close(fig)  # Sprostim figura za varčevanje z pomnilnikom


# ---------------------- Glavni zagon ----------------------

def run(args):
    """Glavna rutina: pripravi podatke, uglasuje detekcijo, izvede prepoznavo in shrani metrike."""
    csv_path = Path(args.csv)  # Pot do CSV z anotacijami
    img_root = Path(args.img_root)  # Pot do mape s slikami
    results_dir = Path(args.results_dir)  # Pot do izhodne mape za rezultate
    ensure_dir(results_dir)  # Ustvari izhodno mapo, ce ne obstaja
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx_id = 0 if torch.cuda.is_available() else -1
    print(f"Device: {device}")

    print("Loading metadata...")
    records = load_metadata(csv_path)  # Preberi anotacije iz CSV
    train, test = split_records(records)  # Razdeli na train/test
    print(f"Train: {len(train)} images | Test: {len(test)} images")
    gallery, probe = make_gallery_probe(
        records, split=args.rec_split, gallery_per_id=args.gallery_per_id)
    print(
        f"Gallery images: {len(gallery)} | Probe images: {len(probe)} | Identities: {len(set(r.identity for r in gallery))}")
    if len(gallery) == 0 or len(probe) == 0:
        raise RuntimeError(
            "No gallery/probe images built. Check split/galleries or gallery_per_id.")

    det_results = []

    # Viola-Jones detekcija in uglasitev
    if not args.skip_detection:
        param_grid = {
            "scaleFactor": [1.05, 1.1, 1.2],
            "minNeighbors": [3, 5, 7],
            "minSize": [(24, 24), (32, 32), (40, 40)],
        }
        print("Running detection grid search (Viola-Jones)...")
        best_params, grid_logs = grid_search_detection(
            train, img_root, param_grid, max_images=args.tune_images)
        print(f"Best detection params: {best_params}")
        (results_dir / "detection_vj_grid.json").write_text(json.dumps(grid_logs, indent=2))
    else:
        best_params = {"scaleFactor": 1.1,
                       "minNeighbors": 5, "minSize": (32, 32)}
        print(
            f"Skipping tuning. Using default detection params: {best_params}")
    vj_test_stats = evaluate_detection(test, img_root, cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"), best_params)
    (results_dir / "detection_vj_test.json").write_text(json.dumps(vj_test_stats, indent=2))
    det_train_vj = detect_dataset(train, img_root, best_params)
    det_test_vj = detect_dataset(test, img_root, best_params)
    det_results.append(
        {"name": "viola_jones", "stats": vj_test_stats, "boxes_all": {**det_train_vj, **det_test_vj}})

    # YOLO detektor
    print("\nLoading YOLO detector...")
    yolo_model = YOLO(args.yolo_weights)
    yolo_det = _yolo_detector(yolo_model)
    yolo_test_stats = evaluate_detection_dl(
        test, img_root, yolo_det, max_images=args.det_eval_limit)
    (results_dir / "detection_yolo_test.json").write_text(json.dumps(yolo_test_stats, indent=2))
    yolo_train_boxes = detect_dataset_dl(train, img_root, yolo_det)
    yolo_test_boxes = detect_dataset_dl(test, img_root, yolo_det)
    det_results.append({"name": "yolo", "stats": yolo_test_stats,
                       "boxes_all": {**yolo_train_boxes, **yolo_test_boxes}})

    # InsightFace detektor
    print("\nLoading InsightFace detector...")
    det_app = FaceAnalysis(name=args.insightface_det,
                           allowed_modules=["detection"])
    det_app.prepare(ctx_id=ctx_id)
    insight_det = _insightface_detector(det_app)
    insight_test_stats = evaluate_detection_dl(
        test, img_root, insight_det, max_images=args.det_eval_limit)
    (results_dir / "detection_insightface_test.json").write_text(json.dumps(insight_test_stats, indent=2))
    insight_train_boxes = detect_dataset_dl(train, img_root, insight_det)
    insight_test_boxes = detect_dataset_dl(test, img_root, insight_det)
    det_results.append({"name": "insightface", "stats": insight_test_stats,
                       "boxes_all": {**insight_train_boxes, **insight_test_boxes}})

    best_det = max(det_results, key=lambda d: d["stats"].get("mean_iou", 0.0))
    det_all = best_det["boxes_all"]
    summary = [{"name": d["name"], "mean_iou": d["stats"]["mean_iou"],
               "detection_rate": d["stats"]["detection_rate"]} for d in det_results]
    (results_dir / "detection_summary.json").write_text(
        json.dumps({"best": best_det["name"], "summary": summary}, indent=2))
    print(
        f"\nBest detection for pipeline: {best_det['name']} (mean IoU {best_det['stats']['mean_iou']:.3f})")

    # DL prepoznavalniki
    feature_sets = dict(FEATURES)
    print("\nLoading InsightFace ArcFace recognizer...")
    arcface_model = model_zoo.get_model(args.arcface_model)
    if arcface_model is None:
        print(f"Could not load ArcFace model {args.arcface_model}; skipping.")
    else:
        arcface_model.prepare(ctx_id=ctx_id)
        feature_sets["arcface_insightface"] = (
            make_insightface_embedder(arcface_model), "cosine")

    # Drugi InsightFace model (npr. AdaFace) kot rezervni DL prepoznavalnik
    if args.extra_insightface_model:
        try:
            print(
                f"Loading InsightFace extra recognizer ({args.extra_insightface_model})...")
            extra_model = model_zoo.get_model(args.extra_insightface_model)
            if extra_model is None:
                print(
                    f"Could not load extra InsightFace model {args.extra_insightface_model}; skipping.")
            else:
                extra_model.prepare(ctx_id=ctx_id)
                feature_sets[f"{args.extra_insightface_model}_insightface"] = (
                    make_insightface_embedder(extra_model), "cosine")
        except Exception as e:
            print(
                f"Could not load extra InsightFace model {args.extra_insightface_model}: {e}")
    else:
        print("No extra InsightFace model specified; skipping.")

    if FACENET_AVAILABLE:
        print("Loading Facenet (VGGFace2) recognizer...")
        facenet_model = InceptionResnetV1(
            pretrained="vggface2").to(device)  # pretrained model
        feature_sets["facenet_vggface2"] = (
            make_facenet_embedder(facenet_model, device), "cosine")
    else:
        print("Facenet not available (facenet_pytorch not installed). Skipping this recognizer.")

    # Eksperimenti prepoznavanja z razlicnimi znacilkami
    metrics = {}  # Shranjene metrike po metodah
    for name, (extractor, metric) in feature_sets.items():
        print(f"\nFeature: {name}")
        whole = recognition_eval(
            gallery, probe, img_root, extractor, det_gallery=None, det_probe=None, metric=metric)
        metrics[f"whole_{name}"] = {"rank1": whole["rank1"], "rank5": whole["rank5"], "cmc": whole["cmc"],
                                    "probe_count": whole["probe_count"], "gallery_count": whole["gallery_count"]}
        plot_cmc(whole["cmc"], results_dir /
                 f"cmc_whole_{name}.png", label=f"Whole {name}")
        print(
            f"Whole images -> Rank1: {whole['rank1']:.3f} | Rank5: {whole['rank5']:.3f}")

        pipe = recognition_eval(gallery, probe, img_root, extractor,
                                det_gallery=det_all, det_probe=det_all, metric=metric)
        metrics[f"pipeline_{name}"] = {"rank1": pipe["rank1"], "rank5": pipe["rank5"],
                                       "cmc": pipe["cmc"], "probe_count": pipe["probe_count"], "gallery_count": pipe["gallery_count"]}
        plot_cmc(pipe["cmc"], results_dir /
                 f"cmc_pipeline_{name}.png", label=f"Pipeline {name}")
        print(
            f"Pipeline -> Rank1: {pipe['rank1']:.3f} | Rank5: {pipe['rank5']:.3f} | Probes used: {pipe['probe_count']}")

    (results_dir / "recognition_metrics_a4.json").write_text(json.dumps(metrics,
                                                                        indent=2))
    print(f"Saved metrics to {results_dir}")


def parse_args():
    """Parsiranje CLI argumentov za eksperiment."""
    parser = argparse.ArgumentParser(
        description="Eksperimenti cevovoda prepoznavanja obrazov")  # Opis programa
    parser.add_argument(
        # Argumen za CSV
        "--csv", default="Assignment_3/data/CelebA-HQ-small.csv", help="Pot do CSV z anotacijami")
    parser.add_argument(
        # Argumen za mapo s slikami
        "--img-root", default="Assignment_3/data/CelebA-HQ-small", help="Mapa s slikami")
    parser.add_argument("--results-dir", default="results_a4",
                        help="Izhodna mapa za rezultate")  # Argumen za izhodno mapo
    parser.add_argument("--tune-images", type=int, default=120,
                        # Argumen za omejitev slik
                        help="Najvecje stevilo ucnih slik za uglasitev detekcije (None = vse)")
    parser.add_argument("--det-eval-limit", type=int, default=None,
                        help="Najvecje stevilo slik za ocenjevanje DL detektorjev (None = vse)")
    parser.add_argument("--skip-detection", action="store_true",
                        # Argumen za preskakanje detekcije
                        help="Preskoci grid search in uporabi privzete parametre")
    parser.add_argument("--rec-split", default="test", choices=[
                        # Argumen za izbor splita
                        "train", "test"], help="Split, iz katerega vzamemo galerijo/probe (identitete se morajo prekrivati)")
    parser.add_argument("--gallery-per-id", type=int, default=2,
                        # Argumen za broj slik v galeriji
                        help="Koliko slik na identiteto obdrzimo v galeriji (preostanek gre v probe)")
    parser.add_argument("--yolo-weights", default="deep-face-detection/yolov8n-face.pt",
                        help="Pot do YOLOv8 face teze")
    parser.add_argument("--insightface-det", default="buffalo_l",
                        help="Ime InsightFace modela za detekcijo (npr. buffalo_l)")
    parser.add_argument("--arcface-model", default="arcface_r100_v1",
                        help="Ime InsightFace modela za prepoznavo (npr. arcface_r100_v1)")
    parser.add_argument("--extra-insightface-model", default="adaface_ir101_ms1mv3",
                        help="Drugi InsightFace model za prepoznavo (npr. adaface_ir101_ms1mv3)")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())  # Presemli CLI argumente in zagon glavne rutine
