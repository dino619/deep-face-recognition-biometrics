# Deep Face Recognition with Modern Detectors and Embeddings

This assignment builds on the classical face recognition pipeline from Assignment 3. Here I added deep face detectors and deep face embeddings, then compared them with the handcrafted features under the same gallery/probe setup.

## What I did

- Compared classical descriptors against deep face embeddings for closed-set identification.
- Evaluated Viola-Jones, YOLO face detection, and InsightFace detection.
- Integrated ArcFace and FaceNet/VGGFace2 embeddings.
- Used cosine similarity and CMC metrics for Rank-1 / Rank-5 evaluation.
- Compared how much recognition improves when using deep embeddings instead of handcrafted features.

## Results

Dataset protocol:

- CelebA-HQ-small with 50 identities.
- 475 training images and 412 test images.
- 100 gallery images and 312 probe images.
- Pipeline evaluation used 310/312 probes due to two failed detections.

Detection:

| Detector | Mean IoU | Detection rate |
| --- | ---: | ---: |
| Viola-Jones | 0.686 | 96.84% |
| YOLO face detector | 0.841 | 96.12% |
| InsightFace detector | 0.435 | 48.06% |

Recognition:

| Feature | Whole Rank-1 | Whole Rank-5 | Pipeline Rank-1 | Pipeline Rank-5 |
| --- | ---: | ---: | ---: | ---: |
| LBP_ms | 11.2% | 30.1% | 15.2% | 36.1% |
| HOG | 15.1% | 38.1% | 19.4% | 41.9% |
| Dense SIFT | 19.9% | 42.3% | 25.2% | 45.5% |
| ArcFace (InsightFace) | 100.0% | 100.0% | 97.1% | 98.4% |
| FaceNet (VGGFace2) | 95.2% | 100.0% | 97.4% | 98.7% |

YOLO achieved the best localization quality and was used for the final cropped-face pipeline. Deep embeddings substantially outperformed classical descriptors.

## Steps

1. Reuse the gallery/probe evaluation protocol from Assignment 3.
2. Evaluate detector quality using mean IoU and detection rate.
3. Select the best detector for pipeline crops.
4. Extract classical descriptors and deep embeddings from whole images and face crops.
5. L2-normalize deep embeddings and compare them with cosine similarity.
6. Compute CMC curves and Rank-1 / Rank-5 identification rates.
7. Analyze how deep embeddings change recognition accuracy compared with handcrafted features.

## Repository Structure

```text
scripts/
  run_experiments_a4.py        # Main detector + recognition comparison
  plot_cmc_classical_a4.py     # Classical CMC plotting helper
deep-face-detection/
  YOLO-IBB.ipynb               # YOLO detector exploration
deep-face-recognition/
  FaceRecognition-IBB.ipynb    # Deep embedding exploration
results_a4_retry/
  recognition_metrics_a4.json  # Final recognition metrics
  detection_summary.json       # Detector comparison
  cmc_*.png                    # CMC figures
main.tex                       # IEEE-style report source
Dzaferagic_Dino_SB_Assignment_4.pdf
```

## Tools

- Python
- OpenCV
- YOLOv8 face detector
- InsightFace / ArcFace
- FaceNet with VGGFace2 weights
- scikit-learn, NumPy, Matplotlib
- CMC evaluation and cosine similarity

## How to Reproduce

Install the required Python packages and model dependencies for YOLO, InsightFace, and FaceNet. Then run:

```bash
python scripts/run_experiments_a4.py
```

The raw dataset, local virtual environments, duplicate Assignment 3 copy, and large model weights are excluded from GitHub publication.
