# DeepDetect-2025: Intelligent System for Deepfake Detection

This repository houses the formal research methodology, algorithmic preprocessing pipelines, and system architecture framework for **IT41043 — Intelligent Systems (Milestone 2)**. This project investigates the automated detection of authentic versus artificially generated (deepfake) facial imagery by implementing a carefully curated dataset subset alongside an advanced dual-branch detection framework.


## 🚀 Project Overview

* **Objective:** To develop a highly accurate and explainable intelligent system capable of distinguishing between real and digitally manipulated (deepfake) facial images using the `deepdetect-2025` dataset.
* **The Research Gap:** Traditional deepfake detectors often act as "black boxes," providing probability scores without justification. This project directly addresses this limitation by emphasizing model transparency and human-interpretable evidence.
* **Core Architecture:** The system leverages a novel **dual-branch architecture**:
  * **Branch A (Deep Features):** A fine-tuned Xception CNN extracts global manipulation artifacts, such as blending boundaries and texture flaws across the face.
  * **Branch B (Biological Features):** A biological eye-feature extraction module uses facial landmarks to analyze corneal specular reflections and pupil symmetry, targeting GAN-generated irregularities.
* **Explainability & Output:** Rather than a simple binary classification, the final output fuses both branches to provide a `Real` or `Fake` decision alongside a **Grad-CAM heatmap** (visual evidence) and a **biological inconsistency score** (numerical evidence).
* **Data Collection & Bias Mitigation:** Data sourced from the `deepdetect-2025` database is automatically scanned and filtered for specific target demographics using the DeepFace library. This approach is designed to evaluate and mitigate potential racial bias in deepfake detectors[cite: 2].
* **Dataset Size:** To prevent class imbalances and ensure stable training, the curated dataset is strictly balanced at a 1:1 ratio, consisting of 1,000 authentic and 1,000 synthetic (fake) high-resolution images[cite: 2].
* **Preprocessing Pipeline:** Prior to feature extraction, raw images undergo an automated preprocessing pipeline that includes face cropping, alignment, standard resizing to $224 \times 224$ pixels, and RGB pixel-value normalization to $[0, 1]$ to optimize gradient convergence[cite: 2].

* ## 📂 Folder Hierarchy 


```text
deepdetect-2025/
│
├── data/
│   ├── filtered_real/     
│   └── filtered_fake/      
│
├── preprocessing/
│   ├── filter_faces.py      
│
├── notebooks      
├── requirements.txt       
└── README.md
```

---

## ⚙️ Setup & Execution Guidelines

To run this project on your local machine or within the Kaggle environment, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/your-username/deepdetect-2025.git](https://github.com/your-username/deepdetect-2025.git)
   cd deepdetect-2025
