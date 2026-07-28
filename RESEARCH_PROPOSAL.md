# RESEARCH PROPOSAL & CLINICAL PROTOCOL

**Project Title:**  
*An Explainable, End-to-End Clinical Diagnostic Workstation for Pediatric Pneumonia Classification Using Deep Residual Networks (ResNet-50), Real-Time Grad-CAM Visual Attribution, and Automated Clinical Documentation*

**Principal Investigator & Lead System Architect:**  
**Abdullah Ishaq**  
*Proprietary Medical AI Research Pipeline — All Rights Reserved*

---

## 1. Executive Summary & Statement of Novelty

Recent advancements in deep learning have demonstrated notable sensitivity in detecting thoracic abnormalities from chest radiographs. However, the overwhelming majority of academic research in medical imaging is relegated to static, isolated "black-box" models executed in offline notebooks. These basic classification scripts fail to satisfy critical clinical requirements: they lack interpretability, provide no mechanism for sensitivity calibration, and cannot be integrated into real-world hospital workflows.

This research proposes and implements a comprehensive, **explainable Clinical Decision Support System (CDSS)** specifically engineered for pediatric pneumonia screening. Unlike standard classification scripts, this project introduces a fully operational, end-to-end workstation combining high-precision deep residual learning (ResNet-50) with **Gradient-Weighted Class Activation Mapping (Grad-CAM)**, dynamic diagnostic thresholding, and automated clinical ReportLab PDF report generation. This platform bridges the gap between theoretical deep learning and clinical deployment.

---

## 2. Clinical Background & Rationale

Pediatric pneumonia remains one of the leading infectious causes of mortality in children under five globally. Rapid and accurate interpretation of pediatric chest radiographs (CXR) is paramount for timely antimicrobial intervention. However, pediatric radiographs present unique diagnostic challenges:
1. **Subtle Pathology:** Early alveolar infiltrates and peribronchial thickening can be easily obscured by thymic shadows or patient motion artifacts.
2. **Physician Fatigue & Variability:** In high-volume emergency departments, inter-reader variability among general radiologists and residents can lead to delayed or missed diagnoses.
3. **The Black-Box Trust Deficit:** Clinicians rightfully reject automated diagnostic systems that provide a binary label without visual evidence or attribution.

**The Solution:** An interactive diagnostic workstation that acts as an expert second reader, highlighting exact pathological regions of interest while allowing radiologists to retain final diagnostic authority.

---

## 3. Technical Architecture: Why This Extends Beyond a "Basic AI Model"

To overcome the limitations of conventional AI papers, this system integrates four advanced pillars:

```
[Pediatric CXR Input] 
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ Deep Feature Extraction (Modified ResNet-50 Backbone)  │
└────────────────────────────────────────────────────────┘
         │
         ├──────────────────────────────────────┐
         ▼                                      ▼
┌──────────────────────────────────┐   ┌─────────────────────────────────┐
│  Logits & Probability Inference  │   │  Grad-CAM Visual Attribution    │
│  (Dynamic Threshold Calibration) │   │  (Last Convolutional Layer XAI) │
└──────────────────────────────────┘   └─────────────────────────────────┘
         │                                      │
         └───────────────────┬──────────────────┘
                             ▼
┌────────────────────────────────────────────────────────┐
│ Proprietary Clinical Workstation (Next.js / FastAPI)   │
│  ─ Real-time Heatmap Overlay & Opacity Sliders         │
│  ─ Automated ReportLab Diagnostic PDF Export           │
└────────────────────────────────────────────────────────┘
```

### A. Deep Residual Feature Extraction (ResNet-50)
Rather than relying on shallow convolutional networks prone to overfitting on chest radiographs, our architecture utilizes a 50-layer deep Residual Network (ResNet-50) optimized via transfer learning. The network is trained on validated pediatric cohorts, utilizing spatial data augmentation (random rotations, horizontal flips, contrast jitter) to ensure invariant feature representations of lobar consolidations and interstitial opacities.

### B. Explainable AI via Real-Time Grad-CAM
To eliminate the black-box paradigm, the inference engine integrates **Grad-CAM (Gradient-Weighted Class Activation Mapping)** directly into the forward/backward pass. By computing the gradient of the PNEUMONIA score with respect to feature maps of the final convolutional layer (`layer4`), the system generates a coarse localization heatmap. This heatmap is upsampled and blended onto the patient's radiograph in real time, pinpointing the precise alveolar infiltrates driving the neural network's classification.

### C. Dynamic Diagnostic Sensitivity Calibration
Standard models enforce a static 0.50 decision boundary. In clinical practice, diagnostic thresholds must adapt to patient risk profiles. Our workstation implements an interactive **Diagnostic Threshold Slider**:
* **High-Sensitivity Screening Mode (Threshold $\approx$ 0.25 - 0.35):** Deployed in triage settings to minimize false negatives, ensuring no developing infection is overlooked.
* **High-Specificity Confirmatory Mode (Threshold $\approx$ 0.65 - 0.75):** Deployed when evaluating ambiguous infiltrates to minimize false positives and prevent unnecessary antibiotic administration.

### D. Automated Clinical Documentation & Edge Integration
The backend is architected as an asynchronous FastAPI microservice capable of GPU or lightweight CPU inference. It communicates seamlessly with a modern Next.js client interface featuring glassmorphism aesthetics, real-time image manipulation (contrast, brightness, zoom), and an embedded **ReportLab PDF Engine** that instantly compiles patient metrics, confidence scores, visual XAI heatmaps, and medical disclaimers into formal diagnostic records.

---

## 4. Methodology & Evaluation Protocol

1. **Dataset Curation & Preprocessing:**
   * Stratified partitioning of pediatric chest X-rays into Training, Validation, and independent Test cohorts.
   * Standardized normalization to ImageNet RGB distributions with adaptive resizing.
2. **Model Training Protocol:**
   * Loss Function: Weighted Cross-Entropy to account for any class imbalances between Normal and Pneumonia presentations.
   * Optimization: AdamW optimizer with cosine annealing learning rate schedules.
3. **Clinical Validation Metrics:**
   * Beyond standard Accuracy, the model is rigorously evaluated on **ROC-AUC (Receiver Operating Characteristic Area Under Curve)**, **Sensitivity (Recall)**, **Specificity**, and **F1-Score**.
   * Expert radiological qualitative review of Grad-CAM localization accuracy against known clinical findings.

---

## 5. Expected Clinical & Academic Contributions

1. **Academic Impact:** Demonstrates how explainable AI (XAI) and dynamic thresholding can be embedded into real-time web applications without sacrificing inference latency.
2. **Clinical Utility:** Provides emergency departments and pediatric clinics with an intuitive, zero-latency second reader that enhances diagnostic confidence and standardizes reporting.
3. **Open-Source vs. Proprietary Roadmap:** During the peer-review and validation phase, this system remains under strict proprietary governance by Abdullah Ishaq to protect intellectual property and patient data privacy protocols.

---

## 6. Project Timeline & Status

* **Phase 1 (Completed):** Core ResNet-50 architecture design, training pipeline implementation, and validation evaluation.
* **Phase 2 (Completed):** Real-time Grad-CAM XAI X-ray overlay generation and FastAPI microservice integration.
* **Phase 3 (Completed):** High-performance clinical web workstation UI (Next.js) and automated ReportLab PDF generation.
* **Phase 4 (Current):** Clinical demonstration, trial evaluations, and formal manuscript submission for peer review.

---
*Proprietary Research Proposal — Copyright (c) 2026 Abdullah Ishaq. All rights reserved.*
