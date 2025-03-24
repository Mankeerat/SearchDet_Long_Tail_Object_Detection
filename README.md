# 🔥 SearchDet: Training-Free Long Tail Object Detection via Web-Image Retrieval 🔥
This repository contains the official code for SearchDet, a training-free framework for long-tail, open-vocabulary object detection. SearchDet leverages web-retrieved positive and negative support images to dynamically generate query embeddings for precise object localization—all without additional training.

---

SearchDet is designed to:

- ✅ Enhance Open-Vocabulary Detection: Improve detection performance on long-tail classes by retrieving and leveraging web images.
- ✅ Operate Training-Free: Eliminate the need for costly fine-tuning and continual pre-training by computing query embeddings at inference time.
- ✅ Utilize State-of-the-Art Models: Integrate off-the-shelf models like DINOv2 for robust image embeddings and SAM for generating region proposals.

Our method demonstrates substantial mAP improvements over existing approaches on challenging datasets—all while keeping the inference pipeline lightweight and training-free.

---

Key Features
- Web-Based Exemplars: Retrieve positive and negative support images from the web to create dynamic, context-sensitive query embeddings.
- Attention-Based Query Generation: Enhance detection by weighting support images based on cosine similarity with the input query.
- Robust Region Proposals: Use SAM to generate high-quality segmentation proposals that are refined via similarity heatmaps.
- Adaptive Thresholding: Apply frequency-based thresholding to automatically select the most relevant region proposals.
- Scalable Inference: Achieve strong performance with just a few support images—ideal for long-tailed object detection scenarios.
