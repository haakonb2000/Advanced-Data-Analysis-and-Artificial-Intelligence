# Brain Tumor MRI Classification

A deep learning project for classifying brain MRI images into four categories: 
**glioma**, **meningioma**, **no tumor**, and **pituitary tumor**. This project 
was completed individually for the course *Advanced Data Analysis & Artificial 
Intelligence* 

## Table of Contents
1. [Dataset](#dataset)
2. [Project Structure](#project-structure)
3. [Methods](#methods)
4. [Results](#results)
5. [Discussion](#discussion)
6. [Future Work](#future-work)
7. [Conclusion](#conclusion)
8. [References](#references)

---

## Dataset

- **Source:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle, Nickparvar, 2021)
- **Classes:** Glioma, Meningioma, No Tumor, Pituitary
- **Training set:** 5,600 images (1,400 per class, balanced)
- **Test set:** 1,600 images (400 per class, balanced)

Exploratory data analysis ([`01_eda.ipynb`](notebooks/01_eda.ipynb)) revealed two key preprocessing challenges:
- Inconsistent image dimensions across samples
- Mixed color formats (grayscale and RGB)

Both were addressed by resizing all images to 224×224 and converting to RGB.

---

## Project Structure

| Notebook | Description |
|---|---|
| [`01_eda.ipynb`](notebooks/01_eda.ipynb) | Exploratory data analysis |
| [`02_baseline_resnet50.ipynb`](notebooks/02_baseline_resnet50.ipynb) | Baseline ResNet50 (transfer learning) + fine-tuning |
| [`03_gradcam.ipynb`](notebooks/03_gradcam.ipynb) | Explainability analysis with Grad-CAM |
| [`04_twostage_classifier.ipynb`](notebooks/04_twostage_classifier.ipynb) | Two-stage classification + fine-tuning of Stage 1 and Stage 2 |
| [`05_efficientnet.ipynb`](notebooks/05_efficientnet.ipynb) | EfficientNet-B0 comparison + fine-tuning |
| [`06_ensemble.ipynb`](notebooks/06_ensemble.ipynb) | Ensemble and sensor fusion experiments |
| [`08_final_model.ipynb`](notebooks/08_final_model.ipynb) | Final model selection, evaluation and error analysis |
| [`results.csv`](results.csv) | Tracked results across all experiments |

---

## Methods

### 1. Baseline (ResNet50)
A ResNet50 pretrained on ImageNet was used with transfer learning: all 
convolutional layers frozen, only the final classification layer retrained. 
This established a baseline of **85.75%** test accuracy.

Grad-CAM analysis (Selvaraju et al., 2017) revealed that the model partially 
relied on shortcut features (e.g., skull edges, background) rather than tumor 
tissue itself for the glioma and meningioma classes, which aligned with their 
lower recall scores.

### 2. Two-Stage Classification
Motivated by the confusion between glioma and meningioma, a two-stage pipeline 
was tested, separating "is there a tumor?" (Stage 1, binary) from "which tumor 
type?" (Stage 2, 3-class). This improved glioma recall (69% → 75.2%) but 
slightly reduced meningioma recall, with similar overall accuracy (85.50%).

### 3. EfficientNet-B0 Comparison
An EfficientNet-B0 (Tan & Le, 2019) model was trained under identical 
conditions to compare architectures. Despite being ~5x smaller than ResNet50 
(20.5MB vs 97.8MB), it achieved comparable but slightly lower accuracy (84.50%).

### 4. Ensemble Methods
Three ensemble strategies combined predictions from ResNet50, EfficientNet, 
and the two-stage models:
- **Equal ensemble** (simple averaging): 87.75%
- **Weighted ensemble** (more weight to the strongest model for tumors): 87.00%
- **Sensor fusion** (class-specific model trust, inspired by sensor fusion in 
  autonomous systems): 86.69%

The equal ensemble performed best among these, primarily by improving 
meningioma recall (85.5%).

### 5. Fine-Tuning
Based on the shortcut behavior identified via Grad-CAM, the last residual 
block (`layer4`) of ResNet50 was unfrozen and retrained with a reduced 
learning rate (1e-4). This was the single most impactful change in the 
project, raising accuracy from 85.75% to **94.19%**. The same fine-tuning 
strategy was applied to EfficientNet (84.50% → 88.75%) and the Stage 2 model.

### 6. Final Model: Fully Fine-tuned Sequential Pipeline
After fine-tuning all models, the sequential pipeline was revisited with both 
Stage 1 and the classifier fully fine-tuned:

- **Fine-tuned Stage 1** (tumor detection): improved from 96.46% → 99.70% 
  training accuracy
- **Fine-tuned ResNet50** as the tumor type classifier (strongest single model)
- **Threshold optimization:** Lowering Stage 1's decision threshold from 0.5 
  to 0.05 reduced missed tumors from 34 to 13 out of 1200 (1.1%) while 
  maintaining 100% notumor recall

The final model achieves **94.88% test accuracy** with only 13/1200 tumors 
misclassified as no tumor.

---

## Results

| Model | Accuracy | Glioma Recall | Meningioma Recall | Notumor Recall | Pituitary Recall | F1 (macro) |
|---|---|---|---|---|---|---|
| ResNet50 baseline | 85.75% | 69.0% | 80.5% | 98.3% | 94.8% | 0.86 |
| Two-stage | 85.50% | 75.2% | 74.5% | 97.2% | 95.0% | 0.85 |
| EfficientNet | 84.50% | 68.0% | 78.0% | 99.0% | 93.0% | 0.84 |
| Equal ensemble | 87.75% | 71.8% | 85.5% | 97.2% | 96.5% | 0.88 |
| Weighted ensemble | 87.00% | 72.8% | 81.8% | 97.2% | 96.2% | 0.87 |
| Sensor fusion | 86.69% | 74.0% | 80.0% | 97.2% | 95.0% | 0.87 |
| Fine-tuned ResNet50 | 94.19% | 79.0% | 99.0% | 99.0% | 100.0% | 0.94 |
| Fine-tuned EfficientNet | 88.75% | 69.0% | 89.0% | 99.0% | 97.0% | 0.88 |
| Sensor fusion (fine-tuned) | 92.38% | 78.8% | 93.8% | 97.2% | 99.8% | 0.92 |
| Sequential + fine-tuned (original Stage 1) | 92.75% | 80.2% | 93.8% | 97.2% | 99.8% | 0.93 |
| Fully fine-tuned sequential (threshold=0.5) | 94.44% | 79.5% | 98.8% | 100.0% | 99.5% | 0.94 |
| **Fully fine-tuned sequential (threshold=0.05)** | **94.88%** | **81.0%** | **99.0%** | **100.0%** | **99.5%** | **0.95** |

Full results are tracked in [`results.csv`](results.csv).

### Explainability (Grad-CAM)

Grad-CAM visualizations on the fine-tuned model show improved focus on tumor 
regions compared to the baseline, particularly for meningioma and pituitary. 
Some shortcut behavior persists for glioma, which is consistent with it 
remaining the most difficult class across all model variants tested.

Error analysis on the worst glioma misclassifications revealed two patterns:
- Some errors are due to remaining shortcut behavior (model focuses on skull edges)
- Some errors appear to be genuinely ambiguous images where the tumor boundary 
  is unclear even visually

---

## Discussion

**Why did fine-tuning help so much?**
Freezing all of ResNet50 forces the model to rely on generic ImageNet features 
(edges, textures, object shapes) that were never optimized to distinguish 
brain tissue patterns. Unfreezing the final residual block allowed the network 
to learn MRI-specific representations, which directly translated to better 
accuracy and more clinically meaningful Grad-CAM activations.

**Why didn't the ensemble improve on the fine-tuned model?**
Ensembles help most when combined models have complementary strengths and 
similar overall quality. Once ResNet50 was fine-tuned, it became substantially 
stronger than EfficientNet and Stage 2, so averaging with weaker models pulled 
predictions away from the (mostly correct) fine-tuned ResNet50 output. This is 
a useful negative result: ensemble methods are not universally beneficial and 
should be validated empirically rather than assumed.

**Why does glioma remain the hardest class?**
Across every model variant, glioma had the lowest recall. Gliomas are 
diffuse and infiltrative tumors with less distinct boundaries than 
meningioma or pituitary tumors, which may make their visual signature in MRI 
inherently harder to learn from a relatively small dataset (1,400 training 
images per class).

**Why did threshold optimization help?**
Stage 1 (tumor detection) was very confident when classifying notumor images, 
but occasionally uncertain on actual tumor images. By lowering the threshold 
from 0.5 to 0.05, meaning an image is sent to the tumor classifier unless 
Stage 1 is at least 95% confident it is notumor, we significantly reduced 
missed tumors (34 → 13) without any loss in notumor accuracy.

**Limitations:**
- Only one dataset was used; generalization to other MRI scanners/protocols 
  is untested.
- Grad-CAM was only evaluated qualitatively on a small sample of images per 
  class, not systematically across the full test set.


---

## Future Work

Given more time, several directions could be explored further:

- **Weighted loss functions:** Penalize misclassification of glioma more 
  heavily during training (e.g. `nn.CrossEntropyLoss(weight=...)`) to directly 
  target its persistently low recall.
- **Confidence-based fusion:** Instead of fixed class-specific trust (as in 
  the sensor fusion experiment), weight each model's vote dynamically based 
  on its prediction confidence for that specific image.
- **More aggressive / targeted augmentation:** Elastic deformations or 
  intensity-based augmentations tailored to MRI data, focused specifically on 
  the glioma class.
- **Systematic Grad-CAM evaluation:** Quantify shortcut behavior across the 
  full test set (e.g. measuring heatmap overlap with a tumor segmentation 
  mask) rather than inspecting a handful of sample images.
- **Additional architectures:** Vision Transformers (ViT) or ConvNeXt could 
  be benchmarked against ResNet50 and EfficientNet.
- **Cross-dataset validation:** Testing the final model on MRI data from a 
  different source/scanner to assess real-world generalization.
- **Full fine-tuning schedule:** Gradually unfreezing earlier layers with a 
  learning rate scheduler, rather than unfreezing a single block at a fixed 
  low learning rate.

---

## Conclusion

Starting from an 85.75% baseline, this project systematically explored 
two-stage classification, an alternative architecture (EfficientNet), multiple 
ensemble strategies, and fine-tuning. The final model, a fully fine-tuned 
sequential pipeline with an optimized decision threshold, achieved **94.88% 
test accuracy**, 81% glioma recall, 100% notumor recall, and only 13/1200 
tumors missed as notumor.

Grad-CAM analysis was used throughout to validate why the model performed 
as it did, not just how well, revealing that the baseline model relied partly 
on non-tumor shortcut features — a risk that fine-tuning partially, but not 
fully, resolved. The most clinically critical finding is that by optimizing 
Stage 1's decision threshold, the pipeline missed only 1.1% of all tumor 
images, an important property for any medical screening system. Even though it got better the model is therefore not good enough for medical use. Even if it was it would need more testing and validation to be certified for medical use.

---

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep Residual Learning for 
  Image Recognition*. arXiv:1512.03385. https://arxiv.org/abs/1512.03385
- Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for 
  Convolutional Neural Networks*. arXiv:1905.11946. https://arxiv.org/abs/1905.11946
- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, 
  D. (2017). *Grad-CAM: Visual Explanations from Deep Networks via 
  Gradient-based Localization*. arXiv:1610.02391. https://arxiv.org/abs/1610.02391
- Nickparvar, M. (2021). *Brain Tumor MRI Dataset*. Kaggle. 
  https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
