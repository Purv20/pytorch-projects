# PyTorch Deep Learning & Computer Vision Repository 🚀

A comprehensive, hands-on repository dedicated to Deep Learning in **PyTorch**, spanning foundational tensor mechanics, custom dataset pipelines, computer vision benchmarks, and a standalone **Real-Time Facial Emotion Detection** application powered by Convolutional Neural Networks (CNNs).

---

## 📌 Project Architecture & Overview

| File / Module | Category | Primary Focus & Architecture |
| :--- | :--- | :--- |
| **[00pytorch.ipynb](00pytorch.ipynb)** | Core Foundations | Tensor operations, GPU acceleration (`cuda`), NumPy bridge, random reproducibility seeds. |
| **[01workflow.ipynb](01workflow.ipynb)** | Core Workflow | End-to-end ML pipeline, Linear Regression, `nn.Module`, training/evaluation loops, model saving/loading. |
| **[02binaryclassfication.ipynb](02binaryclassfication.ipynb)** | Classification | Synthetic circles, decision boundaries, `ReLU` non-linearity, `BCEWithLogitsLoss`. |
| **[03multiclassclassification.ipynb](03multiclassclassification.ipynb)** | Classification | Multi-class synthetic blobs/moons, `BlobModel`, `CrossEntropyLoss`, Softmax probability mapping. |
| **[04computervision.ipynb](04computervision.ipynb)** | Computer Vision | **FashionMNIST**, `DataLoader` batching, Baseline Linear vs CNN (**TinyVGG**), confusion matrix evaluation. |
| **[05customdatasets.ipynb](05customdatasets.ipynb)** | Data Engineering | Custom `torch.utils.data.Dataset` subclass, `ImageFolder`, Data Augmentation transforms, TinyVGG training. |
| **[facialEmotionDetectorCNN.ipynb](facialEmotionDetectorCNN.ipynb)** | **Standalone Project** | **Real-Time Facial Emotion Recognition CNN (`CNNv2`), OpenCV live webcam inference pipeline.** |

---

## 📚 Learning Modules

### 00. PyTorch Fundamentals (`00pytorch.ipynb`)
**Overview:** A concise introduction to PyTorch fundamentals covering tensor creation, mathematical operations, indexing, GPU device setup (`torch.cuda.is_available()`), PyTorch <-> NumPy conversion (`torch.from_numpy`, `.numpy()`), and seed management for reproducibility (`torch.manual_seed`).

---

### 01. PyTorch End-to-End Workflow (`01workflow.ipynb`)
**Overview:** Demonstrates the fundamental 5-step PyTorch model-building workflow on a Synthetic Linear Regression dataset:
1. Data preparation and train/test splitting.
2. Model construction subclassing `nn.Module` (comparing explicit parameters vs `nn.Linear`).
3. Loss function selection (`nn.L1Loss` / MAE) & optimizer configuration (`torch.optim.SGD`).
4. Training and validation loops incorporating `loss.backward()`, `optimizer.step()`, and `torch.inference_mode()`.
5. Model serialization and reloading via `torch.save(state_dict)` and `torch.load_state_dict()`.

---

### 02. Binary Classification (`02binaryclassfication.ipynb`)
**Overview:** Explores binary classification on non-linearly separable circular data (`make_circles`).
* **Key Concept:** Demonstrates why linear models fail on non-linear spatial data (~50% baseline accuracy) and how introducing non-linear activation functions (`nn.ReLU`) warps decision boundaries to achieve **99.5% accuracy**.
* **Loss & Optimization:** Implements `nn.BCEWithLogitsLoss` (which integrates Sigmoid activation into loss computation for numerical stability) paired with `SGD`/`Adam` optimizers.
* **Visualization:** Includes decision boundary plotting utilities to visualize model state transitions.

---

### 03. Multi-Class Classification (`03multiclassclassification.ipynb`)
**Detailed Technical Insights:**
* **Problem Domain & Data:** Extends classification concepts to multi-class non-linear problems using synthetic multi-cluster datasets (`make_blobs` and `make_moons`).
* **Model Architecture (`BlobModel` / `MoonModel`):**
  * Subclasses `nn.Module` using stacked linear layers (`nn.Linear`) interspersed with `nn.ReLU` non-linearities.
  * Configurable hidden layer dimensionality and output node counts matching the number of target classes.
* **Loss Function & Logit Decoding:**
  * Uses `nn.CrossEntropyLoss()`, which combines Softmax activation and Negative Log-Likelihood loss in a single numerically stable function.
  * Maps unnormalized network outputs (logits) into predicted class probabilities via `torch.softmax(logits, dim=1)` and extracts class indices using `torch.argmax()`.
* **Evaluation & Decision Boundary Mapping:** Plots multi-class decision surfaces across feature space to evaluate cluster separation performance.

---

### 04. Computer Vision with PyTorch (`04computervision.ipynb`)
**Detailed Technical Insights:**
* **Benchmark Dataset:** Uses **FashionMNIST** (60,000 training and 10,000 testing 28x28 grayscale images across 10 clothing categories).
* **DataLoader Pipelines:** Implements `torch.utils.data.DataLoader` for batch processing (`batch_size=32`), shuffling, and memory-efficient data streaming.
* **Architectural Progression & Benchmarking:**
  1. **Model 0 (`FashionMNISTModelV0`):** Baseline linear network using `nn.Flatten` and linear layers.
  2. **Model 1 (`FashionMNISTModelV1`):** Non-linear feedforward network with `nn.ReLU`.
  3. **Model 2 (`FashionMNISTModelV2`):** Convolutional Neural Network based on the **TinyVGG** architecture utilizing `nn.Conv2d`, `nn.MaxPool2d`, and `nn.ReLU` blocks.
* **Model Comparison & Diagnostics:**
  * Evaluates execution speed using timer benchmarking (`timeit`).
  * Generates confusion matrices using `torchmetrics.ConfusionMatrix` and `mlxtend.plotting.plot_confusion_matrix` to identify intra-class misclassifications (e.g. Shirts vs T-shirts).
  * Performs random visual sampling with predicted vs true labels and prediction probabilities.

---

### 05. Custom Datasets & Pipeline Engineering (`05customdatasets.ipynb`)
**Detailed Technical Insights:**
* **Custom Data Structuring:** Focuses on custom image classification datasets (Food images: Pizza, Steak, Sushi) organized in standard directory hierarchies (`train/test/class_name/image.jpg`).
* **Data Exploration & Augmentation:**
  * Directory traversal utilities (`os.scandir`, `pathlib.Path`) to inspect class distributions and image shapes.
  * Transform pipelines (`torchvision.transforms.Compose`) incorporating `Resize((64, 64))`, `RandomHorizontalFlip(p=0.5)`, `RandomRotation`, `ToTensor()`, and channel normalization.
* **Building a Custom `Dataset` Subclass:**
  * Creates a custom PyTorch `Dataset` subclass (`class CustomDataset(Dataset)`) by overriding `__len__()` and `__getitem__()`.
  * Implements dynamic image loading from disk via PIL (`PIL.Image.open`), converting paths to numerical target tensors, and building helper dictionary mappings (`find_classes()`).
  * Benchmarks custom dataset loading against `torchvision.datasets.ImageFolder`.
* **Model Training & Regularization:**
  * Trains a **TinyVGG** CNN architecture from scratch on custom loaded data.
  * Analyzes loss/accuracy curves to evaluate the impact of data augmentation in mitigating overfitting.

---

## 🌟 Standalone Project: Real-Time Facial Emotion Detector (`facialEmotionDetectorCNN.ipynb`)

An end-to-end, real-time Computer Vision application designed to perform multi-class **Facial Emotion Recognition (FER)** from live video input.

```
       [ Live Webcam Input ]
                 │
                 ▼
       [ OpenCV Frame Capture ] ──► [ RGB Conversion & PIL Resizing ]
                                                 │
                                                 ▼
[ OpenCV Video Window Display ] ◄── [ Probability & Label Mapping ] ◄── [ CNN Output & Softmax ]
```

### 🔑 Key Machine Learning & Engineering Aspects

* **Deep CNN Architecture (`CNNv2`):**
  * Built with stacked 2D Convolutional blocks (`nn.Conv2d`), Batch Normalization (`nn.BatchNorm2d`), Max Pooling (`nn.MaxPool2d`), and `nn.ReLU` activations.
  * Classifier head features dense linear projections (`nn.Linear`), flattened representations (`nn.Flatten`), and Dropout regularization (`p=0.3`) to prevent feature co-adaptation.

* **Data Augmentation & Normalization:**
  * Training pipeline applies spatial data augmentation (`transforms.RandomHorizontalFlip(p=0.5)`) and standard ImageNet channel normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

* **Model Persistence & Checkpointing:**
  * Evaluates test accuracy after each epoch and automatically serializes the best performing checkpoint state (`model_state`, `class_names`, `img_size`) to disk.

* **Real-Time OpenCV Webcam Inference Engine (`inferencev2()`):**
  * Captures live video stream frames via `cv2.VideoCapture(0)`.
  * Preprocesses frames on-the-fly (RGB color transformation, tensor scaling, normalization, and `unsqueeze(0)` batch dimension expansion).
  * Runs efficient real-time forward pass inside `torch.no_grad()`.
  * Converts logits to class probabilities via `torch.nn.functional.softmax()`, extracts top confidence scores, and overlays real-time emotion labels and confidence percentages on the video feed.

* **Performance Benchmark:**
  * Achieves **65.9% Test Accuracy** on multi-class emotion classification using the augmented deep CNN pipeline.

---

## 🛠️ Requirements & Installation

```bash
# Core Dependencies
pip install torch torchvision
pip install numpy matplotlib pandas scikit-learn
pip install opencv-python pillow torchmetrics mlxtend
```

---

## 🚀 Quick Start

1. **Run Basic Tutorials:** Open and execute notebooks `00pytorch.ipynb` through `05customdatasets.ipynb` sequentially in Jupyter Notebook or VS Code.
2. **Launch Real-Time Facial Emotion Detector:**
   Open `facialEmotionDetectorCNN.ipynb` and run the `inferencev2()` function to launch the live webcam emotion detector interface.
