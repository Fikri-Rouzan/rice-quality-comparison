# Rice Quality Comparison

## 👥 Group 2

| Member Name                      | Student ID       |
| :------------------------------- | :--------------- |
| Hafidz Surya Afifi               | `11230910000002` |
| Ahmad Fauzan Albahy              | `11230910000005` |
| Fahmi Zakaria Nurhasan           | `11230910000053` |
| Muhammad Fikri Rouzan Ash Shidik | `11230910000063` |

---

## 📌 Description

This deep learning implementation identifies and classifies the physical quality of rice grains based on visual features extracted from digital images. The modeling leverages transfer learning across multiple benchmark CNN architectures (MobileNetV2, ResNet50, and EfficientNetB0) to evaluate and compare classification performance across four physical quality categories. These categories comprise **whole** for intact rice grains in prime physical condition, **chalky** for grains exhibiting opaque, chalky white patches, **broken** for fractured or fragmented kernels, and **discolored** for grains suffering from yellowish discoloration, microbial damage, or heat defects.

---

## 💾 Dataset

The dataset used in the development of this model is primary data collected manually, totaling 1,000 digital images. The entire dataset has a balanced class distribution to maintain model performance stability during the training process, with each class containing exactly 250 image samples. This dataset is evenly divided into four physical rice quality categories based on real-world grain conditions: whole, chalky, broken, and discolored.

---

## 🛠️ Tech Stack

| Category                    | Technologies Used                                                                                           |
| :-------------------------- | :---------------------------------------------------------------------------------------------------------- |
| 🌐 **Programming Language** | `Python`                                                                                                    |
| 🌱 **Environment**          | `Jupyter Notebook`                                                                                          |
| 🧩 **Frameworks**           | `TensorFlow`, `Streamlit`                                                                                   |
| ⚛️ **Libraries**            | `NumPy`, `pandas`, `Matplotlib`, `seaborn`, `scikit-learn`, `OpenCV Python`, `SciPy`,<br>`Plotly`, `Pillow` |
| ⚡ **Tool**                 | `Google Colab`                                                                                              |
| 🚀 **Deployment**           | `Streamlit Community Cloud`                                                                                 |

---

## ⚙️ Setup Instructions

1. **Prerequisites**
   - Python 3.11 or higher.
   - Git installed on your system.

2. **Clone the Repository**

```bash
git clone https://github.com/Fikri-Rouzan/rice-quality-comparison.git
cd rice-quality-comparison
```

3. **Create a Virtual Environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

4. **Install Dependencies**

```bash
pip install -r requirements.txt
```

5. **Run the Streamlit Dashboard**

```bash
streamlit run dashboard/dashboard.py
```
