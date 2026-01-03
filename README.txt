#  Point Cloud Completion GAN (V3)

[![Python](https://img.shields.io/badge/Python-3.12.7-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-green.svg)](https://developer.nvidia.com/cuda-zone)

Bu çalışma, eksik veya parçalı nokta bulutu (Point Cloud) verilerini **GAN (Generative Adversarial Network)** mimarisi kullanarak tamamlamayı amaçlayan modüler bir derin öğrenme projesidir.

---

##  Geliştirici Bilgileri
* **İsim:** Elif Özbek
* **Üniversite:** Ostim Teknik Üniversitesi - Yapay Zeka Mühendisliği

---

##  Öne Çıkan Özellikler
* **Mimari:** PointNet tabanlı Encoder/Decoder yapısı kullanılmıştır.
* **Eğitim Stratejisi:** Geometri (Chamfer Distance) ve GAN kaybı tabanlı geçişli (Interleaved) eğitim döngüsü uygulanmıştır.
* **Performans:** NVIDIA RTX 3050 GPU (CUDA) optimizasyonu ile yüksek hızlı eğitim sağlanmıştır.
* **Görselleştirme:** Otomatik 2D kuş bakışı ve 3D karşılaştırmalı sonuç raporlaması mevcuttur.

---

##  Veri Seti (Dataset)
Eğitim ve test süreçlerinde **ShapeNetPart** veri kümesi kullanılmıştır.
 [ShapeNetPart Dataset - Kaggle](https://www.kaggle.com/datasets/majdouline20/shapenetpart-dataset)

---

##  Modüler Dosya Yapısı
| Dosya / Klasör | Görev |
| :--- | :--- |
| `models.py` | Generator ve Discriminator sinir ağı mimarileri. |
| `dataset.py` | .pts dosyalarını yükler ve otomatik maskeleme (occlusion) yapar. |
| `main.py` | Ana eğitim döngüsü ve GPU/Model kayıt yönetimi. |
| `visualize_results.py` | Eğitilmiş modelleri test ederek görsel sonuçlar üretir. |
| `/data` | .pts uzantılı ham nokta bulutu verileri. |
| `/models` | Eğitim sırasında kaydedilen .pth model ağırlıkları. |
| `/Sonuc_Raporu` | Üretilen 2D ve 3D görsel test sonuçları. |

---

##  Kurulum ve Çalıştırma

### 1. Sistem Gereksinimleri
* **Python**: 3.12.7 (Anaconda 'base' ortamı).
* **GPU**: NVIDIA RTX 3050.

### 2. Kütüphane Kurulumu
```bash
# GPU Destekli PyTorch Kurulumu (CUDA 12.4)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)