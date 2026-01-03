======================================================================
         POINT CLOUD COMPLETION (NOKTA BULUTU TAMAMLAMA) GAN - V3
======================================================================
Geliştirici: Elif Özbek
----------------------------------------------------------------------
1. PROJE TANIMI VE ÖZELLİKLERİ
----------------------------------------------------------------------
Bu proje, eksik/parçalı nokta bulutu (Point Cloud) verilerini GAN 
(Generative Adversarial Network) mimarisi kullanarak tamamlamayı 
amaçlar. 

Özellikler:
* PointNet tabanlı Encoder/Decoder yapısı.
* Geometri (Chamfer Distance) ve GAN kaybı tabanlı geçişli eğitim.
* NVIDIA RTX 3050 GPU (CUDA) desteği.
* 2D ve 3D karşılaştırmalı sonuç raporlama.

----------------------------------------------------------------------
2. VERİ SETİ KAYNAĞI (DATASET SOURCE)
----------------------------------------------------------------------
Projede ShapeNetPart veri seti kullanılmıştır.
Dataset Linki: https://www.kaggle.com/datasets/majdouline20/shapenetpart-dataset

----------------------------------------------------------------------
3. DOSYA YAPISI
----------------------------------------------------------------------
* models.py            : Model mimarilerini içerir.
* dataset.py           : .pts dosyalarını yükler ve maskeleme yapar.
* main.py              : Eğitim döngüsü ve model kayıt işlemlerini yönetir.
* visualize_results.py : Test yapar ve görselleri kaydeder.
* requirements.txt     : Gerekli kütüphane listesi.
* /data                : .pts uzantılı veri dosyaları.
* /models              : Kaydedilen .pth dosyaları.
* /Sonuc_Raporu        : Üretilen test sonuçları.

----------------------------------------------------------------------
4. KURULUM VE ÇALIŞTIRMA
----------------------------------------------------------------------
Python: 3.12.7 (Anaconda 'base')
Donanım: NVIDIA RTX 3050

Kurulum:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Çalıştırma Komutları:
Eğitim: python main.py
Test  : python visualize_results.py

Not: os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' ayarı kütüphane 
çakışmalarını önlemek için kodlara dahil edilmiştir.
======================================================================