# Machine Learning Project : Klasifikasi Gambar Makanan dengan 19 Kelas

# Deskripsi :
Proyek ini bertujuan untuk membangun dan melatih model klasifikasi gambar menggunakan **Convolutional Neural Networks (CNN)**. Model ini mengklasifikasikan gambar ke dalam 19 kelas makanan yang berbeda.
#### 19 Kelas:
`apple`, `avocado`, `banana`, `broccoli`, `carrot`, `chicken`, `corn`, `dragon fruit`, `egg`, `grape`, `green vegetables`, `orange`, `porridge`,     `potato`, `rice`, `tempeh`, `tofu`, `tomato`, `watermelon`

- <img width="612" alt="1_prep_index" src="https://github.com/user-attachments/assets/9fbb51ca-f3cd-494d-b4c0-aa52a7e72bf5">

# Tools yang Digunakan:
### TensorFlow, Keras untuk membangun dan melatih model CNN.
### OpenCV untuk pemrosesan gambar

# Prepocessing Data
* Augmentasi gambar untuk meningkatkan variasi data dan performa model.
* Setiap kelas memiliki sekitar 1.000 gambar, dengan total 19.000 gambar.
* Setelah augmentasi, dataset dibagi menjadi:
    * Training: 15,200 gambar
    * Validation: 1,900 gambar
    * Test: 1,900 gambar

#### 1. Augmentasi Gambar
         - Gambar yang telah diproses melalui augmentasi untuk memperbesar variasi data pelatihan dan meningkatkan kinerja model menggunakan       `ImageDataGenerator` dari Keras dengan parameter berikut:
             - Rescale: Semua nilai piksel gambar dinormalisasi dengan membagi dengan 255, sehingga nilai piksel berada dalam rentang 0-1.
             - Rotation Range: Gambar dapat diputar secara acak dalam rentang 20 derajat.
             - Width Shift Range dan Height Shift Range: Gambar dapat digeser secara acak pada sumbu horizontal dan vertikal dengan rentang 0.2.
             - Shear Range: Menerapkan transformasi geser acak pada gambar dengan rentang 0.2.
             - Zoom Range: Gambar dapat diperbesar atau diperkecil dengan rentang 0.2.
             - Horizontal Flip: Gambar dapat dibalik secara horizontal secara acak.

# Notebook Versi Lengkap
Proyek ini versi lengkap dengan format ".ipynb" yang bisa digunakan untuk eksplorasi dari awal sampai menghasilkan output file model.h5 dengan lebih mendalam terkait model, pelatihan, dan evaluasi. 
Semua proses yang digunakan dalam pelatihan model, evaluasi, dan visualisasi disertakan dalam file `notebook.ipynb`.

# Script .py
**Catatan:** menyediakan file .py terpisah fokus untuk test model dengan menggunakan gambar diluar dataset  : 
   1. **predict_model.py** (untuk test model terbaik yang dihasilkan selama pelatihan menggunakan 50 epoch menghasilkan file **best_model.keras**
   

# Arsitektur Model :
Model menggunakan **CNN**  untuk klasifikasi gambar dengan 4 lapisan konvolusi (Conv2D) untuk mengekstraksi fitur, diikuti dengan batch normalization dan max pooling untuk mereduksi dimensi. **GlobalAveragePooling2D** digunakan untuk mereduksi dimensi lebih lanjut sebelum memasuki lapisan Dense untuk klasifikasi akhir. Dropout diterapkan untuk mencegah overfitting.
## Kompilasi Model :
Model dikompilasi dengan optimizer Adam

# Evaluasi Model
Mengukur performa model dengan menggunakan Confusion Matrix :
Confusion matrix memberikan gambaran tentang seberapa baik model dalam mengklasifikasikan gambar ke dalam kelas yang benar. Berikut adalah confusion matrix yang dihasilkan setelah model diuji pada dataset pengujian:
   <img width="869" alt="5_confusion_matrix" src="https://github.com/user-attachments/assets/51fca590-b652-4aab-baaf-ac8a24ab3c7a">

## Penyimpanan model terbaik :
Model terbaik disimpan otomatis selama pelatihan sebagai best_model.keras. File model ini kemudian bisa dikonversi ke format .h5 untuk digunakan di lingkungan produksi atau pengujian dengan menjalankan save_model_h5.py.

# File Requirements: 
File requirements untuk menjalankan proyek ini dapat ditemukan di file dengan nama "requirements.txt". Bisa menginstall semua dependensi yang diperlukan 


# Cara Clone GitHub Project: 
##### Pastikan sudah menginstal Git di komputer. Jika belum, instal Git terlebih dahulu.
## 1. Clone Repository
Buka Terminal di VSCode, dan jalankan perintah berikut untuk meng-clone repository:
            - git clone https://github.com/Kiddos-Kids-Food-Discoveries/kiddos-ml.git

## 2. Masuk ke Folder Project
Setelah proses cloning selesai, masuk ke folder project dengan perintah:
            - cd kiddos-ml
## 3. Periksa Isi Folder
Untuk memastikan isi folder dan file yang ada, jalankan perintah:
            - ls
## 4. Membuat dan Mengaktifkan Virtual Environment (Opsional, tapi disarankan):
Untuk menginstal semua dependensi yang dibutuhkan oleh proyek ini, bisa menggunakan pip dengan mengikuti langkah berikut:
###### Untuk macOS/Linux:
            - python -m venv venv
            - source venv/bin/activate
###### Untuk Windows:
            - python -m venv venv
            - .\venv\Scripts\activate

## 5. Menginstal Dependensi:
Setelah virtual environment aktif, jalankan perintah berikut untuk menginstal semua dependensi yang tercantum di dalam requirements.txt:
            - pip install -r requirements.txt

## 6. Menjalankan Project:
### 6.1 Menjalankan Notebook Jupyter (notebook.ipynb):
Untuk menjalankan notebook Jupyter, jalankan perintah berikut di terminal:
            - jupyter notebook

### 6.2 Menjalankan Script Python (predict_model.py):
Jika ingin menjalankan file Python seperti predict_model.py untuk langsung menguji model langsung dengan gambar diluar dataset yang disiapkan, jalankan perintah berikut di terminal:
            - python predict_model.py




# Good luck! 😉
