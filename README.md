# Machine Learning Project : Klasifikasi Gambar Makanan dengan 19 Kelas

# Deskripsi :
Proyek ini bertujuan untuk membangun dan melatih model klasifikasi gambar menggunakan **Convolutional Neural Networks (CNN)**. Model ini mengklasifikasikan gambar ke dalam 19 kelas makanan yang berbeda.

## 19 Kelas:
`apple`, `avocado`, `banana`, `broccoli`, `carrot`, `chicken`, `corn`, `dragon fruit`, `egg`, `grape`, `green vegetables`, `orange`, `porridge`,     `potato`, `rice`, `tempeh`, `tofu`, `tomato`, `watermelon`

- <img width="612" alt="1_prep_index" src="https://github.com/user-attachments/assets/9fbb51ca-f3cd-494d-b4c0-aa52a7e72bf5">

# Tools yang Digunakan:
### TensorFlow, Keras, OpenCV untuk pemrosesan gambar

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
Proyek ini juga disertai dengan versi lengkap dengan format ".ipynb" yang bisa digunakan untuk eksplorasi dari awal sampai menghasilkan output file model.h5 dengan lebih mendalam terkait model, pelatihan, dan evaluasi. 
Semua proses yang digunakan dalam pelatihan model, evaluasi, dan visualisasi disertakan dalam file `notebook.ipynb`.

**Catatan:** Proses pelatihan model yang menggunakan hingga 50 epoch dan visualisasi menggunakan `plt.show()` bisa memakan waktu cukup lama, terutama pada dataset besar. Oleh karena itu dibagi menjadi beberapa file/script dalam bentuk ".py" yaitu : 
   1. prep_train_model.py (untuk preprocessing data, build model dan train model)
   2. predict_model.py (untuk test model terbaik yang dihasilkan sebelum save model dalam format h5)
   3. save_model_h5.py (untuk save file best_model.keras dalam format ".h5" agar bisa diproses tim MD dan CC)

# Arsitektur Model :
Model menggunakan **CNN**  untuk klasifikasi gambar dengan 4 lapisan konvolusi (Conv2D) untuk mengekstraksi fitur, diikuti dengan batch normalization dan max pooling untuk mereduksi dimensi. **GlobalAveragePooling2D** digunakan untuk mereduksi dimensi lebih lanjut sebelum memasuki lapisan Dense untuk klasifikasi akhir. Dropout diterapkan untuk mencegah overfitting.

## Kompilasi Model :
- Model dikompilasi dengan optimizer Adam


## Penyimpanan model terbaik :
    - secara otomatis selama pelatihan (saved as `best_model.keras`)
## Konversi model ke format `.h5`
    - untuk digunakan di lingkungan produksi atau pengujian
    - kode tersimpan pada file "save_model_h5.py" di mana dapat dijalankan setelah file "prep_train_model.py" selesai dijalankan dan file   "best_model.keras" berhasil tersimpan secara otomatis setelah pelatihan selesai dengan menemukan model terbaik.

# File Requirements: 
File requirements untuk menjalankan proyek ini dapat ditemukan di file dengan nama "requirements.txt". Anda bisa menginstall semua dependensi yang diperlukan dengan menjalankan perintah berikut: 'pip install -r requirements.txt'


