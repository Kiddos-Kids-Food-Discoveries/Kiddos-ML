# Bagian : Machine Learning Project
- Klasifikasi Gambar Makanan dengan 19 Kelas

# Deskripsi :
- Proyek ini bertujuan untuk membangun dan melatih model klasifikasi gambar menggunakan "Convolutional Neural Networks (CNN)". 
- Model ini mengklasifikasikan gambar ke dalam 19 kelas yang berbeda, yang mencakup berbagai jenis makanan.
- Model dibangun dengan menggunakan "TensorFlow","*Keras", dan "OpenCV" untuk pemrosesan gambar.
  ## Preprocessing Data
  - Dalam proyek ini, preprocessing data dilakukan dengan augmentasi tahap awal pada gambar dan penyusunan ulang dataset untuk pelatihan model.
  - Setiap kelas memiliki sekitar 1.000 gambar, dengan total 19 kelas yang menghasilkan 19.000 gambar. 
    ### 1. Augmentasi Gambar
         - Gambar yang telah diproses melalui augmentasi untuk memperbesar variasi data pelatihan dan meningkatkan kinerja model.
         - Augmentasi gambardilakukan dengan menggunakan `ImageDataGenerator` dari Keras dengan parameter berikut:
             - Rescale: Semua nilai piksel gambar dinormalisasi dengan membagi dengan 255, sehingga nilai piksel berada dalam rentang 0-1.
             - Rotation Range: Gambar dapat diputar secara acak dalam rentang 20 derajat.
             - Width Shift Range dan Height Shift Range: Gambar dapat digeser secara acak pada sumbu horizontal dan vertikal dengan rentang 0.2.
             - Shear Range: Menerapkan transformasi geser acak pada gambar dengan rentang 0.2.
             - Zoom Range: Gambar dapat diperbesar atau diperkecil dengan rentang 0.2.
             - Horizontal Flip: Gambar dapat dibalik secara horizontal secara acak.
- Dataset sudah disiapkan ke dalam 3 bagian yaitu 'test' 'train' dan 'validation' setelah dilakukan preprocessing berupa augmentasi terpisah.
- Menghasilkan 15,200 gambar untuk pelatihan, 1,900 gambar untuk validasi, dan 1,900 gambar untuk pengujian.


# Notebook Versi Lengkap
Proyek ini juga disertai dengan versi lengkap dengan format ".ipynb" yang bisa digunakan untuk eksplorasi lebih mendalam terkait model, pelatihan, dan evaluasi. Semua proses yang digunakan dalam pelatihan model, evaluasi, dan visualisasi disertakan dalam file `notebook.ipynb`.

**Catatan:** Proses pelatihan model yang menggunakan hingga 50 epoch dan visualisasi menggunakan `plt.show()` bisa memakan waktu cukup lama, terutama pada dataset besar. Oleh karena itu dibagi menjadi beberapa file/script dalam bentuk ".py" yaitu : 
  - 1. prep_train_model.py (untuk preprocessing data, build model dan train model)
  - 2. predict_model.py (untuk test model terbaik yang dihasilkan)
  - 3. save_model_h5.py (untuk save file best_model.keras dalam format ".h5" agar bisa diproses tim MD dan CC)

# Fitur :
- Model CNN untuk klasifikasi gambar
- Terdapat 19 kelas:
- `apple`, `avocado`, `banana`, `broccoli`, `carrot`, `chicken`, `corn`, `dragon fruit`, `egg`, `grape`, `green vegetables`, `orange`, `porridge`, `potato`, `rice`, `tempeh`, `tofu`, `tomato`, `watermelon`
- <img width="612" alt="1_prep_index" src="https://github.com/user-attachments/assets/9fbb51ca-f3cd-494d-b4c0-aa52a7e72bf5">

- **Validasi model**
    - menggunakan data validasi
- **Penyimpanan model terbaik**
    - secara otomatis selama pelatihan (saved as `best_model.keras`)
- **Konversi model ke format `.h5`**
    - untuk digunakan di lingkungan produksi atau pengujian
    - kode tersimpan pada file "save_model_h5.py" di mana dapat dijalankan setelah file "prep_train_model.py" selesai dijalankan dan file   "best_model.keras" berhasil tersimpan secara otomatis setelah pelatihan selesai dengan menemukan model terbaik.


