## Bagian : Machine Learning Project
- Klasifikasi Gambar Makanan dengan 19 Kelas

# Deskripsi :
- Proyek ini bertujuan untuk membangun dan melatih model klasifikasi gambar menggunakan "Convolutional Neural Networks (CNN)". 
- Model ini mengklasifikasikan gambar ke dalam 19 kelas yang berbeda, yang mencakup berbagai jenis makanan.
- Model dibangun dengan menggunakan "TensorFlow","*Keras", dan "OpenCV" untuk pemrosesan gambar.
- Model ini dilatih menggunakan dataset yang terdiri dari 15,200 gambar untuk pelatihan, 1,900 gambar untuk validasi, dan 1,900 gambar untuk pengujian.
- Dataset sudah disiapkan ke dalam 3 bagian yaitu 'test' 'train' dan 'validation' setelah dilakukan preprocessing berupa augmentasi terpisah.
  

# Fitur :
- Model CNN untuk klasifikasi gambar
- Terdapat 19 kelas:<img width="612" alt="1_prep_index" src="https://github.com/user-attachments/assets/9fbb51ca-f3cd-494d-b4c0-aa52a7e72bf5">

- `apple`, `avocado`, `banana`, `broccoli`, `carrot`, `chicken`, `corn`, `dragon fruit`, `egg`, `grape`, `green vegetables`, `orange`, `porridge`, `potato`, `rice`, `tempeh`, `tofu`, `tomato`, `watermelon`
- **Validasi model** menggunakan data validasi
- **Penyimpanan model terbaik** secara otomatis selama pelatihan (saved as `best_model.keras`)
- **Konversi model ke format `.h5`** untuk digunakan di lingkungan produksi atau pengujian.


