# Predicting-Hotel-Booking-Cancellations
### Disusun oleh : Farhan Riyandi

# Domain Proyek
__Latar Belakang__

Perkembangan pesat penggunaan internet saat ini mencerminkan evolusi teknologi yang semakin canggih menuju media berbasis online. Banyak konsumen yang cenderung mencari informasi produk dan jasa melalui internet serta melakukan pembelian online karena keterbatasan waktu dan kemudahan yang diberikan. Sektor pariwisata dan perjalanan adalah salah satu industri yang telah mengadopsi teknologi untuk memesan tempat makan, akomodasi, transportasi, dan layanan lainnya. Pemanfaatan teknologi ini tidak hanya mempermudah konsumen dalam bertransaksi, tetapi juga membantu pengusaha dalam efektif mengelola pendapatan dan mengurangi potensi kerugian perusahaan.

![gambar hotel](https://ak-d.tripcdn.com/images/0224v12000a3j493m0E5C_R_600_400_R5_D.webp)

[Referensi gambar](https://id.trip.com/hotels/antalya-hotel-detail-3050567/grida-city-hotel/)

Contohnya, dalam pemesanan kamar hotel, teknologi, khususnya dalam bidang ilmu data, memungkinkan pengelola untuk memahami perilaku pelanggan dan meramalkan kejadian di masa depan. Dengan data yang ada, pengelola dapat menilai apakah pelanggan berpotensi membatalkan pemesanan. Informasi ini memungkinkan pengelola untuk meningkatkan layanan mereka, mengurangi risiko kerugian, dan secara efektif mengelola operasional perusahaan. Selain mengurangi risiko kerugian, pemahaman yang baik tentang perilaku pelanggan juga dapat meningkatkan kepuasan pelanggan, yang pada gilirannya dapat merangsang rekomendasi positif kepada keluarga dan teman-teman. Secara tidak langsung, hal ini dapat meningkatkan pendapatan perusahaan.

* Jelaskan mengapa masalah tersebut harus diselesaikan?
  Masalah tersebut harus diselesaikan, karena dapat membuat kerugian karena adanya pelanggan yang membatalkan pesanan dalam memesan hotel.
  
* bagaimana masalah tersebut harus diselesaikan?
  Masalah tersebut dapat diselesaikan dengan membangun model machine learning untuk memprediksi pembatalan pesanan hotel.
  
Referensi: [Pengaruh Seleksi Fitur Pada Algoritma Machine Learning Untuk Memprediksi Pembatalan Pesanan Hotel](https://conference.upnvj.ac.id/index.php/senamika/article/view/1290)

# Business Understanding

Pada bagian ini, kamu perlu menjelaskan proses klarifikasi masalah.
Bagian laporan ini mencakup:

### Problem Statements
Menjelaskan pernyataan masalah latar belakang:

* Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap pembatalan pemesanan hotel?
* Bagaimana cara mengetahui pelanggan akan membatalkan pesanan atau tidak dengan karakteristik atau fitur-fitur tertentu?

### Goals
Menjelaskan tujuan dari pernyataan masalah:
* Mengetahui fitur yang paling berkolerasi terhadap pembatalan pemesanan hotel
* Membuat model machine learning yang dapat memprediksi pelanggan akan membatalkan pesanan atau tidak seakurat mungkin berdasarkan fitur-fitur yang ada.

### Solution statements
* Mengajukan 2 atau lebih solution statement. Misalnya, menggunakan dua atau lebih algoritma untuk mencapai solusi yang diinginkan atau melakukan improvement pada baseline model dengan hyperparameter tuning.
* Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.

# Data Understanding
Dataset yang digunakan dalam proyek ini adalah dataset yang berupa informasi pemesanan untuk hotel kota dan hotel resor
dataset ini dapat diunduh di[Kaggle: Hotel booking demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand).

### Variabel-variabel pada Hotel booking demand dataset adalah sebagai berikut:
* accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
* cuisine : merupakan jenis masakan yang disajikan pada restoran.
* dst

### Rubrik/Kriteria Tambahan (Opsional):
* Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

# Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

### Rubrik/Kriteria Tambahan (Opsional):
* Menjelaskan proses data preparation yang dilakukan
* Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

# Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

### Rubrik/Kriteria Tambahan (Opsional):
* Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
* Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. Jelaskan proses improvement yang dilakukan.
* Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. Jelaskan mengapa memilih model tersebut sebagai model terbaik.

# Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik akurasi, precision, recall, dan F1 score. Jelaskan mengenai beberapa hal berikut:
* Penjelasan mengenai metrik yang digunakan
* Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

### Rubrik/Kriteria Tambahan (Opsional):
* Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

---Ini adalah bagian akhir laporan---

Catatan:
* Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor Dillinger, Github Guides: Mastering markdown, atau sumber lain di internet. Semangat!
* Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.


