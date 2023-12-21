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
* Menggunakan Algoritma Random Forest dengan hyperparameter tuning menggunakan gridsearchcv.
* Kemudian mencoba meningkatkan akurasi model dengan mencari feature importance, dengan kata lain ingin cut feature agar machine tidak terlalu banyak informasi dalam mencari pola.

# Data Understanding
Dataset yang digunakan dalam proyek ini adalah dataset yang berupa informasi pemesanan untuk hotel kota dan hotel resor
dataset ini dapat diunduh di [Kaggle: Hotel booking demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand).

Berikut informasi pada dataset :

* Dataset memiliki format CSV (Comma-Seperated Values).
* Dataset memiliki 119390 sample dengan 32 fitur.
* Dataset memiliki 16 fitur bertipe int64, 4 fitur bertipe float64, 12 fitur bertipe object.
* Terdapat missing value pada data

## Exploratory Data Analysis - Deskripsi Variabel
### Variabel-variabel pada Hotel booking demand dataset adalah sebagai berikut:.
* hotel: menunjukkan jenis hotel, apakah itu "Resort Hotel" atau "City Hotel".
* is_canceled: variabel target yang menunjukkan apakah pemesanan dibatalkan atau tidak (1: dibatalkan, 0: tidak dibatalkan)
* lead_time: menunjukkan jumlah hari antara tanggal pemesanan dan tanggal kedatangan.
* arrival_date_year: menunjukkan informasi tentang tahun kedatangan pemesanan.
* arrival_date_month: menunjukkan informasi tentang bulan kedatangan pemesanan.
* arrival_date_week_number: fitur yang menunjukkan nomor minggu dalam setahun ketika tamu dijadwalkan untuk tiba
* arrival_date_day_of_month: menunjukkan informasi tentang tanggal kedatangan pemesanan
* stays_in_weekend_nights: menunjukkan jumlah malam yang dihabiskan oleh tamu pada akhir pekan
* stays_in_week_nights: menunjukkan jumlah malam yang dihabiskan oleh tamu pada hari kerja
* adults: Jumlah tamu dewasa
* children: Jumlah tamu anak-anak
* babies: Jumlah tamu bayi
* meal: Menunjukkan jenis paket makanan yang dimiliki oleh tamu (e.g., BB: Bed & Breakfast, HB: Half Board).
* country: Negara asal tamu
* market_segment: menunjukkan segmen pasar tempat pemesanan dibuat
* distribution_channel: merujuk pada saluran distribusi atau cara di mana tamu membuat pemesanan
* is_repeated_guest: menunjukkan apakah tamu adalah pengunjung berulang atau bukan
* previous_cancellations: jumlah pemesanan yang sebelumnya dibatalkan oleh tamu
* previous_bookings_not_canceled: jumlah pemesanan sebelumnya yang tidak dibatalkan oleh tamu
* reserved_room_type: tipe kamar yang telah dipesan
* assigned_room_type: tipe kamar yang diberikan saat check-in
* booking_changes: jumlah perubahan yang dibuat pada pemesanan sebelum check-in
* deposit_type: Jenis deposit yang diberikan oleh tamu
* agent: ID agen
* company:  ID perusahaan yang terkait dengan pemesanan hotel
* days_in_waiting_list: jumlah hari dalam waiting list sebelum pemesanan dikonfirmasi
* customer_type: jenis tamu
* adr: Average Daily Rate, rata-rata biaya per kamar per malam
* required_car_parking_spaces: jumlah tempat parkir mobil yang dibutuhkan tamu
* total_of_special_requests: total permintaan khusus yang dibuat oleh tamu.
* reservation_status: status pemesanan
* reservation_status_date: tanggal status pemesanan
  
Fitur reservation_status memiliki nilai yang sama dengan target yaitu is_canceled. Maka dari itu diputuskan untuk menghapus fitur tersebut

### Menangani Missing Value
![image](https://github.com/farhanriyandi/Predicting-Hotel-Booking-Cancellations/assets/67671418/7fd61b8d-ba5a-4dec-8e06-c06839f55eec)

Terdapat data missing value, karena pada fitur company terlalu banyak data yang missng, maka diputuskan untuk menghapus fitur company. Untuk data country akan diisi oleh modus dan agent dan children akan diisi oleh median.


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


