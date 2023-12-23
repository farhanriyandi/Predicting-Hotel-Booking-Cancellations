# Predicting-Hotel-Booking-Cancellations
### Disusun oleh : Farhan Riyandi

# Domain Proyek
__Latar Belakang__

Perkembangan pesat penggunaan internet saat ini mencerminkan evolusi teknologi yang semakin canggih menuju media berbasis online. Banyak konsumen yang cenderung mencari informasi produk dan jasa melalui internet serta melakukan pembelian online karena keterbatasan waktu dan kemudahan yang diberikan. Sektor pariwisata dan perjalanan adalah salah satu industri yang telah mengadopsi teknologi untuk memesan tempat makan, akomodasi, transportasi, dan layanan lainnya. Pemanfaatan teknologi ini tidak hanya mempermudah konsumen dalam bertransaksi, tetapi juga membantu pengusaha dalam efektif mengelola pendapatan dan mengurangi potensi kerugian perusahaan.

![gambar hotel](https://ak-d.tripcdn.com/images/0224v12000a3j493m0E5C_R_600_400_R5_D.webp)

[Referensi gambar](https://id.trip.com/hotels/antalya-hotel-detail-3050567/grida-city-hotel/)

Contohnya, dalam pemesanan kamar hotel, teknologi, khususnya dalam bidang ilmu data, memungkinkan pengelola untuk memahami perilaku pelanggan dan meramalkan kejadian di masa depan. Dalam industri perhotelan, manajemen pembatalan pemesanan merupakan aspek kritis yang dapat mempengaruhi kinerja bisnis secara signifikan. Pembatalan pemesanan dapat menyebabkan penurunan pendapatan, meningkatkan biaya operasional, dan bahkan memengaruhi citra merek hotel. Oleh karena itu, pemahaman yang mendalam tentang faktor-faktor yang mempengaruhi pembatalan pemesanan menjadi suatu keharusan.

Saat ini, data pemesanan hotel telah menjadi sumber informasi berharga untuk memahami perilaku pelanggan dan mengidentifikasi pola pembatalan. Dalam konteks ini, tujuan latar belakang masalah ini adalah untuk mengembangkan suatu model prediktif menggunakan algoritma Random Forest guna mengantisipasi dan meminimalkan dampak pembatalan pemesanan hotel. Dengan sistem prediksi pembatalan pemesanan hotel yang dibuat dapat 
dapat memberikan prediksi pembatalan dengan tepat waktu, memungkinkan manajemen hotel untuk mengidentifikasi reservasi yang berisiko tinggi pembatalan, Dengan mengetahui potensi pembatalan, hotel dapat melakukan penyesuaian harga atau menawarkan insentif kepada pelanggan untuk mempertahankan pemesanan, sehingga mengoptimalkan pendapatan, Dengan informasi yang akurat tentang kecenderungan pembatalan, hotel dapat mengelola kapasitas kamar secara lebih efisien dan mengurangi risiko overbooking. Secara tidak langsung, hal ini dapat meningkatkan pendapatan perusahaan.

* Jelaskan mengapa masalah tersebut harus diselesaikan?
  Masalah tersebut harus diselesaikan, karena dapat membuat kerugian karena adanya pelanggan yang membatalkan pesanan dalam memesan hotel.
  
* Bagaimana masalah tersebut harus diselesaikan?
  Masalah tersebut dapat diselesaikan dengan membangun model machine learning untuk memprediksi pembatalan pesanan hotel dan mencari tahu fitur-fitur yang paling berpengaruh dalam pembatalan pemesanan hotel.

# Business Understanding
Berdasarkan kondisi yang telah diuraikan sebelumnya, perusahaan akan mengembangkan sebuah sistem prediksi pembatalan pemesanan hotel untuk menjawab permasalahan berikut.

### Problem Statements
* Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap pembatalan pemesanan hotel?
* Bagaimana cara mengetahui pelanggan akan membatalkan pesanan atau tidak dengan karakteristik atau fitur-fitur tertentu?

### Goals
Menjelaskan tujuan dari pernyataan masalah:
* Mengetahui fitur yang paling penting terhadap pembatalan pemesanan hotel.
* Membuat model machine learning yang dapat memprediksi pelanggan akan membatalkan pesanan atau tidak seakurat mungkin berdasarkan fitur-fitur yang ada.


### Solution statements
* Menggunakan algoritma random forest dengan hyperparameter tuning menggunakan gridsearchcv.
* Meningkatkan akurasi model dengan menentukan feature importance dari pelatihan model random forest sebelumnya, dengan kata lain ingin memotong fitur ke pelatihan model selanjutnya agar machine tidak terlalu banyak informasi dalam mencari pola.

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

| Fitur                          |  Jumlah Missing Value |
|--------------------------------|-----------------------|
|  hotel                         |  0                    |
|  is_canceled                   |  0                    |
|  lead_time                     |  0                    |
|  arrival_date_year             |  0                    |
|  arrival_date_month            |  0                    |
|  arrival_date_week_number      |  0                    |
|  arrival_date_day_of_month     |  0                    |
|  stays_in_weekend_nights       |  0                    |
|  stays_in_week_nights          |  0                    |
|  adults                        |  0                    |
|  children                      |  4                    |
|  babies                        |  0                    |
|  meal                          |  0                    |
|  country                       |  0                    |
|  market_segment                |  0                    |
|  distribution_channel          |  0                    |
|  is_repeated_guest             |  0                    |
|  previous_cancellations        |  0                    |
|  previous_bookings_not_canceled|  0                    |
|  reserved_room_type            |  0                    |
|  assigned_room_type            |  0                    |
|  booking_changes               |  0                    |
|  deposit_type                  |  0                    |
|  agent                         |  16340                |
|  company                       |  112593               |
|  day_in_waiting_list           |  0                    |
|  customer_type                 |  0                    |
|  adr                           |  0                    |
|  required_car_parking_spaces   |  0                    |
|  total_of_special_request      |  0                    |
|  reservation_status_date       |  0                    |
   



Terdapat data missing value, karena pada fitur company terlalu banyak data yang missng, maka diputuskan untuk menghapus fitur company. Untuk data country akan diisi oleh modus dan agent dan children akan diisi oleh median.

## Visualisasi countplot pada label

![image](https://github.com/farhanriyandi/Predicting-Hotel-Booking-Cancellations/assets/67671418/ecf4de44-a268-47ec-aec4-42529ecfef31)

Jika dilihat perbedaan pada kedua class tidak terlalu jauh, maka dari itu penulis tidak melakukan oversampling atau undersampling untuk menyeimbangkan data.

# Data Preparation

* Label Encoding

Pada Label Encoding, setiap kategori pada suatu feature akan diurutkan secara alfabet dan direpresentasikan dengan sebuah nilai integer. Pada proyek ini mengapa penulis menggunakan label encoding karena jumlah kategori yang ada relatif banyak maka diputuskan menggunakan label encoding ketimbang one hot encoding.

* Dataset Splitting / Train Test Split

Train test split proses membagi data menjadi data latih dan data uji. Data latih digunakan untuk melatih model pembelajaran mesin. Saat proses pelatihan, model belajar dari pola-pola dalam data latih untuk memahami hubungan antara fitur (variabel independen) dan variabel target (variabel dependen). data uji digunakan untuk mengevaluasi kinerja model. Model diuji pada data yang tidak pernah dilihat selama proses pelatihan untuk mengukur seberapa baik model tersebut mampu menggeneralisasi pada data baru. Pada proyek ini penulis dari dataset 119390 membagi data latih 80% dan data uji 20% yang mana 95512 untuk data latih dan 23878 untuk data uji.
Dan menggunakan stratify=y, yang memastikan bahwa distribusi kelas pada data latih dan data uji tetap seimbang sesuai dengan distribusi kelas pada dataset keseluruhan.

# Modeling

Algoritma pada proyek ini hanya menggunakan 1 algoritma yaitu Random Forest dengan menggunakan hyperparameter tuning menggunakan gridsearchcv. Adapun parameter yang dituning pada proyek ini adalah:
* n_estimators: 100, 150, 200
* max_depth: 20, 50, 80
* max_features: 0.3, 0.6, 0.8
* min_samples_leaf: 1, 5, 10

Setelah dilakukan modeling dengan semua fitur, kemudian mencari feature importance (Fitur-fitur yang dianggap penting dapat memberikan wawasan tentang faktor-faktor yang paling memengaruhi hasil prediksi model), dengan kata lain ingin memotong fitur dengan mencoba fitur yang benar-benar dianggap penting agar machine tidak terlalu banyak informasi dalam mencari pola. Berikut adalah hasil dari feature importance:

![image](https://github.com/farhanriyandi/Predicting-Hotel-Booking-Cancellations/assets/67671418/4a56e16d-a353-4625-b335-060cf861df17)

Setelah melihat hasil feature importance penulis mempertimbangkan hanya mengambil 3 fitur paling penting saja untuk melakukan peningkatan akurasi pada model yaitu: reservation_status_date, arrival_date_week_number, arrival_date_year.

# Evaluation
Karena dalam proyek ini adalah klasifikasi metrik evaluasi yang digunakan pada proyek ini adalah metrik akurasi. Dimana formula akurasi adalah sebagai berikut:

accuracy = (TP + TN) / (TP + FP TN + FN)

| Model       | accuracy train | accuracy test |
| ------------| ---------------| --------------|
| RF          | 1.0            | 0.96          |
| RF (3 fitur)| 0.97           | 0.97          |

Berdasarkan hasil diatas model Random Forest (3 fitur)  merupakan pilihan yang lebih baik karena menunjukkan performa yang baik pada kedua set data train dan test, dan cenderung menghindari overfitting yang mungkin terjadi pada model Random Forest dengan semua fitur. 

Referensi: [Pengaruh Seleksi Fitur Pada Algoritma Machine Learning Untuk Memprediksi Pembatalan Pesanan Hotel](https://conference.upnvj.ac.id/index.php/senamika/article/view/1290)



