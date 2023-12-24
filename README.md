# *Predicting-Hotel-Booking-Cancellations*
### Disusun oleh : Farhan Riyandi

# Domain Proyek
__Latar Belakang__

Perkembangan pesat penggunaan internet saat ini mencerminkan evolusi teknologi yang semakin canggih menuju media berbasis online. Banyak konsumen yang cenderung mencari informasi produk dan jasa melalui internet serta melakukan pembelian online karena keterbatasan waktu dan kemudahan yang diberikan. Sektor pariwisata dan perjalanan adalah salah satu industri yang telah mengadopsi teknologi untuk memesan tempat makan, akomodasi, transportasi, dan layanan lainnya. Pemanfaatan teknologi ini tidak hanya mempermudah konsumen dalam bertransaksi, tetapi juga membantu pengusaha dalam efektif mengelola pendapatan dan mengurangi potensi kerugian perusahaan.

![gambar hotel](https://ak-d.tripcdn.com/images/0224v12000a3j493m0E5C_R_600_400_R5_D.webp)

[Referensi gambar](https://id.trip.com/hotels/antalya-hotel-detail-3050567/grida-city-hotel/)

Contohnya, dalam pemesanan kamar hotel, teknologi, khususnya dalam bidang ilmu data, memungkinkan pengelola untuk memahami perilaku pelanggan dan meramalkan kejadian di masa depan [1]. Dalam industri perhotelan, manajemen pembatalan pemesanan merupakan aspek kritis yang dapat mempengaruhi kinerja bisnis secara signifikan. Pembatalan pemesanan dapat menyebabkan penurunan pendapatan, meningkatkan biaya operasional, dan bahkan memengaruhi citra merek hotel. Oleh karena itu, pemahaman yang mendalam tentang faktor-faktor yang mempengaruhi pembatalan pemesanan menjadi suatu keharusan.

Saat ini, data pemesanan hotel telah menjadi sumber informasi berharga untuk memahami perilaku pelanggan dan mengidentifikasi pola pembatalan. Dalam konteks ini, tujuan latar belakang masalah ini adalah untuk mengembangkan suatu model prediktif menggunakan algoritma Random Forest guna mengantisipasi dan meminimalkan dampak pembatalan pemesanan hotel. Dengan sistem prediksi pembatalan pemesanan hotel yang dibuat
dapat memberikan prediksi pembatalan dengan tepat waktu, memungkinkan manajemen hotel untuk mengidentifikasi reservasi yang berisiko tinggi pembatalan, Dengan mengetahui potensi pembatalan, hotel dapat melakukan penyesuaian harga atau menawarkan insentif kepada pelanggan untuk mempertahankan pemesanan, sehingga mengoptimalkan pendapatan, Dengan informasi yang akurat tentang kecenderungan pembatalan, hotel dapat mengelola kapasitas kamar secara lebih efisien dan mengurangi risiko *overbooking*. Secara tidak langsung, hal ini dapat meningkatkan pendapatan perusahaan.

* Jelaskan mengapa masalah tersebut harus diselesaikan?
  Masalah tersebut harus diselesaikan, karena dapat membuat kerugian karena adanya pelanggan yang membatalkan pesanan dalam memesan hotel.
  
* Bagaimana masalah tersebut harus diselesaikan?
  Masalah tersebut dapat diselesaikan dengan membangun model machine learning untuk memprediksi pembatalan pesanan hotel dan mencari tahu fitur-fitur yang paling berpengaruh dalam pembatalan pemesanan hotel.

# *Business Understanding*
Berdasarkan kondisi yang telah diuraikan sebelumnya, perusahaan akan mengembangkan sebuah sistem prediksi pembatalan pemesanan hotel untuk menjawab permasalahan berikut.

### *Problem Statements*
* Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap pembatalan pemesanan hotel?
* Bagaimana cara mengetahui pelanggan akan membatalkan pesanan atau tidak dengan karakteristik atau fitur-fitur tertentu?
* Bagaimana perbandingan akurasi, presisi, recall dan f1-score dari hasil model dengan fitur yang lengkap pemilihan feature importance atau menggunakan fitur-fitur penting saja pada model random forest?

### *Goals*
Menjelaskan tujuan dari pernyataan masalah:
* Mengetahui fitur yang paling penting terhadap pembatalan pemesanan hotel.
* Membuat model machine learning yang dapat memprediksi pelanggan akan membatalkan pesanan atau tidak seakurat mungkin berdasarkan fitur-fitur yang ada.
* Membandingkan hasil akurasi, presisi, recall dan f1-score dari random forest dengan fitur yang lengkap dengan pemilihan feature importance atau menggunakan fitur-fitur penting saja pada model *random forest*.


### *Solution statements*
* Menggunakan algoritma *random forest* dengan *hyperparameter tuning* menggunakan *gridsearchcv* sebagai baseline model, lalu mencoba melakukan *improvement* dengan menggunakan *feature importance*, yang mana hanya ingin menggunakan fitur-fitur paling berpengaruh dalam pelatihan model pertama untuk pelatihan model kedua.
* Membandingkan tingkat kedua model yakni model pertama dengan fitur lengkap dengan fitur model kedua dengan hanya fitur yang paling berpengaruh pada model pertama.

# *Data Understanding*
Dataset yang digunakan dalam proyek ini adalah dataset yang berupa informasi pemesanan untuk hotel kota dan hotel resor
dataset ini dapat diunduh di [Kaggle: Hotel booking demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand).

Berikut informasi pada dataset :

* Dataset memiliki format CSV *(Comma-Seperated Values)*.
* Dataset memiliki 119390 sample dengan 32 fitur.
* Dataset memiliki 16 fitur bertipe int64, 4 fitur bertipe *float64*, 12 fitur bertipe *object*.
* Terdapat *missing value* pada data

## Deskripsi variabel
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

### Menangani *Missing Value*

| Fitur                          |  Jumlah *Missing Value* |
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

Begitu banyak data yang missing value pada kolom *company* dan maka dari itu diputuskan untuk menghapus fitur tersebut.

## EDA Terhadap data *missing value*

- Fitur *Agent*

![image](https://github.com/farhanriyandi/Predicting-Hotel-Booking-Cancellations/assets/67671418/22cbdce7-e8c2-4dc0-8f4b-98c49368426b)

![image](https://github.com/farhanriyandi/Predicting-Hotel-Booking-Cancellations/assets/67671418/7994d1ce-30dd-463d-b015-2e78a59f2b83)

Berdasarkah hasil visualisasi data pada fitur agent dengan boxplot dan histogram, tidak ada outlier tetapi data tersebut tidak berdistribusi normal melainkan skewness positif. Oleh karena itu, median adalah pilihan yang lebih baik untuk mengisi missing value pada data tersebut. Median adalah nilai tengah dari data, dan merupakan nilai yang paling mewakili data. Median tidak terpengaruh oleh outlier, dan tidak sensitif terhadap distribusi data.

- Fitur *Children*

![image](https://github.com/farhanriyandi/Predicting-Hotel-Booking-Cancellations/assets/67671418/f46421f5-a34d-4527-978c-3da1b4ede71c)

![image](https://github.com/farhanriyandi/Predicting-Hotel-Booking-Cancellations/assets/67671418/eabb3670-5e36-426f-8623-4d235725d8ef)

Berdasarkah hasil visualisasi data pada fitur children dengan boxplot dan histogram, ada outlier dan data tersebut tidak berdistribusi normal melainkan skewness positif. Oleh karena itu, median adalah pilihan yang lebih baik untuk mengisi missing value pada data tersebut. Median adalah nilai tengah dari data, dan merupakan nilai yang paling mewakili data. Median tidak terpengaruh oleh outlier, dan tidak sensitif terhadap distribusi data.

- Fitur *Country*

Pada data country yakni data kategorik terdapat missing value sebanyak 488 dari 119390. Data tersebut hanya terhilang 0.41% proporsi missing value relatif kecil dibawah 1%, maka dari itu pada proyek ini diputuskan menggunakan modus dalam mengisi missing value tersebut.

## Visualisasi *countplot* pada label

![image](https://github.com/farhanriyandi/Predicting-Hotel-Booking-Cancellations/assets/67671418/ecf4de44-a268-47ec-aec4-42529ecfef31)

Jika dilihat perbedaan pada kedua *class* tidak terlalu jauh, maka dari itu diputuskan tidak melakukan *oversampling* ataupun *undersampling* untuk menyeimbangkan data.

# *Data Preparation*

* *Label Encoding*

Pada _Label Encoding_, setiap kategori pada suatu *feature* akan diurutkan secara alfabet dan direpresentasikan dengan sebuah nilai integer. Pada proyek ini mengapa menggunakan _label encoding_ ketimbang _one hot encoding_:
  * Memiliki variabel kategori dengan jumlah kategori yang sangat banyak yakni 30 fitur.
  * Label Encoding dapat lebih efisien dalam hal penggunaan memori karena tidak membuat kolom baru untuk setiap kategori seperti yang dilakukan oleh One-Hot Encoding.

* _Dataset Splitting / Train Test Split_

*Train test split* proses membagi data menjadi data latih dan data uji. Data latih digunakan untuk melatih model pembelajaran mesin. Saat proses pelatihan, model belajar dari pola-pola dalam data latih untuk memahami hubungan antara fitur (variabel independen) dan variabel target (variabel dependen). data uji digunakan untuk mengevaluasi kinerja model. Model diuji pada data yang tidak pernah dilihat selama proses pelatihan untuk mengukur seberapa baik model tersebut mampu menggeneralisasi pada data baru. Dalam kasus prediksi pembatalan pemesanan hotel digunakan rasio 80:20 dikarenakan 119390 data dalam dataset ini.

Hasil dari pembagian data latih dan data uji dengan rasio 80:20 adakah sebagai berikut:

  * Total data keseluruhan  119390
  * Total data latih  95512
  * Total data uji  23878

Dikarenakan dataset sudah besar yakni 119390, pembagian rasio data latih dan data uji 80:20 pada data uji sudah memiliki cukup data untuk menguji model memiliki kinerja yang baik.

# *Modeling*

Pada proyek ini menggunakan algoritma random forest dikarenakan pada penelitian 'Pengaruh Seleksi Fitur Pada Algoritma Machine Learning Untuk Memprediksi Pembatalan Pesanan Hotel'[1], pada random forest memiliki akurasi yang tinggi yakni mencapai 98%. Oleh karena itu dalam proyek ini menggunakan algoritma *random forest*. 

_Random Forest_ adalah sebuah algoritma machine learning yang digunakan untuk tugas klasifikasi, regresi, dan juga untuk menentukan _feature importance_. Algoritma ini bekerja dengan menggabungkan prediksi dari beberapa pohon keputusan _(decision trees)_ yang dibangun secara acak.

Berikut adalah langkah-langkah umum bagaimana algoritma Random Forest bekerja:

1. _Bootstrapping (Random Sampling with Replacement)_:
   * Setiap pohon keputusan dalam Random Forest dibangun dengan menggunakan subset acak dari data latih. Ini dilakukan dengan cara melakukan pengambilan sampel         secara acak dengan penggantian dari dataset latih. Proses ini disebut bootstrapping.

2. Membangun Pohon Keputusan _(Decision Tree)_:
   * Untuk setiap subset yang dihasilkan dari _bootstrapping_, sebuah pohon keputusan dibangun. Pohon ini dibangun dengan mengambil pertimbangan dari subset 
     tersebut dan menggunakan metode seperti CART _(Classification and Regression Trees)_ untuk membagi data berdasarkan fitur-fitur yang ada.

3. Prediksi oleh Setiap Pohon:
   * Setelah pohon-pohon keputusan dibangun, setiap pohon memberikan prediksi untuk setiap data uji. Pada tugas klasifikasi, prediksi dari setiap pohon diambil 
     berdasarkan mayoritas suara, sedangkan pada regresi, dapat diambil rata-rata prediksi.

4. Aggregasi Prediksi:
   * Prediksi dari semua pohon digabungkan untuk menghasilkan prediksi akhir. Proses ini disebut penggabungan _(aggregation)_ atau voting.

5. _Feature Importance_:
   * Salah satu kelebihan utama dari _Random Forest_ adalah kemampuannya untuk mengukur tingkat kepentingan _(importance)_ dari setiap fitur dalam membuat prediksi. 
     Ini dapat diukur dengan melihat seberapa sering suatu fitur digunakan untuk membagi data di semua pohon, dan seberapa baik fitur tersebut dapat meningkatkan 
     prediksi akhir.
   *  Skor kepentingan fitur dihitung berdasarkan seberapa banyak mengurangi ketidakmurnian _(impurity)_ atau seberapa banyak meningkatkan kriteria pemisahan di setiap pohon ketika suatu fitur digunakan. Fitur 
      yang sering digunakan dan memberikan kontribusi besar terhadap pemisahan yang baik akan memiliki skor 
      kepentingan yang tinggi.
    * _Feature importance_ dapat digunakan untuk mendapatkan pemahaman yang lebih baik tentang kontribusi setiap fitur terhadap model dan membantu dalam pemilihan 
      fitur.

Pada proyek ini menggunakan _hyperparameter tuning_ menggunakan _gridsearchcv_. *Hyperparameter tuning* dapat meningkatkan kinerja model karena memungkinkan menemukan kombinasi nilai *hyperparameter* yang optimal untuk model.*Grid Search* melibatkan pengujian semua kombinasi yang mungkin dari sejumlah hyperparameter yang telah ditentukan sebelumnya. Ini menciptakan grid dari nilai yang mungkin dan menguji setiap kombinasinya.

adapun parameter yang di *tuning* menggunakan *gridsearchcv* pada proyek ini adalah:
* *n_estimators*: 100, 150, 200
* *max_depth*: 20, 50, 80
* *max_features*: 0.3, 0.6, 0.8
* *min_samples_leaf*: 1, 5, 10

Setelah dilakukan modeling dengan semua fitur, kemudian mencari *feature importance* (Fitur-fitur yang dianggap penting dapat memberikan wawasan tentang faktor-faktor yang paling memengaruhi hasil prediksi model), dengan kata lain ingin memotong fitur dengan mencoba fitur yang benar-benar dianggap penting agar machine tidak terlalu banyak informasi dalam mencari pola. Berikut adalah hasil dari *feature importance*:

![image](https://github.com/farhanriyandi/Predicting-Hotel-Booking-Cancellations/assets/67671418/bc0b9996-6c44-491b-9e95-37de67c2d8b0)

Setelah melihat hasil *feature importance*, diputuskan hanya mengambil 3 fitur paling penting saja untuk melakukan peningkatan akurasi pada model yaitu: *reservation_status_date, arrival_date_week_number, arrival_date_year*.

# *Evaluation*
Metrik evaluasi yang digunakan pada proyek ini adalah Akurasi, Presisi, Recall, dan F1-score. Akurasi, Presisi, *Recall*, dan *F1-Score* adalah metrik evaluasi yang digunakan dalam konteks klasifikasi untuk mengevaluasi kinerja model. 

1. Akurasi:
   * Formula:

     ![image](https://github.com/farhanriyandi/Predicting-Hotel-Booking-Cancellations/assets/67671418/518aed33-840a-47ce-9bf0-25c2931460f1)

   * Akurasi mengukur sejauh mana model dapat memprediksi dengan benar. Ini adalah rasio antara jumlah prediksi benar dengan total jumlah prediksi.

2. Presisi:
   * Formula:
     
     ![image](https://github.com/farhanriyandi/Predicting-Hotel-Booking-Cancellations/assets/67671418/6f1e517d-eaa3-48e0-91a1-71c33394757f)

   * Presisi mengukur sejauh mana model dapat memprediksi positif dengan benar. Ini memberikan informasi tentang berapa persen dari kelas positif yang diprediksi benar oleh model.

3. *Recall*:
   * Formula:
     
     ![image](https://github.com/farhanriyandi/Predicting-Hotel-Booking-Cancellations/assets/67671418/02b83806-035f-4b7d-8291-45784a9676a8)
     
   * Recall mengukur sejauh mana model dapat mendeteksi semua instans positif. Ini memberikan informasi tentang berapa persen dari seluruh kelas positif yang berhasil diidentifikasi oleh model.
   
4.  *F1-Sore*
    * Formula:
      
      ![image](https://github.com/farhanriyandi/Predicting-Hotel-Booking-Cancellations/assets/67671418/1b67a285-181d-47e6-ba16-fbb6a17550e4)

    * F1-Score adalah harmonic mean dari presisi dan recall. Ini memberikan keseimbangan antara presisi dan recall. F1-Score tinggi menunjukkan bahwa model memiliki keseimbangan yang baik antara presisi dan 
      recall. 


Evaluasi pada model dengan semua fitur:

![image](https://github.com/farhanriyandi/Predicting-Hotel-Booking-Cancellations/assets/67671418/5991ea7a-7be2-4ceb-b4a4-725bb3ad4182)

Evaluasi dengan 3 fitur (*reservation_status_date, arrival_date_week_number, arrival_date_year*):

![image](https://github.com/farhanriyandi/Predicting-Hotel-Booking-Cancellations/assets/67671418/8de81c89-6a6c-4949-8406-9657f5ac1bb2)


Berdasarkan hasil diatas model Random Forest dengan 3 fitur hasil *feature importance* saja yaitu reservation_status_date, arrival_date_week_number, arrival_date_year merupakan pilihan yang lebih baik karena menunjukkan performa yang baik  pada akurasi, presisi, *recall* dan *F1-score* pada kedua set data train dan test, dan cenderung menghindari overfitting yang mungkin terjadi pada model Random Forest dengan semua fitur. Dengan hasil metrik pada data uji Accuracy 0.97, pada prediksi 0 (tidak membatalkan pesanan) precision 0.96, recall 1.00 dan f1-score 0.98. Pada prediksi 1 (membatalkan pesanan)  precision 0.99, recall 0.93 dan f1-score 0.96. Dari hasil tersebut sudah memenuhi kebutuhan pengguna untuk memprediksi kemungkinan pembatalan pemesanan hotel dengan akurat

# Referensi
[1] I. G. N. Daffa Adnyana, et al., "Pengaruh Seleksi Fitur Pada Algoritma *Machine Learning* Untuk Memprediksi Pembatalan Pesanan Hotel," *Prosiding* Seminar Nasional Mahasiswa Bidang Ilmu Komputer dan Aplikasinya, vol. 2, no. 1, 2021.




