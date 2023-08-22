
# Laporan Proyek Machine Learning - Jauza Krito

## Project Overview

_Sistem rekomendasi_ adalah sebuah sistem yang bertujuan untuk memprediksi sejumlah item atau data yang mungkin disukai oleh pengguna berdasarkan aktivitas masa lalu, dan kemudian menghadirkan daftar rekomendasi atas item tersebut. Umumnya, sistem rekomendasi telah menjadi hal yang lazim dalam mengelola informasi dengan memberikan saran kepada pengguna mengenai produk-produk yang paling relevan. Sistem-sistem serupa juga telah diterapkan dalam produk-produk media, platform-platform film, dan aspek-aspek komersial.

Beberapa platform yang menawarkan layanan film, seperti Vidio, Netflix, WeTV, Viu, dan lainnya, menggunakan sistem rekomendasi yang serupa. Sistem rekomendasi yang dibangun ini memberikan rekomendasi kepada pengguna berdasarkan preferensi genre yang disukai oleh pengguna, serta nilai _rating_ dari film-film tersebut. Hasil akhir yang diharapkan dari sistem rekomendasi ini adalah membantu pengguna dalam menemukan film-film yang sesuai dengan keinginan mereka, baik itu berdasarkan kesamaan preferensi film atau rekomendasi berdasarkan nilai _rating_.

## Business Understanding

### Problem Statements
Bagaimana caranya memberikan rekomendasi film yang disukai oleh pengguna?

### Goals
Untuk mengatasi permasalahan yang telah dijelaskan dalam bagian _Problem Statements_, maka dibangunlah sistem rekomendasi yang mampu memberikan rekomendasi film berdasarkan _ratings_ dan aktivitas pengguna di masa lalu.

### Solution statements
Solusi yang diusulkan adalah dengan menerapkan 1 algoritma machine learning, terbatas pada **Content Based Filtering** dan **Collaborative Filtering**. Kedua algoritma ini digunakan dengan tujuan yang sama, yaitu memberikan rekomendasi film kepada pengguna. Algoritma content based filtering akan merekomendasikan film kepada pengguna berdasarkan aktivitas film yang pernah ditonton oleh pengguna di masa lalu. Sementara itu, algoritma collaborative filtering akan memberikan rekomendasi kepada pengguna berdasarkan rating tertinggi.

- **Content Based Filtering**
Algoritma Content Based Filtering adalah metode yang memanfaatkan fitur-fitur item untuk merekomendasikan item lain yang memiliki kemiripan dengan item yang disukai pengguna, berdasarkan sejarah tindakan atau umpan balik eksplisit. Algoritma ini juga menggunakan kesamaan dalam produk, layanan, atau fitur konten, serta informasi yang terkumpul mengenai pengguna untuk membuat rekomendasi.

- **Collaborative Filtering**
Algoritma Collaborative Filtering adalah pendekatan yang memanfaatkan kesamaan antara pengguna dan item secara bersama-sama untuk memberikan rekomendasi. Metode ini juga bergantung pada preferensi pengguna lain yang serupa untuk memberikan rekomendasi kepada pengguna tertentu.

## Data Understanding
Dataset yang digunakan dalam proyek machine learning terdiri dari 105.339 data penilaian (ratings) dan 10.329 data film (movies) yang diperoleh dari situs [kaggle](https://www.kaggle.com/datasets/ayushimishra2809/movielens-dataset). 

**Variabel-variabel pada Movielens Dataset adalah sebagai berikut:**

1.  movieId = ID movie
2.  title = judul movie
3.  genres = genre dari movie
4.  userId = ID user
5.  rating = rating yang diberikan oleh user terhadap movie
6.  timestamp = waktu ketika user memberikan rating


### Explanatory Data Analysis
Untuk memahami dataset `movies` dan `ratings`, dilakukan Analisis Data Univariat.

| Variabel | Mean | Std | Min | 25% | 50% | 75% | Max |
|---|---|---|---|---|---|---|---|
| movieId | 364.92 | 197.49 | 1.00 | 192.00 | 383.00 | 557.00 | 668.00 |
| rating | 13381.31 | 26170.46 | 1.00 | 1073.00 | 2497.00 | 5991.00 | 149532.00 |
| timestamp | 3.52 | 1.04 | 0.50 | 3.00 | 3.50 | 4.00 | 5.00 |

Tabel di atas menunjukkan rata-rata, standar deviasi, nilai minimum, persentil ke-25, median (persentil ke-50), persentil ke-75, dan nilai maksimum dari setiap variabel dalam dataset `ratings`. Data `rating` memiliki nilai minimum sebesar 1 dan maksimum sebesar 5, serta rata-rata sekitar 3.52.

## Data Preparation

Pada tahap persiapan data, dilakukan beberapa langkah seperti:


1.  Seleksi Data: Data yang memiliki nilai kosong dihapus. Namun, dalam dataset `ratings` dan `movies` tidak ditemukan data yang kosong, yang diverifikasi menggunakan `isnull().sum()`.
    
2.  Pembagian Data: Data dibagi menjadi data _training_ dan data _testing_ dengan rasio 80:20.
    
3.  Pengurutan Data: Data diurutkan berdasarkan `movieId` secara menaik.
    
4.  Penghapusan Duplikasi: Data dengan nilai yang sama dihapus.
    
5.  Pembobotan dengan TF-IDF: Digunakan untuk memberikan bobot pada kata-kata dalam deskripsi film.
    
6.  _Cosine Similarity_: Menggunakan metode `cosine_similarity` dari pustaka `sklearn` untuk mengukur tingkat kesamaan antara film berdasarkan vektor yang diperoleh dari pembobotan TF-IDF.


## Modeling
Dalam tahap ini, dibahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Proses ini melibatkan tiga algoritma, yaitu **Content Based Filtering** dan **Collaborative Filtering**. Tujuan akhir dari sistem rekomendasi ini adalah memudahkan pengguna dalam mencari film yang sesuai dengan preferensi mereka, baik berdasarkan kesamaan genre film maupun rekomendasi berdasarkan rating.

1.  **Content Based Filtering**: Pertama-tama, pada algoritma ini, dilakukan pembobotan pada fitur `genre` menggunakan modul `TfidfVectorizer` dari pustaka `sklearn` untuk memperoleh vektor genre yang ada. Kemudian, digunakan fungsi `cosine_similarity` dari pustaka yang sama. Fungsi `movie_recommendation` digunakan dengan parameter `movie_name` untuk membangun model. Fungsi ini juga menetapkan nilai `k = 5`, yang berarti akan menghasilkan rekomendasi 5 film teratas berdasarkan genre.
    
    Contoh film yang disukai oleh pengguna pada masa lalu:

| id | title | genre | timestamp |
|---|---|---|---|
| 113 | Toy Story (1995) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 105339.000000000 |

Film-film yang direkomendasikan melalui fungsi movie_recommendation, menghasilkan 5 film teratas berdasarkan genre yang sama:

|  | title | genre |
|---:|---:|---|
| 0 | Shrek the Third (2007) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
| 1 | Asterix and the Vikings (Astérix et les Viking... | Adventure\|Animation\|Children\|Comedy\|Fantasy |
| 2 | Boxtrolls, The (2014) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
| 3 | Adventures of Rocky and Bullwinkle, The (2000) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
| 4 | Tale of Despereaux, The (2008) | Adventure\|Animation\|Children\|Comedy\|Fantasy |

2. **Collaborative Filtering**: Dalam algoritma ini, dilakukan _training_ dan pembuatan model `RecommenderNet`. _Training_ dilakukan menggunakan optimisasi `Adam` dan evaluasi menggunakan matriks RMSE. Model `RecommenderNet` akan menghitung skor _match_ antara dua _embedding layers_ dari _user_ dan _movie_ melalui operasi `dot_product`, dan kemudian menambahkan _bias_ ke kedua lapisan tersebut. Skor _match_ ini kemudian diubah menjadi skala interval antara 0 hingga 1 melalui fungsi _sigmoid_.

Contoh film yang direkomendasikan berdasarkan _rating_ tertinggi:

| movie with high ratings from user |
|---:|
| Wings of Desire (Himmel über Berlin, Der) (1987) : Drama\|Fantasy\|Romance |
| Henry V (1989) : Action\|Drama\|Romance\|War |
| Dead Poets Society (1989) : Drama |
| Touch of Evil (1958) : Crime\|Film-Noir\|Thriller |
| Princess Mononoke (Mononoke-hime) (1997) : Action\|Adventure\|Animation\|Drama\|Fantasy |


- Film TOP 10 yang direkomendasikan:

| Top 10 movie recommendation |
|---:|
| Lawrence of Arabia (1962) : Adventure\|Drama\|War |
| Once Upon a Time in the West (C'era una volta il West) (1968) : Action\|Drama\|Western |
| Ran (1985) : Drama\|War |
| Harold and Maude (1971) : Comedy\|Drama\|Romance |
| Chinatown (1974) : Crime\|Film-Noir\|Mystery\|Thriller |
| All Quiet on the Western Front (1930) : Action\|Drama\|War |
| South Pacific (1958) : Musical\|Romance\|War |
| Witness for the Prosecution (1957) : Drama\|Mystery\|Thriller |
| Dr. Horrible's Sing-Along Blog (2008) : Comedy\|Drama\|Musical\|Sci-Fi |
| Cosmos (1980) : Documentary |


## Evaluation
### Content Based Filtering
Evaluasi yang dapat digunakan adalah matriks kepresisian (precision matrix). Kepresisian mengukur kemampuan suatu metrik untuk memberikan prediksi yang benar terhadap seluruh prediksi yang dilakukan.

Rumus perhitungan matriks kepresisian:
![pres](https://user-images.githubusercontent.com/57740421/196231953-e943707a-8221-4f64-80f7-d6826a514c58.png)

Dari rekomendasi yang telah ditampilkan pada tahap pemodelan, dapat dilihat bahwa pengguna mencari rekomendasi film terkait `Toy Story (1995)`. Sistem rekomendasi kemudian memberikan 5 film terkait yang memiliki genre serupa, yaitu `Adventure\|Animation\|Children\|Comedy\|Fantasy`. Berdasarkan rumus kepresisian di atas, dapat dilihat bahwa seluruh rekomendasi yang diberikan memiliki genre yang serupa dengan film yang dicari. Dengan demikian, nilai kepresisian sistem yang dibangun adalah 5/5 atau 100%.

### Collaborative Filtering
Metrik evaluasi yang digunakan untuk mengukur kinerja model adalah RMSE (Root Mean Squared Error). RMSE mengukur perbedaan antara nilai yang diprediksi oleh model dengan nilai yang diamati. Semakin kecil nilai RMSE, semakin baik kinerja model dalam memprediksi nilai yang sebenarnya.

Rumus perhitungan RMSE:
![image](https://user-images.githubusercontent.com/57740421/196224758-6f05beb8-a8bd-4abb-ab5c-72801d9c3b9f.png)

ket:

$\mathrm{RMSE}$	=	mean squared error

${n}$	=	_number of data points_

$Y_{i}$	=	_observed values_ atau _ground truth_ dari nilai sebenarnya.

$\hat{Y}_{i}$	=	_predicted values_ atau _estimated target values_.

Hasil dari evaluasi matriks adalah sebagai berikut:
![download](https://github.com/median91/movielens/assets/62655457/6811bc57-294c-4fc4-b01f-9571ced50b2a)

Dari visualisasi proses pelatihan model di atas, dapat dilihat bahwa grafiknya relatif halus dan model tampaknya konvergen sekitar 25 epoch. Dari proses ini, diperoleh nilai RMSE akhir sekitar 0.19 untuk data latih dan sekitar 0.21 untuk data validasi. Analisis akhir menunjukkan bahwa model cenderung mengalami overfitting, karena grafik error pada data validasi masih turun meskipun error pada data latih sudah stabil.

Referensi:

[1] [Recommender System for Movielens Datasets using an Item-based Collaborative Filtering in Python](https://www.scipublications.com/journal/index.php/ijmebac/article/view/340)

[2] [Employing opposite ratings users in a new approach to collaborative filtering](https://ijeecs.iaescore.com/index.php/IJEECS/article/viewFile/24894/15925)

[3] [The MovieLens Datasets: History and Context](https://dl.acm.org/doi/10.1145/2827872)

[4] [Movielens Dataset](https://www.kaggle.com/datasets/ayushimishra2809/movielens-dataset)
