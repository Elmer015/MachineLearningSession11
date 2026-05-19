# Spam Detector

Notebook ini membuat model machine learning untuk deteksi spam email/chat. Dataset baru yang dipakai ada di [data/email.csv](data/email.csv).

## Format data training

Gunakan file CSV dengan minimal dua kolom:

- kolom teks: `text`, `message`, `email`, `content`, atau `chat`
- kolom label: `label`, `target`, `class`, `spam`, atau `category`

Label yang didukung:

- `spam`, `1`, `yes`, `true`, `positive`
- `ham`, `0`, `no`, `false`, `negative`, `not_spam`

## Cara pakai

Buka [spam_detector.ipynb](spam_detector.ipynb) lalu jalankan cell dari atas ke bawah untuk melatih ulang model dan menyimpan artifact. Untuk UI langsung, jalankan Streamlit:

```powershell
& 'c:\MLLECSESSION11\.venv\Scripts\python.exe' -m streamlit run 'c:\MLLECSESSION11\streamlit_app.py'
```

Atau jalankan launcher batch berikut:

```bat
.\run_streamlit.bat
```

Notebook akan:

- mendeteksi format `data/email.csv` secara otomatis
- melatih model yang sesuai
- menampilkan metrik evaluasi
- menyimpan model ke `spam_model.joblib`
- menyimpan metadata ke `spam_model_metadata.json`

Streamlit app akan:

- memuat model pretrained dari artifact notebook
- memberi prediksi langsung tanpa training
- mendukung input tunggal dan batch

## Catatan model

Notebook melakukan normalisasi ringan pada teks, seperti menandai URL dan email, lalu memakai fitur kata dan karakter agar kuat untuk pesan pendek yang sering dipakai di chat atau email spam.