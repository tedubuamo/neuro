import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from PIL import Image, ImageTk  # Impor dari Pillow
import time

# Membuat folder untuk menyimpan gambar jika belum ada
save_dir = 'captured_images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Membuat folder untuk ekspresi emosi yang berbeda
for emotion in ['Netral', 'Senang', 'Sedih']:
    emotion_dir = os.path.join(save_dir, emotion)
    if not os.path.exists(emotion_dir):
        os.makedirs(emotion_dir)

# Fungsi untuk mengambil gambar dari webcam
def capture_image(cap):
    # Menggunakan OpenCV untuk menangkap gambar
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat membaca frame.")
        return None

    # Menampilkan gambar di layar
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame  # Mengembalikan gambar

# Fungsi untuk menyimpan gambar ke dalam folder sesuai ekspresi
def save_image(image, emotion):
    if image is not None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Menyimpan gambar di folder yang sesuai dengan ekspresi
        image_filename = os.path.join(save_dir, emotion, f"{emotion}_{timestamp}.jpg")
        cv2.imwrite(image_filename, image)
        return image_filename
    return None

# Fungsi untuk memilih ekspresi dan menyimpan gambar
def choose_emotion_and_save():
    # Membuka kamera
    cap = cv2.VideoCapture(0)  # 0 untuk webcam default
    if not cap.isOpened():
        print("Tidak dapat membuka kamera.")
        return

    # Menampilkan jendela Tkinter untuk memilih ekspresi
    window = tk.Tk()
    window.title("Pilih Ekspresi Wajah")
    
    # Label untuk instruksi
    label = tk.Label(window, text="Pilih ekspresi wajah:")
    label.pack(pady=10)
    
    # Pilihan ekspresi
    emotion = tk.StringVar(value="Netral")
    emotions = ['Netral', 'Senang', 'Sedih']
    combo_box = ttk.Combobox(window, textvariable=emotion, values=emotions)
    combo_box.pack(pady=10)

    captured_image = None  # Variabel untuk menyimpan gambar yang diambil

    # Menampilkan feed kamera dan memperbarui gambar
    def update_camera_feed():
        nonlocal captured_image
        captured_image = capture_image(cap)  # Ambil gambar setiap kali
        if captured_image is not None:
            frame_rgb = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
            # Mengubah gambar menjadi format yang bisa ditampilkan di Tkinter
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(img)
            label_video.config(image=img_tk)
            label_video.image = img_tk
        window.after(10, update_camera_feed)  # Terus memperbarui feed kamera

    # Tombol untuk menyimpan gambar
    def save_action():
        selected_emotion = emotion.get()
        if captured_image is not None:
            saved_image_path = save_image(captured_image, selected_emotion)
            if saved_image_path:
                print(f"Gambar berhasil disimpan di {saved_image_path}")
                window.quit()  # Menutup jendela setelah menyimpan
            else:
                print("Terjadi kesalahan saat menyimpan gambar.")
    
    save_button = tk.Button(window, text="Simpan Gambar", command=save_action)
    save_button.pack(pady=20)

    # Label untuk menampilkan feed kamera
    label_video = tk.Label(window)
    label_video.pack()

    # Mulai feed kamera
    window.after(10, update_camera_feed)  # Mulai memperbarui feed kamera

    # Menjalankan aplikasi Tkinter
    window.mainloop()

    # Menutup kamera dan jendela OpenCV setelah proses selesai
    cap.release()
    cv2.destroyAllWindows()

# Menjalankan fungsi untuk memilih ekspresi dan menyimpan gambar
choose_emotion_and_save()
