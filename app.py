from flask import Flask, Response, render_template, request, jsonify
import cv2, time, csv, os
from datetime import datetime

app = Flask(__name__)

VIDEO_BACKEND = cv2.CAP_DSHOW
ABSEN_COOLDOWN_SEC = 15

last_absen_time_by_name = {}
DATA_DIR = "data"
KNOWN_DIR = os.path.join(DATA_DIR, "known_faces")
ABSEN_CSV = os.path.join(DATA_DIR, "absensi.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(KNOWN_DIR, exist_ok=True)
if not os.path.exists(ABSEN_CSV):
    with open(ABSEN_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["nama", "timestamp", "status"])

# Load Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def append_absen_csv(nama, ts, status="pending"):
    with open(ABSEN_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([nama, ts, status])

def list_pending_from_csv():
    rows = []
    with open(ABSEN_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("status") == "pending":
                rows.append(r)
    return rows

def update_status_in_csv(nama, timestamp, new_status="confirmed"):
    with open(ABSEN_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(ABSEN_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["nama","timestamp","status"])
        writer.writeheader()
        for r in rows:
            if r["nama"] == nama and r["timestamp"] == timestamp and r["status"]=="pending":
                r["status"] = new_status
            writer.writerow(r)

def generate_frames(camera_index):
    cap = cv2.VideoCapture(camera_index, VIDEO_BACKEND)
    if not cap.isOpened():
        print("[ERROR] Kamera tidak terbuka.")
        return
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60,60))

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            label = "Wajah"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # contoh absensi sederhana (tanpa pengenalan identitas)
            nama = "Orang"
            now = time.time()
            last_t = last_absen_time_by_name.get(nama,0)
            if now - last_t >= ABSEN_COOLDOWN_SEC:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                append_absen_csv(nama, ts, "pending")
                last_absen_time_by_name[nama] = now

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.route("/")
def index():
    return render_template("layout.html")

@app.route("/video_feed/<int:camera_id>")
def video_feed(camera_id):
    return Response(generate_frames(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/pending")
def pending():
    return jsonify(list_pending_from_csv())

@app.route("/confirm", methods=["POST"])
def confirm():
    data = request.json or {}
    nama = data.get("nama")
    ts = data.get("timestamp")
    if not nama or not ts:
        return jsonify({"ok":False,"msg":"Param tidak lengkap"}),400
    update_status_in_csv(nama, ts, "confirmed")
    return jsonify({"ok":True})

if __name__=="__main__":
    app.run(debug=True)
