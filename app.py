from flask import Flask, Response, render_template, request, jsonify
import cv2, time, csv, os
from datetime import datetime
import mediapipe as mp
import math

app = Flask(__name__)

VIDEO_BACKEND = cv2.CAP_DSHOW
ABSEN_COOLDOWN_SEC = 15

last_absen_time_by_name = {}
DATA_DIR = "data"
ABSEN_CSV = os.path.join(DATA_DIR, "absensi.csv")

os.makedirs(DATA_DIR, exist_ok=True)
if not os.path.exists(ABSEN_CSV):
    with open(ABSEN_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["nama", "timestamp", "status", "emosi"])

# Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark index
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
MOUTH     = [61, 81, 311, 291, 308, 402]
CHEEK     = 50
MOUTH_CORNER = 61

def dist(a, b):
    return math.dist(a, b)

def aspect_ratio(p1, p2, p3, p4, p5, p6, landmarks):
    return (dist(landmarks[p2], landmarks[p6]) + dist(landmarks[p3], landmarks[p5])) / (
        2.0 * dist(landmarks[p1], landmarks[p4])
    )

def predict_emotion(landmarks):
    EAR_right = aspect_ratio(33, 160, 158, 133, 153, 144, landmarks)
    EAR_left  = aspect_ratio(263, 387, 385, 362, 380, 373, landmarks)
    EAR = (EAR_left + EAR_right) / 2.0

    MAR = aspect_ratio(61, 81, 311, 291, 308, 402, landmarks)

    mouth_corner = landmarks[61]
    cheek_point  = landmarks[50]
    cheek_lift = mouth_corner[1] - cheek_point[1]

    # Senang → MAR tinggi + pipi naik
    if MAR >= 0.30 and cheek_lift < 45:
        return "Senang"
    # Lelah → EAR rendah (<0.23) dan tidak senyum
    elif EAR < 0.23:
        return "Lelah"
    # Netral → sisanya
    else:
        return "Netral"


def append_absen_csv(nama, ts, status="pending", emosi="Netral"):
    with open(ABSEN_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([nama or "", ts or "", status or "", emosi or ""])

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
        writer = csv.DictWriter(f, fieldnames=["nama","timestamp","status","emosi"])
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

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                h, w, _ = frame.shape
                landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face.landmark]

                # Prediksi emosi
                emosi = predict_emotion(landmarks)

                # Gambar outline wajah penuh
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
                )

                # Gambar titik mata
                for idx in LEFT_EYE + RIGHT_EYE:
                    cv2.circle(frame, landmarks[idx], 2, (0,255,0), -1)
                # Gambar titik mulut
                for idx in MOUTH:
                    cv2.circle(frame, landmarks[idx], 2, (255,0,0), -1)

                # Label emosi
                x_min = min([p[0] for p in landmarks])
                y_min = min([p[1] for p in landmarks])
                cv2.putText(frame, f"Emosi: {emosi}", (x_min, y_min-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

                # Absensi sederhana
                nama = "Orang"
                now = time.time()
                last_t = last_absen_time_by_name.get(nama, 0)
                if now - last_t >= ABSEN_COOLDOWN_SEC:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    append_absen_csv(nama, ts, "pending", emosi)
                    last_absen_time_by_name[nama] = now

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    cap.release()

@app.route("/")
def index():
    return render_template("layout.html")

@app.route("/video_feed/<int:camera_id>")
def video_feed(camera_id):
    return Response(generate_frames(camera_id),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/pending")
def pending():
    return jsonify(list_pending_from_csv())

@app.route("/confirm", methods=["POST"])
def confirm():
    data = request.json or {}
    nama = data.get("nama")
    ts = data.get("timestamp")
    if not nama or not ts:
        return jsonify({"ok": False, "msg": "Param tidak lengkap"}), 400
    update_status_in_csv(nama, ts, "confirmed")
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
