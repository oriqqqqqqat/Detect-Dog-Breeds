from flask import Flask, request, render_template
from ultralytics import YOLO
from PIL import Image
import io, base64

app = Flask(__name__)
model = YOLO(r"D:/YOLO/data/runs/detect/train/weights/best.pt")  # ปรับ path ตามเครื่องคุณ

@app.route("/", methods=["GET", "POST"])
def index():
    orig_b64 = None
    result_b64 = None
    prediction_text = ""

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            return render_template("index.html", orig_b64=None, result_b64=None, prediction="ไม่พบไฟล์รูป")

        # เก็บต้นฉบับเป็น base64 เพื่อแสดง
        file_bytes = file.read()
        orig_b64 = base64.b64encode(file_bytes).decode("utf-8")

        # เปิดภาพด้วย PIL (จากหน่วยความจำ) เป็น RGB
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        # ย่อเพื่อลดเวลา/แรม (เลือกเปิดใช้ได้)
        # img.thumbnail((1600, 1600))

        # รัน YOLO แบบ in-memory
        results = model.predict(source=img, save=False, save_txt=False, verbose=False)
        r = results[0]

        # สรุปผลตรวจจับ
        if r.boxes is not None and len(r.boxes) > 0:
            names = model.names
            parts = []
            for b in r.boxes:
                cls_id = int(b.cls)
                conf = float(b.conf)
                parts.append(f"{names[cls_id]} ({conf:.0%})")
            prediction_text = "ตรวจพบ: " + ", ".join(parts)
        else:
            prediction_text = "ไม่พบวัตถุ"

        # ทำภาพ annotated -> base64
        annotated_bgr = r.plot()                  # numpy (BGR)
        annotated_rgb = annotated_bgr[:, :, ::-1] # เป็น RGB
        buf = io.BytesIO()
        Image.fromarray(annotated_rgb).save(buf, format="JPEG", quality=90)
        result_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return render_template("index.html",
                           orig_b64=orig_b64,
                           result_b64=result_b64,
                           prediction=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
