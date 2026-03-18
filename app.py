from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
import pytesseract
import os

app = Flask(__name__)
CORS(app)

# 👉 IMPORTANT: Tesseract path (change if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 👉 Check if Tesseract exists
if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
    print("❌ Tesseract not found! Check path.")


@app.route("/verify", methods=["POST"])
def verify():

    try:
        data = request.json
        image_data = data.get("image")

        if not image_data:
            return jsonify({"status": "error", "message": "No image received"})

        # Remove base64 header
        image_data = image_data.split(",")[1]

        # Decode image
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"status": "error", "message": "Invalid image"})

        # 🔥 Preprocessing (important for OCR accuracy)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        # 🔥 OCR
        text = pytesseract.image_to_string(
            thresh,
            config="--oem 3 --psm 6 -l eng"
        )

        print("📄 Detected Text:\n", text)

        # 🔥 Simple validation
        if len(text.strip()) > 20:
            return jsonify({
                "status": "valid",
                "text": text
            })
        else:
            return jsonify({
                "status": "invalid",
                "text": "No readable text"
            })

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({
            "status": "error",
            "message": str(e)
        })


# 👉 Run server
if __name__ == "__main__":
    app.run(debug=True)