import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, UploadFile, File
from PIL import Image
import pillow_heif  # รองรับไฟล์ HEIC

# โหลดโมเดลที่เทรนไว้
MODEL_PATH = "model/MobileNetV2_model_20250119_132758.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# ค่าคลาสที่โมเดลสามารถทำนายได้
CLASS_NAMES = ['algal_leaf_spot', 'healthy', 'leaf_blight', 'leaf_spot']

# สร้าง FastAPI app พร้อมเอกสาร API
app = FastAPI(
    title="Durian Leaf Disease Detection API",
    description="API สำหรับตรวจจับโรคในใบของทุเรียนโดยใช้โมเดล Deep Learning",
    version="1.0.0",
)

# ฟังก์ชันประมวลผลภาพก่อนป้อนเข้าโมเดล
def preprocess_image(image: Image.Image) -> np.ndarray:
    """ปรับขนาดและ normalize ภาพให้ตรงกับโมเดล"""
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ฟังก์ชันเปิดภาพ รองรับ HEIC, JPEG, PNG
def load_image(file: UploadFile) -> Image.Image:
    """โหลดไฟล์ภาพ และแปลงเป็น RGB"""
    image_data = file.file.read()
    
    try:
        if file.filename.lower().endswith(".heic"):
            heif_file = pillow_heif.open_heif(io.BytesIO(image_data))
            image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data)
        else:
            image = Image.open(io.BytesIO(image_data))

        return image.convert("RGB")
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format for {file.filename}. Error: {str(e)}")

# API สำหรับทำนาย 1 รูป
@app.post("/predict", summary="ทำนายโรคจากภาพใบของทุเรียน", description="อัปโหลดไฟล์ภาพเพื่อตรวจจับโรคในใบของทุเรียน")
async def predict_single(file: UploadFile = File(...)):
    """รับภาพเดียว วิเคราะห์ และส่งคืนผลลัพธ์การตรวจจับโรค"""
    image = load_image(file)
    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    return {"filename": file.filename, "predicted_class": predicted_class, "confidence": confidence}

# API สำหรับทำนายหลายรูป
@app.post("/predict_multiple", summary="ทำนายโรคจากภาพใบของทุเรียน (หลายไฟล์)", description="อัปโหลดหลายไฟล์พร้อมกันเพื่อตรวจจับโรคในใบของทุเรียน")
async def predict_multiple(files: list[UploadFile] = File(...)):
    """รับภาพหลายไฟล์ วิเคราะห์ และส่งคืนผลลัพธ์"""
    results = []
    
    for file in files:
        image = load_image(file)
        processed_image = preprocess_image(image)

        predictions = model.predict(processed_image)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        results.append({"filename": file.filename, "predicted_class": predicted_class, "confidence": confidence})
    
    return {"predictions": results}
