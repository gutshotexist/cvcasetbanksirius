import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from PIL import Image
from ultralytics import YOLO
import torch

# --- 1. Определение Pydantic моделей согласно контракту ---

class BoundingBox(BaseModel):
    """Абсолютные координаты BoundingBox"""
    x_min: int = Field(..., description="Левая координата", ge=0)
    y_min: int = Field(..., description="Верхняя координата", ge=0)
    x_max: int = Field(..., description="Правая координата", ge=0)
    y_max: int = Field(..., description="Нижняя координата", ge=0)

class Detection(BaseModel):
    """Результат детекции одного логотипа"""
    bbox: BoundingBox = Field(..., description="Результат детекции")

class DetectionResponse(BaseModel):
    """Ответ API с результатами детекции"""
    detections: List[Detection] = Field(..., description="Список найденных логотипов")

class ErrorResponse(BaseModel):
    """Ответ при ошибке"""
    error: str = Field(..., description="Описание ошибки")
    detail: Optional[str] = Field(None, description="Дополнительная информация")

# --- 2. Инициализация FastAPI и загрузка модели ---

app = FastAPI(title="T-Bank Logo Detection API")

# Загружаем нашу лучшую модель. Это происходит один раз при запуске.
model_path = "runs/final_model/weights/best.pt"
try:
    model = YOLO(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- 3. Реализация эндпоинта /detect ---

@app.post("/detect", response_model=DetectionResponse)
async def detect_logo(file: UploadFile = File(...)):
    """
    Детекция логотипа Т-банка на изображении

    Args:
        file: Загружаемое изображение (JPEG, PNG, BMP, WEBP)

    Returns:
        DetectionResponse: Результаты детекции с координатами найденных логотипов
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    # Проверяем формат файла
    if file.content_type not in ["image/jpeg", "image/png", "image/bmp", "image/webp"]:
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload JPEG, PNG, BMP, or WEBP.")

    # Чтение и обработка изображения
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Запуск детекции
    results = model.predict(image, verbose=False)
    
    # Форматирование результата
    detections = []
    # results[0].boxes.xyxy возвращает [x_min, y_min, x_max, y_max] в виде тензора
    for box in results[0].boxes.xyxy:
        # Конвертируем координаты в целые числа
        x_min, y_min, x_max, y_max = map(int, box)
        
        # Создаем BoundingBox и Detection объекты
        bbox = BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
        detections.append(Detection(bbox=bbox))
        
    return DetectionResponse(detections=detections)

# --- 4. Добавляем корневой эндпоинт для проверки ---
@app.get("/")
def read_root():
    return {"status": "T-Bank Logo Detection API is running."}
