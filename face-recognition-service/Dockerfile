FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копирование зависимостей
COPY requirements.txt .

# Установка Python зависимостей
# insightface тянет onnxruntime, заменяем на onnxruntime-openvino с поддержкой OpenVINO
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y onnxruntime \
    && pip install --no-cache-dir onnxruntime-openvino

# Копирование приложения
COPY . .

# Создание директории для моделей
RUN mkdir -p models

# Открытие порта
EXPOSE 8000

# Запуск без --reload для прода
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
