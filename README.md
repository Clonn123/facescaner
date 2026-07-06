# Face Recognition Service

Сервис распознавания лиц для HR-модуля. Детекция лиц из RTSP/веб-камеры, anti-spoofing, распознавание и контроль доступа через бэкенд API.

## Возможности

- Детекция лиц (SCRFD) и распознавание (ArcFace)
- Anti-spoofing: защита от фото/видео на экранах + flicker-детекция (PWM)
- RTSP камеры: автоматические воркеры для дверных камер
- IoU-трекинг лиц между кадрами
- Скользящее окно для стабильности результатов
- Интеграция с бэкендом для открытия дверей
- REST API + WebSocket
- Регистрация биометрии сотрудника и сохранения в БД его embedding

## Технологии

- **Python 3.11+**
- **FastAPI** — веб-фреймворк
- **InsightFace (ArcFace + SCRFD)** — детекция и распознавание
- **MiniFASNet (ONNX)** — anti-spoofing
- **PostgreSQL** — хранение данных (SQLAlchemy async)
- **Docker** — контейнеризация

## Быстрый старт

### 1. Настройка окружения

```bash
cd face-recognition-service
cp .env.example .env
# Отредактируйте .env
```

### 2. Запуск через Docker

```bash
docker-compose up --build
```

### 3. Локальный запуск

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

alembic upgrade head
uvicorn app.main:app --reload
```

### 4. Тестирование

Swagger: `http://localhost:8000/docs`

Тест-скрипты:
```bash
# Веб-камера (localhost)
python test_camera_local.py

# RTSP дверная камера
python test_camera_door.py --url rtsp://admin:pass@ip/cam/...
```

## API Endpoints

| Method | Endpoint | Описание |
|--------|----------|----------|
| POST | `/api/v1/users/register` | Регистрация пользователя |
| GET | `/api/v1/users/` | Список пользователей |
| POST | `/api/v1/recognize/` | Распознавание лица |
| POST | `/api/v1/liveness/verify` | Проверка liveness |
| GET | `/health` | Health check |

## Переменные окружения

### База данных
| Переменная | Описание |
|------------|----------|
| `DATABASE_URL` | URL подключения к PostgreSQL |

### Распознавание
| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `FACE_RECOGNITION_THRESHOLD` | `0.35` | Порог cosine similarity |
| `FACE_DETECTION_THRESHOLD` | `0.5` | Порог det_score |
| `LIVENESS_THRESHOLD` | `0.7` | Порог anti-spoof |

### Камерный пайплайн
| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `DETECT_INTERVAL` | `20` | Детекция лиц каждые N кадров |
| `SPOOF_INTERVAL` | `3` | Anti-spoof каждые N кадров |
| `REAL_WINDOW_SIZE` | `4` | Сколько подряд REAL для распознавания |
| `MOTION_WINDOW` | `3` | Окно для определения движения |
| `MOTION_THRESHOLD` | `1.0` | Порог движения (ниже = статичное лицо) |
| `RECOGNIZE_RETRY_DELAY` | `5` | Задержка повторной попытки (сек) |
| `CANDIDATES_REFRESH` | `30` | Обновление кандидатов из БД (сек) |

### Контроль доступа
| Переменная | Описание |
|------------|----------|
| `BACKEND_API_BASE_URL` | URL бэкенда (например `http://hr-core:3000`) |
| `HR_API_KEY` | API ключ для бэкенда |
| `CAMERA_URL` | `local` для веб-камеры или RTSP URL |
| `CAMERAS_ENABLED` | Запуск RTSP воркеров |
| `DOOR_ENABLED` | Включить открытие дверей |

## Пайплайн обработки кадра

```
Кадр → Детекция лиц (каждые DETECT_INTERVAL кадров)
     → IoU-трекинг (сопоставление с предыдущими лицами)
     → Anti-spoof (каждые SPOOF_INTERVAL кадров)
       + Motion detection (каждый кадр)
       + Flicker detection (каждый кадр)
     → Скользящее окно (REAL_WINDOW_SIZE подряд)
     → Распознавание → POST на бэкенд → Открытие двери
```

## Структура проекта

```
face-recognition-service/
├── app/
│   ├── api/                  # API маршруты
│   │   ├── recognize.py      # Распознавание
│   │   ├── users.py          # CRUD пользователей
│   │   ├── liveness.py       # Проверка liveness
│   │   └── camera.py         # WebSocket
│   ├── core/
│   │   ├── config.py         # Настройки (pydantic)
│   │   └── database.py       # SQLAlchemy async engine
│   ├── models/
│   │   ├── database_models.py  # UserBiometric, DoorAccessDoor
│   │   └── schemas.py         # Pydantic схемы
│   └── services/
│       ├── face_detector.py      # SCRFD детекция
│       ├── face_recognizer.py    # ArcFace распознавание
│       ├── model_singletons.py   # Общие модели
│       ├── storage.py            # Работа с БД
│       ├── camera_worker.py      # Воркер RTSP камеры
│       ├── rtsp_connection.py    # RTSP подключение
│       ├── cache.py              # Кеш дверей
│       └── liveness/
│           ├── anti_spoof_onnx.py  # MiniFASNet
│           ├── flicker_detector.py # PWM flicker
│           ├── preprocess.py       # Preprocessing
│           └── inference.py        # Logits processing
├── test_camera_local.py      # Тест веб-камеры
├── test_camera_door.py       # Тест RTSP камеры
├── Dockerfile
├── docker-compose.yml        # Dev
├── docker-compose.prod.yml   # Production
├── .env
├── .env.prod.example
└── requirements.txt
```

## Deploy

### Nginx (обратное проксирование)

```nginx
location ~ ^/face-api {
    set $upstream face-recognition:8000;
    rewrite ^/face-api/(.*) /$1 break;
    proxy_pass http://$upstream;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_http_version 1.1;
    proxy_connect_timeout 300;
    proxy_send_timeout 300;
    proxy_read_timeout 300;
}
```
