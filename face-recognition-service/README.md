# Face Recognition Service

Сервис распознавания лиц для HR-модуля.

## Возможности

- Регистрация сотрудников по фото/видео
- Распознавание лиц из видеопотока
- Anti-spoofing (защита от фотографий и видео на экранах)
- Оценка качества кадра
- REST API интеграция

## Технологии

- **Python 3.11+**
- **FastAPI** — веб-фреймворк
- **InsightFace (ArcFace + SCRFD)** — детекция и распознавание лиц
- **PostgreSQL** — хранение данных
- **Docker** — контейнеризация

## Быстрый старт

### 1. Настройка окружения

```bash
cd face-recognition-service
cp .env.example .env
# Отредактируйте .env
```

### 2. Запуск через Docker (рекомендуется)

```bash
docker-compose up --build
```

### 3. Локальный запуск

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Инициализация БД
alembic upgrade head

# Запуск
uvicorn app.main:app --reload
```

### 4. Тестирование

Откройте в браузере: `http://localhost:8000/docs`

## API Endpoints

| Method | Endpoint | Описание |
|--------|----------|----------|
| POST | `/api/v1/employees/register` | Регистрация сотрудника |
| POST | `/api/v1/recognize` | Распознавание лица |
| POST | `/api/v1/liveness/verify` | Проверка живого присутствия |
| GET | `/api/v1/employees/{id}` | Получение сотрудника |
| DELETE | `/api/v1/employees/{id}` | Удаление сотрудника |
| GET | `/health` | Проверка работоспособности |

## Переменные окружения

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `DATABASE_URL` | `postgresql://...` | URL подключения к PostgreSQL |
| `FACE_RECOGNITION_THRESHOLD` | `0.35` | Порог сходства для распознавания |
| `FACE_DETECTION_THRESHOLD` | `0.5` | Порог детекции лица |
| `LIVENESS_THRESHOLD` | `0.7` | Порог проверки живого присутствия |
| `MIN_FACE_SIZE` | `80` | Минимальный размер лица в пикселях |
| `MAX_EMBEDDING_DIM` | `512` | Размер вектора embedding |

## Структура проекта

```
face-recognition-service/
├── app/
│   ├── api/              # API маршруты
│   ├── core/             # Конфигурация, БД
│   ├── models/           # SQLAlchemy модели, Pydantic схемы
│   └── services/         # Бизнес-логика
│       ├── face_detector.py
│       ├── face_recognizer.py
│       ├── anti_spoof.py
│       └── storage.py
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── alembic.ini
└── README.md
```

## Anti-Spoofing

Сервис использует встроенную модель anti-spoofing из InsightFace (AntelopeV2).

Поддерживаемые атаки:
- Бумажные фотографии
- Фотографии на экране телефона/монитора
- Видеозаписи

Результат: `is_real` (boolean) + `liveness_score` (float 0-1).

## Распознавание

- Используется модель ArcFace с Cosine Similarity
- Порог по умолчанию: 0.35 (настраивается через env)
- Поддержка усреднения embedding по нескольким фото

## Контроль доступа (будущее)

Сервис спроектирован для лёгкой интеграции с системой контроля доступа:
- Добавление webhook/callback при успешном распознавании
- Логирует все события доступа
- Поддержка RTSP-потоков
