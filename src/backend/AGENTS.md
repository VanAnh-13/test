# AGENTS.md — HAutoML Backend Coding Standards

## 🎯 Mục tiêu

File này quy định các tiêu chuẩn lập trình **BẮT BUỘC** cho toàn bộ backend HAutoML.
Mọi code mới phải tuân thủ 100%. Mọi refactor phải cải thiện độ tuân thủ.

Agent AI **KHÔNG ĐƯỢC** tạo code vi phạm các quy tắc này.

---

## 1. SOLID Principles — BẮT BUỘC

### 1.1 Single Responsibility Principle (SRP)
✅ **BẮT BUỘC**: Mỗi class/function chỉ có MỘT lý do để thay đổi

**Sai ❌**:
```python
class UserService:
    def create_user(self, data):
        # Xác thực
        # Lưu database
        # Gửi email
        # Log
        pass
```

**Đúng ✅**:
```python
class UserService:
    def __init__(self, repo: UserRepository, email: EmailService, logger: Logger):
        self._repo = repo
        self._email = email
        self._logger = logger
    
    def create_user(self, data: UserCreateDTO) -> User:
        user = self._repo.save(User.from_dto(data))
        self._email.send_welcome(user.email)
        self._logger.info(f"User created: {user.id}")
        return user
```

### 1.2 Open/Closed Principle (OCP)
✅ **BẮT BUỘC**: Mở cho mở rộng, đóng cho sửa đổi

**Sai ❌**:
```python
def get_search_strategy(strategy_type: str):
    if strategy_type == "grid":
        return GridSearch()
    elif strategy_type == "random":
        return RandomSearch()
    elif strategy_type == "bayesian":  # Phải sửa code gốc!
        return BayesianSearch()
```

**Đúng ✅**:
```python
from abc import ABC, abstractmethod
from typing import Protocol

class SearchStrategy(Protocol):
    def search(self, space: dict) -> dict:
        ...

class SearchStrategyFactory:
    _strategies: dict[str, type[SearchStrategy]] = {}
    
    @classmethod
    def register(cls, name: str, strategy_class: type[SearchStrategy]):
        cls._strategies[name] = strategy_class
    
    @classmethod
    def create(cls, name: str) -> SearchStrategy:
        if name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {name}")
        return cls._strategies[name]()
```

### 1.3 Liskov Substitution Principle (LSP)
✅ **BẮT BUỘC**: Subclass phải thay thế được base class mà không vi phạm hợp đồng

**Sai ❌**:
```python
class BaseModel:
    def train(self, X, y):
        return self._fit(X, y)

class UnsupervisedModel(BaseModel):
    def train(self, X, y):
        raise NotImplementedError("Unsupervised không dùng y")  # Vi phạm LSP!
```

**Đúng ✅**:
```python
class SupervisedModel(Protocol):
    def train(self, X: np.ndarray, y: np.ndarray) -> "SupervisedModel":
        ...

class UnsupervisedModel(Protocol):
    def train(self, X: np.ndarray) -> "UnsupervisedModel":
        ...
```

### 1.4 Interface Segregation Principle (ISP)
✅ **BẮT BUỘC**: Client không phụ thuộc vào interface không dùng

**Sai ❌**:
```python
class DataStore(ABC):
    @abstractmethod
    def read(self): ...
    @abstractmethod
    def write(self, data): ...
    @abstractmethod
    def delete(self): ...
    # ReadOnlyDataStore vẫn phải implement write/delete!
```

**Đúng ✅**:
```python
class Readable(Protocol):
    def read(self) -> Any: ...

class Writable(Protocol):
    def write(self, data: Any) -> None: ...

class Deletable(Protocol):
    def delete(self) -> None: ...

class DataStore(Readable, Writable, Deletable):
    pass

class ReadOnlyDataStore(Readable):  # Chỉ implement read
    pass
```

### 1.5 Dependency Inversion Principle (DIP)
✅ **BẮT BUỘC**: Phụ thuộc vào abstraction, không phụ thuộc vào implementation

**Sai ❌**:
```python
class TrainingService:
    def __init__(self):
        self.db = MongoDatabase()  # Phụ thuộc trực tiếp vào MongoDB!
```

**Đúng ✅**:
```python
from typing import Protocol

class JobRepository(Protocol):
    def save(self, job: Job) -> str: ...
    def get(self, job_id: str) -> Job | None: ...

class TrainingService:
    def __init__(self, repo: JobRepository):
        self._repo = repo  # Phụ thuộc vào interface!
    
    def submit_job(self, job: Job) -> str:
        return self._repo.save(job)
```

---

## 2. Clean Code — BẮT BUỘC

### 2.1 Naming Conventions

✅ **BẮT BUỘC**:
- `snake_case` cho functions, variables, modules
- `PascalCase` cho Classes
- `SCREAMING_SNAKE_CASE` cho constants
- Tên phải **mô tả chính xác** mục đích

**Sai ❌**:
```python
def proc_data(d):  # Tên mơ hồ
    tmp = []  # tmp là gì?
    for x in d:  # x là gì?
        tmp.append(x * 2)
    return tmp
```

**Đúng ✅**:
```python
def normalize_feature_values(raw_features: list[float]) -> list[float]:
    normalized_values = []
    for feature_value in raw_features:
        normalized_values.append(feature_value * NORMALIZATION_FACTOR)
    return normalized_values
```

### 2.2 Function Rules

✅ **BẮT BUỘC**:
1. **Nhỏ** (< 30 dòng, lý tưởng < 15 dòng)
2. **Làm MỘT việc**
3. **Một mức trừu tượng duy nhất**
4. **Ít tham số** (≤ 3, dùng DTO nếu cần nhiều hơn)

**Sai ❌**:
```python
def process_training_job(job_id, dataset_id, model_type, 
                         hyperparams, search_algo, cv_folds,
                         eval_metric, timeout, callbacks):
    # 100+ dòng code với nhiều mức trừu tượng...
    pass
```

**Đúng ✅**:
```python
@dataclass
class TrainingConfig:
    dataset_id: str
    model_type: str
    hyperparameters: dict
    search_algorithm: str
    cv_folds: int
    evaluation_metric: str
    timeout_seconds: int
    
def process_training_job(job_id: str, config: TrainingConfig) -> Job:
    dataset = _load_dataset(config.dataset_id)
    model = _create_model(config.model_type)
    search_space = _build_search_space(config.hyperparameters)
    return _execute_training(job_id, dataset, model, search_space, config)
```

### 2.3 Comments & Documentation

✅ **BẮT BUỘC**:
- Code phải **tự giải thích** (self-documenting)
- Comment chỉ giải thích **TẠI SAO**, không giải thích **CÁI GÌ**
- Docstring cho mọi public function/class (Google style)

**Sai ❌**:
```python
def calc(x, y):  # Tính toán
    return x * y + 10  # Nhân x với y rồi cộng 10
```

**Đúng ✅**:
```python
def calculate_adjusted_score(raw_score: float, weight: float) -> float:
    """Calculate weighted score with baseline offset.
    
    Uses domain-specific formula: score = raw * weight + BASELINE_OFFSET
    where BASELINE_OFFSET compensates for historical data bias.
    
    Args:
        raw_score: Unweighted performance metric (0.0 to 1.0)
        weight: Importance multiplier for this metric
        
    Returns:
        Adjusted score accounting for weight and baseline offset
    """
    return raw_score * weight + BASELINE_OFFSET
```

---

## 3. NO DUPLICATION — BẮT BUỘC

✅ **BẮT BUỘC**: DRY (Don't Repeat Yourself) — KHÔNG lặp code

**Sai ❌**:
```python
# automl/v2/master.py
def preprocess_classification(df):
    df = df.dropna()
    df = df.drop_duplicates()
    # ... 20 dòng xử lý

# automl/v2/service.py
def preprocess_regression(df):
    df = df.dropna()  # Lặp lại!
    df = df.drop_duplicates()  # Lặp lại!
    # ... 20 dòng xử lý giống nhau
```

**Đúng ✅**:
```python
# automl/pipeline/preprocessor.py
class DataPreprocessor:
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna()
        df = df.drop_duplicates()
        return df
    
    def preprocess_for_task(self, df: pd.DataFrame, task: TaskType) -> pd.DataFrame:
        df = self.clean_data(df)
        if task == TaskType.CLASSIFICATION:
            return self._encode_categorical(df)
        else:
            return self._scale_numerical(df)
```

---

## 4. Separation of Concerns — BẮT BUỘC

✅ **BẮT BUỘC**: Tách biệt rõ ràng các layer

```
┌─────────────────────────────────────────┐
│  API Layer (api/*)                      │  ← HTTP, routing, validation
│  - Nhận request                         │
│  - Validate input                       │
│  - Gọi service                          │
│  - Return response                      │
└─────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  Service Layer (*/service.py)           │  ← Business logic
│  - Orchestrate use cases                │
│  - Transaction boundaries               │
│  - Business rules                       │
└─────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  Repository Layer (database/*)          │  ← Data access
│  - CRUD operations                      │
│  - Query building                       │
│  - ORM/raw queries                      │
└─────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  Infrastructure (infrastructure/*)      │  ← External systems
│  - Kafka, MinIO, SMTP                   │
│  - Third-party APIs                     │
└─────────────────────────────────────────┘
```

**Sai ❌**:
```python
# api/v1/training.py
@router.post("/train")
def train_model(request: TrainRequest):
    # Business logic trong API handler!
    dataset = mongo.collection.find_one({"_id": request.dataset_id})
    if dataset["task_type"] == "classification":
        models = ["RandomForest", "XGBoost"]
    # ... 50 dòng logic
```

**Đúng ✅**:
```python
# api/v1/training.py
@router.post("/train")
def train_model(request: TrainRequest, service: TrainingService = Depends()):
    job = service.submit_training_job(request)
    return JobResponse.from_domain(job)

# automl/v2/service.py
class TrainingService:
    def __init__(self, repo: JobRepository, dataset_repo: DatasetRepository):
        self._repo = repo
        self._dataset_repo = dataset_repo
    
    def submit_training_job(self, request: TrainRequest) -> Job:
        dataset = self._dataset_repo.get_by_id(request.dataset_id)
        job = Job.create_from_request(request, dataset)
        return self._repo.save(job)
```

---

## 5. Package Structure — BẮT BUỘC

✅ **BẮT BUỘC**: Cấu trúc thư mục nghiêm ngặt

```
src/backend/
├── api/                    # API Layer
│   ├── v1/                 # Versioned endpoints
│   │   ├── training.py     # Training endpoints
│   │   ├── datasets.py     # Dataset endpoints
│   │   └── inference.py    # Inference endpoints
│   ├── deps.py             # Dependency injection
│   └── experiment.py       # Legacy (đang deprecate)
│
├── server/                 # Application entry point
│   └── application.py      # FastAPI app setup
│
├── config/                 # Configuration
│   ├── settings.py         # Environment config
│   └── providers.py        # Service providers
│
├── automl/                 # AutoML domain
│   ├── v2/                 # New architecture
│   │   ├── service.py      # Training orchestration
│   │   ├── master.py       # Distributed training
│   │   └── schemas.py      # Domain DTOs
│   ├── pipeline/           # Training pipeline
│   │   ├── preprocessor.py
│   │   ├── trainer.py
│   │   └── evaluator.py
│   └── search/             # HPO strategies
│       ├── strategy/       # Strategy implementations
│       └── factory/        # Factory pattern
│
├── cluster/                # Worker management
│   └── worker.py           # Kafka consumer worker
│
├── database/               # Data access layer
│   ├── repositories.py     # Repository implementations
│   ├── database.py         # DB connection
│   └── get_dataset.py      # Dataset queries
│
├── infrastructure/         # External services
│   └── messaging/
│       └── kafka.py        # Kafka client
│
├── hagent/                 # AI Agent (có AGENTS.md riêng)
│   ├── agent/              # LangGraph orchestration
│   ├── bridge/             # Bridge service
│   ├── chat/               # Chat endpoints
│   └── config/             # Agent configuration
│
├── users/                  # User management
│   ├── routers.py          # User API
│   ├── engine.py           # User service
│   └── utils/              # Auth utilities
│
└── tests/                  # Test suite
    ├── test_*.py           # Unit tests
    └── integration/        # Integration tests
```

### 5.1 File Naming Rules

✅ **BẮT BUỘC**:
- `service.py` — business logic orchestration
- `repository.py` / `repositories.py` — data access
- `router.py` / `routers.py` — API endpoints
- `models.py` / `schema.py` — data models
- `types.py` — type definitions
- `errors.py` / `exceptions.py` — custom exceptions
- `config.py` / `settings.py` — configuration

❌ **KHÔNG ĐƯỢC**:
- `utils.py` trong root (quá chung chung!)
- `helpers.py` (không rõ mục đích!)
- `common.py` (vi phạm SRP!)
- Tên viết tắt không rõ nghĩa

---

## 6. Type Hints — BẮT BUỘC

✅ **BẮT BUỘC**: 100% type hints cho mọi function signature

```python
from typing import Protocol, TypeVar, Generic
from collections.abc import Sequence

# Sai ❌
def process(data):
    return data

# Đúng ✅
def process_features(data: pd.DataFrame) -> pd.DataFrame:
    return data

# Protocol cho dependency injection
class JobRepository(Protocol):
    def save(self, job: Job) -> str: ...
    def find_by_id(self, job_id: str) -> Job | None: ...

# Generic types
T = TypeVar('T')

class Repository(Generic[T], Protocol):
    def save(self, entity: T) -> str: ...
    def find_by_id(self, entity_id: str) -> T | None: ...
```

---

## 7. Error Handling — BẮT BUỘC

✅ **BẮT BUỘC**: Custom exceptions, không dùng generic Exception

**Sai ❌**:
```python
def get_dataset(dataset_id):
    dataset = db.find(dataset_id)
    if not dataset:
        raise Exception("Not found")  # Quá chung chung!
```

**Đúng ✅**:
```python
# database/errors.py
class DatasetNotFoundError(Exception):
    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        super().__init__(f"Dataset not found: {dataset_id}")

class DatasetValidationError(Exception):
    def __init__(self, dataset_id: str, reason: str):
        self.dataset_id = dataset_id
        self.reason = reason
        super().__init__(f"Invalid dataset {dataset_id}: {reason}")

# database/repositories.py
class DatasetRepository:
    def get_by_id(self, dataset_id: str) -> Dataset:
        dataset = self._collection.find_one({"_id": ObjectId(dataset_id)})
        if not dataset:
            raise DatasetNotFoundError(dataset_id)
        return Dataset.from_mongo(dataset)
```

---

## 8. Testing — BẮT BUỘC

✅ **BẮT BUỘC**: Test coverage ≥ 80% cho code mới

### 8.1 Test Structure

```python
# tests/test_training_service.py
import pytest
from unittest.mock import Mock

class TestTrainingService:
    """Test suite for TrainingService business logic."""
    
    @pytest.fixture
    def mock_repo(self):
        return Mock(spec=JobRepository)
    
    @pytest.fixture
    def service(self, mock_repo):
        return TrainingService(repo=mock_repo)
    
    def test_submit_job_creates_valid_job(self, service, mock_repo):
        # Given
        request = TrainRequest(dataset_id="ds1", model_type="rf")
        expected_job_id = "job123"
        mock_repo.save.return_value = expected_job_id
        
        # When
        job = service.submit_training_job(request)
        
        # Then
        assert job.id == expected_job_id
        assert job.status == JobStatus.PENDING
        mock_repo.save.assert_called_once()
    
    def test_submit_job_with_invalid_dataset_raises_error(self, service, mock_repo):
        # Given
        request = TrainRequest(dataset_id="invalid", model_type="rf")
        mock_repo.get_dataset.side_effect = DatasetNotFoundError("invalid")
        
        # When / Then
        with pytest.raises(DatasetNotFoundError):
            service.submit_training_job(request)
```

### 8.2 Test Naming Convention

✅ **BẮT BUỘC**: `test_<method>_<scenario>_<expected_result>`

```python
def test_submit_job_with_valid_data_creates_pending_job()
def test_submit_job_with_missing_dataset_raises_not_found()
def test_submit_job_with_invalid_model_type_raises_validation_error()
```

---

## 9. Dependency Injection — BẮT BUỘC

✅ **BẮT BUỘC**: Sử dụng constructor injection

**Sai ❌**:
```python
class TrainingService:
    def submit_job(self, job_data):
        repo = JobRepository()  # Tạo dependency trong class!
        return repo.save(job_data)
```

**Đúng ✅**:
```python
# config/providers.py
from typing import Protocol

class JobRepository(Protocol):
    def save(self, job: Job) -> str: ...

class MongoJobRepository:
    def __init__(self, db: Database):
        self._collection = db.jobs
    
    def save(self, job: Job) -> str:
        result = self._collection.insert_one(job.to_mongo())
        return str(result.inserted_id)

# api/deps.py
def get_job_repository() -> JobRepository:
    db = get_database()
    return MongoJobRepository(db)

def get_training_service(
    repo: JobRepository = Depends(get_job_repository)
) -> TrainingService:
    return TrainingService(repo=repo)

# api/v1/training.py
@router.post("/train")
def train_model(
    request: TrainRequest,
    service: TrainingService = Depends(get_training_service)
):
    job = service.submit_training_job(request)
    return JobResponse.from_domain(job)
```

---

## 10. Logging & Observability — BẮT BUỘC

✅ **BẮT BUỘC**: Structured logging với context

**Sai ❌**:
```python
print("Starting training")  # Không có context!
logger.info("Job failed")  # Thiếu thông tin!
```

**Đúng ✅**:
```python
import structlog

logger = structlog.get_logger(__name__)

class TrainingService:
    def submit_training_job(self, request: TrainRequest) -> Job:
        log = logger.bind(
            dataset_id=request.dataset_id,
            model_type=request.model_type,
            user_id=request.user_id
        )
        
        log.info("training_job_submitted", request_id=request.id)
        
        try:
            job = self._create_job(request)
            log.info("training_job_created", job_id=job.id, status=job.status)
            return job
        except DatasetNotFoundError as e:
            log.error("dataset_not_found", dataset_id=e.dataset_id)
            raise
        except Exception as e:
            log.exception("training_job_failed", error=str(e))
            raise
```

---

## 11. Configuration — BẮT BUỘC

✅ **BẮT BUỘC**: Sử dụng Pydantic Settings

```python
# config/settings.py
from pydantic_settings import BaseSettings
from pydantic import Field, MongoDsn

class Settings(BaseSettings):
    # MongoDB
    mongodb_url: MongoDsn = Field(alias="MONGODB_URL")
    mongodb_database: str = Field(default="hautoml", alias="MONGODB_DATABASE")
    
    # Kafka
    kafka_bootstrap_servers: str = Field(alias="KAFKA_BOOTSTRAP_SERVERS")
    kafka_topic_training: str = Field(default="training-jobs")
    
    # MinIO
    minio_endpoint: str = Field(alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(alias="MINIO_SECRET_KEY")
    
    # Application
    app_environment: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=False)
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Usage
settings = Settings()

# ❌ KHÔNG BAO GIỜ hardcode config!
# mongodb_url = "mongodb://localhost:27017"  # WRONG!
```

---

## 12. Code Review Checklist

Agent AI phải tự kiểm tra trước khi commit:

### ✅ SOLID
- [ ] Single Responsibility: Mỗi class/function làm 1 việc?
- [ ] Open/Closed: Dễ mở rộng mà không sửa code cũ?
- [ ] Liskov Substitution: Subclass thay thế được base class?
- [ ] Interface Segregation: Interface nhỏ, tập trung?
- [ ] Dependency Inversion: Phụ thuộc vào abstraction?

### ✅ Clean Code
- [ ] Tên biến/function mô tả rõ ràng?
- [ ] Function < 30 dòng?
- [ ] Tham số ≤ 3?
- [ ] Comment giải thích "TẠI SAO", không "CÁI GÌ"?

### ✅ NO DUPLICATION
- [ ] Không có code lặp lại?
- [ ] Logic chung đã được extract?

### ✅ Separation of Concerns
- [ ] API layer chỉ handle HTTP?
- [ ] Service layer chứa business logic?
- [ ] Repository layer chỉ access data?
- [ ] Infrastructure tách biệt external systems?

### ✅ Package Structure
- [ ] File đặt đúng thư mục?
- [ ] Tên file tuân thủ convention?
- [ ] Import path rõ ràng?

### ✅ Type Hints
- [ ] 100% type hints cho function signature?
- [ ] Dùng Protocol cho dependency injection?

### ✅ Error Handling
- [ ] Custom exceptions thay vì Exception generic?
- [ ] Error message rõ ràng, có context?

### ✅ Testing
- [ ] Test coverage ≥ 80%?
- [ ] Test name format: `test_<method>_<scenario>_<expected>`?
- [ ] Mock dependencies đúng cách?

### ✅ Dependency Injection
- [ ] Dependencies inject qua constructor?
- [ ] Không tạo dependency trong class?

### ✅ Logging
- [ ] Structured logging với context?
- [ ] Log level phù hợp (info/warning/error)?

### ✅ Configuration
- [ ] Dùng Pydantic Settings?
- [ ] Không hardcode config?
- [ ] Đọc từ environment variables?

---

## 13. Anti-Patterns — CẤM TUYỆT ĐỐI

### ❌ God Object
```python
# ❌ WRONG!
class AutoMLEngine:
    def load_data(self): ...
    def preprocess(self): ...
    def train_model(self): ...
    def evaluate(self): ...
    def deploy(self): ...
    def send_email(self): ...
    def log_metrics(self): ...
    # 1000+ dòng code!
```

### ❌ Magic Numbers
```python
# ❌ WRONG!
if accuracy > 0.85:  # 0.85 là gì?
    pass

# ✅ RIGHT!
MINIMUM_ACCEPTABLE_ACCURACY = 0.85
if accuracy > MINIMUM_ACCEPTABLE_ACCURACY:
    pass
```

### ❌ Mutable Default Arguments
```python
# ❌ WRONG!
def train_models(models=[]):  # Bug!
    models.append("rf")
    return models

# ✅ RIGHT!
def train_models(models: list[str] | None = None) -> list[str]:
    if models is None:
        models = []
    models.append("rf")
    return models
```

### ❌ Bare except
```python
# ❌ WRONG!
try:
    train_model()
except:  # Bắt cả SystemExit, KeyboardInterrupt!
    pass

# ✅ RIGHT!
try:
    train_model()
except (ValueError, DatasetNotFoundError) as e:
    logger.error("training_failed", error=str(e))
    raise
```

---

## 14. Migration Strategy

### Refactoring Legacy Code

Khi gặp code vi phạm chuẩn:

1. **Không được refactor ngay lập tức** nếu không liên quan đến task
2. **Tạo issue** để tracking việc refactor
3. **Boy Scout Rule**: Code mới phải sạch hơn code cũ
4. **Strangler Fig Pattern**: Dần dần thay thế code cũ

### Ví dụ Migration

```python
# automl/engine.py (legacy)
# → Dần dần chuyển sang automl/v2/service.py

# Step 1: Tạo adapter
class LegacyAutoMLAdapter:
    def __init__(self, new_service: TrainingService):
        self._service = new_service
    
    def train(self, **kwargs):  # Legacy interface
        request = self._convert_to_new_request(kwargs)
        return self._service.submit_training_job(request)

# Step 2: Route traffic dần dần
if feature_flag.is_enabled("use_new_training_service"):
    service = TrainingService(...)
else:
    service = LegacyAutoMLAdapter(TrainingService(...))
```

---

## 15. Tools & Automation

### 15.1 Linting & Formatting

```bash
# pyproject.toml
[tool.ruff]
line-length = 100
select = ["E", "F", "I", "N", "W", "UP", "ANN", "S", "B", "A", "C4", "PT"]
ignore = ["ANN101", "ANN102"]  # self, cls không cần type hint

[tool.ruff.per-file-ignores]
"tests/**/*.py" = ["S101"]  # Allow assert in tests

# Chạy
ruff check src/backend/
ruff format src/backend/
```

### 15.2 Type Checking

```bash
mypy src/backend/ --strict
```

### 15.3 Testing

```bash
pytest tests/ --cov=src/backend --cov-report=html --cov-fail-under=80
```

---

## 16. Tham khảo

- [Clean Code by Robert C. Martin](https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882)
- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
- [Domain-Driven Design](https://www.domainlanguage.com/ddd/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [FastAPI Best Practices](https://github.com/zhanymkanov/fastapi-best-practices)

---

## 17. Liên hệ & Câu hỏi

Nếu có thắc mắc về chuẩn coding:
1. Đọc lại AGENTS.md này
2. Kiểm tra code examples trong repo
3. Hỏi trong team chat

**LƯU Ý**: Mọi PR vi phạm chuẩn này sẽ bị REJECT! 🚫
