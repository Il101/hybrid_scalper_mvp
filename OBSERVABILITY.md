# Model Observability System

Автоматическое логирование артефактов и метрик для Random Forest модели.

## Возможности

### Автоматическое логирование артефактов
- **Версионирование моделей**: каждая тренировка создает timestamped версию в `model/artifacts/version_YYYYMMDD_HHMMSS/`
- **Основные артефакты**: `meta_model.pkl`, `calibrator.pkl`, `report.json` (всегда перезаписываются последней версией)
- **Исторические версии**: полная копия артефактов в версионированных папках

### Расширенные метрики
- **Базовые**: accuracy, AUC, precision, recall, specificity, F1-score
- **Cross-validation**: полные распределения CV scores (accuracy + AUC)
- **Confusion Matrix**: детальная разбивка TP/TN/FP/FN
- **Timing**: время обучения и предсказания
- **Feature Importance**: с детальным анализом для Random Forest (individual tree importances)

### Логи наблюдаемости
- **training_log.jsonl**: компактный лог каждой тренировки (JSONL формат)
- **Comprehensive reports**: полный JSON отчет с метаданными, конфигурацией модели, статистиками данных

## Использование

### Обучение с логированием
```bash
# Стандартное обучение Random Forest
python model/train_meta.py --data data/features.parquet --outdir model/artifacts --model_type rf

# Результат:
# - model/artifacts/meta_model.pkl (latest)
# - model/artifacts/calibrator.pkl (latest) 
# - model/artifacts/report.json (comprehensive)
# - model/artifacts/training_log.jsonl (append-only log)
# - model/artifacts/version_YYYYMMDD_HHMMSS/ (versioned artifacts)
```

### Анализ истории обучений
```bash
# Краткий обзор последних тренировок
python scripts/view_model_history.py

# Детальная статистика производительности
python scripts/view_model_history.py --detailed

# Сравнение последних версий
python scripts/view_model_history.py --compare
```

## Структура артефактов

```
model/artifacts/
├── meta_model.pkl          # Последняя обученная модель
├── calibrator.pkl          # Последний калибратор
├── report.json             # Последний полный отчет
├── training_log.jsonl      # Лог всех тренировок
├── version_20250902_232453/    # Версионированные артефакты
│   ├── meta_model.pkl
│   ├── calibrator.pkl
│   └── report.json
└── version_20250902_232536/    # Новая версия
    ├── meta_model.pkl
    ├── calibrator.pkl
    └── report.json
```

## Ключевые метрики в report.json

### Производительность модели
- `performance_metrics.base_model`: метрики до калибровки
- `performance_metrics.calibrated_model`: метрики после калибровки
- `feature_analysis.importance`: важность фич
- `feature_analysis.importance_details`: детали для Random Forest (individual trees)

### Наблюдаемость
- `metadata`: версии, источники данных, временные метки
- `observability.training_time_seconds`: время обучения
- `observability.cross_validation`: полные CV результаты
- `dataset_info`: статистики датасета

## Random Forest специфичные возможности

### Детальный анализ деревьев
- Важность фич по первым 10 деревьям
- OOB score (если включен)
- Параметры модели (max_features, min_samples_split, etc.)

### Конфигурация модели
Все гиперпараметры логируются в `model_configuration`:
```json
{
  "n_estimators": 100,
  "max_depth": 10,
  "min_samples_split": 2,
  "min_samples_leaf": 1,
  "bootstrap": true,
  "random_state": 42
}
```

## Интеграция с симуляцией

Модель автоматически используется в симуляции если:
1. `config.yaml` содержит `meta_model_enabled: true`
2. Артефакты присутствуют в `model/artifacts/`

Система автоматически подгружает последние версии модели и калибратора.
