# 🧪 Руководство по тестированию

Данный документ описывает комплексную систему тестов для проекта hybrid_scalper_mvp.

## 📋 Обзор тестов

### Типы тестов

1. **Тесты новостного анализа** (`test_news_metrics.py`)
   - Интеграция с CoinGecko API
   - Обработка новостных данных
   - Расчет новостного скора
   - Обработка ошибок API

2. **Тесты технического анализа** (`test_ta_indicators.py`)
   - Расчет технических индикаторов
   - Валидация OHLCV данных
   - Производительность вычислений
   - Граничные условия

3. **Тесты микроструктуры рынка** (`test_microstructure.py`)
   - Обработка данных orderbook
   - Расчет спреда и OBI
   - Анализ ликвидности
   - Валидация данных стакана

4. **Тесты симуляции** (`test_simulation.py`)
   - Торговая симуляция
   - Функциональность брокера
   - Управление позициями
   - Интеграция данных

5. **Интеграционные тесты** (`test_integration.py`)
   - End-to-end тестирование
   - Производительность системы
   - Устойчивость к ошибкам
   - Реальное время

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
# Установка зависимостей для тестирования
python run_tests.py --install-deps

# Или вручную
pip install -r test_requirements.txt
```

### 2. Запуск всех тестов

```bash
# Простой запуск
python run_tests.py

# С подробным выводом
python run_tests.py --verbose

# С анализом покрытия кода
python run_tests.py --coverage
```

### 3. Запуск конкретных типов тестов

```bash
# Только быстрые тесты
python run_tests.py --type fast

# Только юнит-тесты
python run_tests.py --type unit

# Только интеграционные тесты
python run_tests.py --type integration

# Тесты производительности
python run_tests.py --type performance
```

## 📊 Детальное использование

### Запуск конкретного файла тестов

```bash
# Тестирование новостного анализа
python run_tests.py --file test_news_metrics.py

# Тестирование симуляции
python run_tests.py --file test_simulation.py
```

### Обнаружение тестов

```bash
# Показать все доступные тесты
python run_tests.py --discover
```

### Генерация отчетов

```bash
# Полный отчет с покрытием кода
python run_tests.py --report
```

### Параллельное выполнение

```bash
# Ускоренное выполнение на многоядерных системах
python run_tests.py --parallel
```

## 🎯 Маркеры тестов

Тесты организованы по маркерам для удобной фильтрации:

- `unit` - Юнит-тесты отдельных компонентов
- `integration` - Интеграционные тесты
- `performance` - Тесты производительности
- `robustness` - Тесты устойчивости к ошибкам
- `slow` - Медленные тесты
- `api` - Тесты, требующие API доступ

### Примеры фильтрации

```bash
# Исключить медленные тесты
python run_tests.py --type fast

# Исключить тесты с API
python run_tests.py --type no-api

# Только тесты производительности
pytest -m performance

# Исключить интеграционные тесты
pytest -m "not integration"
```

## 🔧 Настройка окружения

### Переменные окружения

```bash
# Пропустить медленные тесты
export SKIP_SLOW_TESTS=1

# Пропустить тесты с API
export SKIP_API_TESTS=1

# CoinGecko API ключ для тестов
export COINGECKO_API_KEY=your_test_api_key
```

### Конфигурация pytest

Настройки в `pytest.ini`:
- Автоматическое обнаружение тестов
- Логирование для отладки
- Игнорирование предупреждений
- Маркеры для категоризации

## 📈 Анализ покрытия

### HTML отчет

```bash
python run_tests.py --coverage
# Откройте htmlcov/index.html в браузере
```

### Консольный отчет

```bash
pytest --cov=features --cov-report=term-missing
```

### Пороги покрытия

Рекомендуемые минимальные пороги:
- **Общее покрытие**: >80%
- **Критичные модули**: >90%
- **Новые функции**: >95%

## 🐛 Отладка тестов

### Детальный вывод

```bash
# Максимально подробный вывод
python run_tests.py --verbose
pytest -vvv

# Показать локальные переменные при ошибках
pytest --tb=long

# Показать stdout/stderr
pytest -s
```

### Запуск одного теста

```bash
# Конкретный тест
pytest tests/test_news_metrics.py::TestCoinGeckoIntegration::test_coingecko_api_key_loaded

# С отладочным выводом
pytest tests/test_news_metrics.py::TestCoinGeckoIntegration::test_coingecko_api_key_loaded -v -s
```

### Профилирование

```bash
# Время выполнения тестов
pytest --durations=10

# Медленные тесты
pytest --durations=0 | grep -E "slow|performance"
```

## 🚨 Непрерывная интеграция

### Pre-commit хуки

```bash
# Установка pre-commit
pip install pre-commit
pre-commit install

# Запуск тестов перед коммитом
git add -A
git commit -m "Your message"  # Автоматически запустит тесты
```

### GitHub Actions

Пример `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r test_requirements.txt
    
    - name: Run tests
      run: python run_tests.py --coverage
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## 📚 Структура тестов

```
tests/
├── __init__.py                 # Тестовый пакет
├── conftest.py                # Фикстуры и утилиты
├── test_news_metrics.py       # Тесты новостного анализа
├── test_ta_indicators.py      # Тесты технического анализа  
├── test_microstructure.py     # Тесты микроструктуры
├── test_simulation.py         # Тесты симуляции
├── test_integration.py        # Интеграционные тесты
└── reports/                   # Отчеты тестирования
    ├── test_report.html
    └── coverage/
```

## 🔍 Лучшие практики

### Написание тестов

1. **Именование**: Тесты должны ясно описывать что проверяют
2. **Изоляция**: Каждый тест независим
3. **Детерминированность**: Результат предсказуем
4. **Быстрота**: Юнит-тесты выполняются быстро
5. **Читаемость**: Код тестов понятен

### Мокирование

```python
# Мокирование внешних API
@patch('features.news_metrics.news_score')
def test_with_mock(self, mock_news):
    mock_news.return_value = 75.0
    # Ваш тест
```

### Параметризация

```python
@pytest.mark.parametrize("symbol,expected", [
    ("BTCUSDT", "bitcoin"),
    ("ETHUSDT", "ethereum"),
])
def test_symbol_mapping(self, symbol, expected):
    result = map_symbol(symbol)
    assert result == expected
```

## ❓ Часто задаваемые вопросы

### Q: Как добавить новый тест?
A: Создайте функцию с префиксом `test_` в соответствующем файле, добавьте нужные маркеры.

### Q: Тесты падают из-за API
A: Используйте `--type no-api` или установите `SKIP_API_TESTS=1`.

### Q: Как ускорить тесты?
A: Используйте `--type fast`, `--parallel` или мокирование медленных операций.

### Q: Как проверить конкретный компонент?
A: Запустите соответствующий файл: `python run_tests.py --file test_news_metrics.py`

### Q: Что означают маркеры?
A: Маркеры группируют тесты по типам (unit, integration, performance, etc.) для удобной фильтрации.

## 📞 Поддержка

При возникновении проблем с тестами:

1. Проверьте зависимости: `python run_tests.py --install-deps`
2. Запустите с отладкой: `python run_tests.py --verbose`
3. Проверьте конкретный компонент отдельно
4. Посмотрите логи и трассировку ошибок
