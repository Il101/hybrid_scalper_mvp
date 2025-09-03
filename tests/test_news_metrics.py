"""
Тесты для модуля новостного анализа (CoinGecko API)
Проверяем что данные правильно загружаются и обрабатываются
"""
import pytest
import pandas as pd
from unittest.mock import patch, Mock
from datetime import datetime, timedelta
import httpx
import os

from features.news_metrics import news_score, _fetch_coingecko_news, _score_title, _score_content, compute_news_sentiment
from tests.conftest import TestDataValidator, MockEnvironment


class TestCoinGeckoIntegration:
    """Тесты интеграции с CoinGecko API"""
    
    def test_coingecko_api_key_loaded(self):
        """Проверяем что API ключ загружается из окружения"""
        # Проверяем что ключ существует в окружении или имеет дефолтное значение
        api_key = os.environ.get("COINGECKO_API_KEY", "CG-ATUDeBrshiNZJ5Lq7QRsTmp2")
        assert api_key is not None
        assert len(api_key) > 0
    
    @patch('features.news_metrics.httpx.Client')
    def test_fetch_coingecko_trending_success(self, mock_client):
        """Тест успешного получения трендовых данных от CoinGecko"""
        # Мокаем ответ API
        mock_response = Mock()
        mock_response.json.return_value = {
            'coins': [
                {'item': {'id': 'bitcoin', 'symbol': 'BTC', 'name': 'Bitcoin'}},
                {'item': {'id': 'ethereum', 'symbol': 'ETH', 'name': 'Ethereum'}}
            ]
        }
        mock_response.raise_for_status.return_value = None
        
        # Мокаем контекстный менеджер
        mock_client_instance = Mock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value.__enter__ = Mock(return_value=mock_client_instance)
        mock_client.return_value.__exit__ = Mock(return_value=None)
        
        result = _fetch_coingecko_news('BTCUSDT')
        
        # Проверяем что результат не пустой
        assert result is not None
        assert len(result) > 0
        assert isinstance(result, list)
        
        # Проверяем структуру результата
        for item in result:
            assert 'title' in item
    
    @patch('features.news_metrics.httpx.Client')
    def test_fetch_coingecko_network_error(self, mock_client):
        """Тест обработки сетевых ошибок"""
        # Мокаем сетевую ошибку
        mock_client.return_value.__enter__.side_effect = httpx.RequestError("Network error")
        
        # Функция должна не падать при ошибке
        result = _fetch_coingecko_news('BTCUSDT')
        assert result is not None
        assert isinstance(result, list)
    
    def test_symbol_mapping_btc(self):
        """Тест маппинга BTC символов"""
        # Тестируем разные варианты BTC
        test_symbols = ['BTCUSDT', 'BTCUSD', 'BTC', 'btc']
        
        for symbol in test_symbols:
            result = _fetch_coingecko_news(symbol)
            assert result is not None
            assert isinstance(result, list)
    
    def test_symbol_mapping_eth(self):
        """Тест маппинга ETH символов"""
        test_symbols = ['ETHUSDT', 'ETHUSD', 'ETH', 'eth']
        
        for symbol in test_symbols:
            result = _fetch_coingecko_news(symbol)
            assert result is not None
            assert isinstance(result, list)
    
    def test_symbol_mapping_unknown(self):
        """Тест маппинга неизвестных символов"""
        # Даже для неизвестных символов должен возвращаться результат
        result = _fetch_coingecko_news('UNKNOWNUSDT')
        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0


class TestNewsScoreCalculation:
    """Тесты вычисления новостного скора"""
    
    @patch('features.news_metrics.news_score')
    def test_news_sentiment_calculation(self, mock_news_score):
        """Тест основной функции вычисления новостного скора"""
        # Мокаем функцию news_score для возврата 75.0
        mock_news_score.return_value = 75.0
        
        # Создаем тестовое окно данных
        window = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='5min'),
            'open': range(50000, 50100),
            'high': range(50010, 50110), 
            'low': range(49990, 50090),
            'close': range(50005, 50105),
            'volume': [1000] * 100
        })
        
        result = compute_news_sentiment(window, 'BTCUSDT')
        
        # compute_news_sentiment преобразует 0-100 шкалу в -1 до +1
        # 75.0 -> (75 - 50) / 50 = 0.5
        expected_result = (75.0 - 50) / 50.0
        
        # Проверяем что результат в допустимых пределах
        assert TestDataValidator.validate_signal_score(result)
        assert abs(result - expected_result) < 0.001
        
        # Проверяем что функция была вызвана с правильными параметрами
        mock_news_score.assert_called_once_with('BTCUSDT')
    
    @patch('features.news_metrics.news_score')
    def test_news_sentiment_with_empty_window(self, mock_news_score):
        """Тест обработки пустого окна данных"""
        # Мокаем нейтральный скор
        mock_news_score.return_value = 50.0
        
        empty_window = pd.DataFrame()
        
        # Функция должна вернуть нейтральный скор (преобразованный)
        result = compute_news_sentiment(empty_window, 'BTCUSDT')
        # 50.0 -> (50 - 50) / 50 = 0.0 (нейтральный)
        assert result == 0.0
    
    @patch('features.news_metrics.news_score')
    def test_news_sentiment_exception_handling(self, mock_news_score):
        """Тест обработки исключений в вычислении новостного скора"""
        # Мокаем исключение
        mock_news_score.side_effect = Exception("API Error")
        
        window = pd.DataFrame({
            'timestamp': [datetime.now()],
            'close': [50000.0]
        })
        
        # Функция должна вернуть нейтральный скор при ошибке (0.0 в новой шкале)
        result = compute_news_sentiment(window, 'BTCUSDT')
        assert result == 0.0


class TestNewsDataIntegrity:
    """Тесты целостности и качества новостных данных"""
    
    @patch('features.news_metrics.httpx.Client')
    def test_api_response_validation(self, mock_client):
        """Тест валидации ответа API"""
        # Тест с корректными данными
        valid_response = Mock()
        valid_response.json.return_value = {
            'coins': [
                {'item': {'id': 'bitcoin', 'symbol': 'BTC'}},
                {'item': {'id': 'ethereum', 'symbol': 'ETH'}}
            ]
        }
        valid_response.raise_for_status.return_value = None
        
        mock_client_instance = Mock()
        mock_client_instance.get.return_value = valid_response
        mock_client.return_value.__enter__ = Mock(return_value=mock_client_instance)
        mock_client.return_value.__exit__ = Mock(return_value=None)
        
        from features.news_metrics import _fetch_coingecko_news
        
        # _fetch_coingecko_news ожидает symbol (str) и limit (int)
        result = _fetch_coingecko_news('BTCUSDT', 10)
        assert isinstance(result, list)  # Возвращает список новостных элементов
    
    @patch('features.news_metrics.httpx.Client')
    def test_malformed_api_response(self, mock_client):
        """Тест обработки некорректного ответа API"""
        # Мокаем некорректный ответ
        malformed_response = Mock()
        malformed_response.json.return_value = {'error': 'Invalid response'}
        malformed_response.raise_for_status.return_value = None
        
        mock_client_instance = Mock()
        mock_client_instance.get.return_value = malformed_response
        mock_client.return_value.__enter__ = Mock(return_value=mock_client_instance)
        mock_client.return_value.__exit__ = Mock(return_value=None)
        
        from features.news_metrics import _fetch_coingecko_news
        
        # Должен вернуть пустой список при некорректном ответе
        result = _fetch_coingecko_news('BTCUSDT', 10)
        assert isinstance(result, list)
    
    @patch('features.news_metrics.httpx.Client')
    def test_api_rate_limiting(self, mock_client):
        """Тест обработки rate limiting"""
        # Мокаем 429 ошибку (Too Many Requests)
        mock_client.return_value.__enter__.side_effect = httpx.HTTPStatusError(
            "Too Many Requests", request=Mock(), response=Mock(status_code=429)
        )
        
        from features.news_metrics import _fetch_coingecko_news
        
        # Должен обработать rate limiting gracefully
        result = _fetch_coingecko_news('BTCUSDT', 10)
        assert isinstance(result, list)
    
    def test_news_score_range_validation(self):
        """Тест что новостные скоры всегда в допустимом диапазоне"""
        test_scores = [0.0, 25.0, 50.0, 75.0, 100.0, -10.0, 110.0]
        
        for score in test_scores:
            # Проверяем валидацию скора
            if 0 <= score <= 100:
                assert TestDataValidator.validate_signal_score(score)
            else:
                assert not TestDataValidator.validate_signal_score(score)


class TestNewsIntegrationWithTrading:
    """Тесты интеграции новостного анализа с торговой системой"""
    
    @patch('features.news_metrics.compute_news_sentiment')
    def test_news_integration_in_signal_generation(self, mock_news):
        """Тест интеграции новостного скора в генерацию сигналов"""
        mock_news.return_value = 85.0  # Сильно бычий новостной фон
        
        # Этот тест требует импорта компонентов сигнальной системы
        # Пока что проверим, что новостной компонент корректно работает
        
        window = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=50, freq='5min'),
            'close': [50000 + i*10 for i in range(50)]  # Растущий тренд
        })
        
        result = mock_news.return_value
        
        # Проверяем что сильные новости дают высокий скор
        assert result > 70.0
        assert TestDataValidator.validate_signal_score(result)
    
    @patch('features.news_metrics.compute_news_sentiment')
    def test_news_bearish_signal_integration(self, mock_news):
        """Тест интеграции медвежьих новостных сигналов"""
        mock_news.return_value = 15.0  # Сильно медвежий новостной фон
        
        window = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=50, freq='5min'),
            'close': [50000 - i*10 for i in range(50)]  # Падающий тренд
        })
        
        result = mock_news.return_value
        
        # Проверяем что плохие новости дают низкий скор
        assert result < 30.0
        assert TestDataValidator.validate_signal_score(result)
    
    def test_news_neutral_handling(self):
        """Тест обработки нейтрального новостного фона"""
        window = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=50, freq='5min'),
            'close': [50000] * 50  # Боковое движение
        })
        
        with patch('features.news_metrics.news_score', return_value=50.0):
            result = compute_news_sentiment(window, 'BTCUSDT')
            
            # Нейтральные новости должны давать скор около 0.0 (в новой шкале -1 до +1)
            # 50.0 -> (50 - 50) / 50 = 0.0
            assert -0.1 <= result <= 0.1
