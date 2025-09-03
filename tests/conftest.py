"""
Базовые фикстуры и утилиты для тестирования
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional, Any

# Тестовые данные для OHLCV
def generate_test_ohlcv(
    symbol: str = "BTCUSDT",
    start_price: float = 50000.0,
    bars: int = 1000,
    volatility: float = 0.02
) -> pd.DataFrame:
    """Генерирует тестовые OHLCV данные с реалистичной волатильностью"""
    
    timestamps = []
    current_time = datetime.now() - timedelta(minutes=5*bars)
    
    for i in range(bars):
        timestamps.append(current_time)
        current_time += timedelta(minutes=5)
    
    # Генерируем цены с волатильностью и трендом
    prices = []
    current_price = start_price
    
    for i in range(bars):
        # Случайное изменение цены
        change = np.random.normal(0, volatility) * current_price
        current_price += change
        
        # Генерируем OHLC из текущей цены
        high = current_price * (1 + abs(np.random.normal(0, 0.001)))
        low = current_price * (1 - abs(np.random.normal(0, 0.001)))
        close_price = current_price + np.random.normal(0, 0.0005) * current_price
        open_price = prices[-1]['close'] if prices else current_price
        
        volume = np.random.uniform(100, 1000)
        
        prices.append({
            'timestamp': timestamps[i],
            'open': open_price,
            'high': max(open_price, high, close_price),
            'low': min(open_price, low, close_price),
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(prices)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Тестовые данные для orderbook
def generate_test_orderbook(
    mid_price: float = 50000.0,
    spread_bps: float = 2.0,
    depth_levels: int = 10
) -> Dict[str, List[List[float]]]:
    """Генерирует тестовый orderbook"""
    
    spread = mid_price * (spread_bps / 10000.0)
    bid_price = mid_price - spread / 2
    ask_price = mid_price + spread / 2
    
    bids = []
    asks = []
    
    for i in range(depth_levels):
        # Bids (убывающие цены)
        bid_level_price = bid_price - (i * 0.5)
        bid_size = np.random.uniform(0.1, 5.0)
        bids.append([bid_level_price, bid_size])
        
        # Asks (возрастающие цены)  
        ask_level_price = ask_price + (i * 0.5)
        ask_size = np.random.uniform(0.1, 5.0)
        asks.append([ask_level_price, ask_size])
    
    return {
        'bids': bids,
        'asks': asks,
        'timestamp': datetime.now().timestamp() * 1000
    }

# Тестовые новостные данные
def generate_test_news_data() -> Dict[str, Any]:
    """Генерирует тестовые новостные данные от CoinGecko"""
    
    trending_coins = [
        {'id': 'bitcoin', 'name': 'Bitcoin', 'symbol': 'BTC', 'market_cap_rank': 1},
        {'id': 'ethereum', 'name': 'Ethereum', 'symbol': 'ETH', 'market_cap_rank': 2},
        {'id': 'binancecoin', 'name': 'BNB', 'symbol': 'BNB', 'market_cap_rank': 3}
    ]
    
    price_changes = {
        'bitcoin': {'usd_24h_change': np.random.uniform(-5, 5)},
        'ethereum': {'usd_24h_change': np.random.uniform(-5, 5)},
        'binancecoin': {'usd_24h_change': np.random.uniform(-5, 5)}
    }
    
    return {
        'trending': {'coins': [{'item': coin} for coin in trending_coins]},
        'price_changes': price_changes
    }

# Mock конфигурация
TEST_CONFIG = {
    "execution": {
        "spread_cap_bps": 50.0,
        "k_stop_atr": 2.0,
        "rr_tp1": 1.5,
        "tp1_frac": 0.5,
        "trail_atr_k": 1.0,
        "time_stop_bars": 20
    },
    "gates": {
        "max_spread_bps": 10.0,
        "news_blackout": False
    },
    "risk": {
        "per_trade_pct": 0.02,
        "max_position_frac": 0.1
    },
    "features": {
        "news": {
            "weight": 0.3,
            "lookback_hours": 24
        },
        "ta": {
            "weight": 0.4
        },
        "sm": {
            "weight": 0.3
        }
    }
}

@pytest.fixture
def test_config():
    """Фикстура для тестовой конфигурации"""
    return TEST_CONFIG.copy()

@pytest.fixture
def test_ohlcv_data():
    """Фикстура для тестовых OHLCV данных"""
    return generate_test_ohlcv()

@pytest.fixture
def test_orderbook_data():
    """Фикстура для тестовых данных orderbook"""
    return generate_test_orderbook()

@pytest.fixture
def test_news_data():
    """Фикстура для тестовых новостных данных"""
    return generate_test_news_data()

@pytest.fixture
def mock_api_responses():
    """Фикстура для мокирования API ответов"""
    responses = {
        'coingecko_trending': generate_test_news_data()['trending'],
        'coingecko_prices': generate_test_news_data()['price_changes'],
        'orderbook': generate_test_orderbook(),
        'ohlcv': generate_test_ohlcv()
    }
    return responses

# Утилиты для проверок
class TestDataValidator:
    """Утилиты для валидации тестовых данных"""
    
    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> bool:
        """Проверяет корректность OHLCV данных"""
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Проверяем наличие колонок
        if not all(col in df.columns for col in required_cols):
            return False
            
        # Проверяем OHLC логику
        for _, row in df.iterrows():
            if not (row['low'] <= row['open'] <= row['high'] and 
                   row['low'] <= row['close'] <= row['high']):
                return False
                
        return True
    
    @staticmethod
    def validate_orderbook(ob: Dict) -> bool:
        """Проверяет корректность orderbook данных"""
        if 'bids' not in ob or 'asks' not in ob:
            return False
            
        # Проверяем что bids убывают по цене
        if len(ob['bids']) > 1:
            for i in range(1, len(ob['bids'])):
                if ob['bids'][i][0] >= ob['bids'][i-1][0]:
                    return False
                    
        # Проверяем что asks возрастают по цене  
        if len(ob['asks']) > 1:
            for i in range(1, len(ob['asks'])):
                if ob['asks'][i][0] <= ob['asks'][i-1][0]:
                    return False
                    
        return True
    
    @staticmethod
    def validate_signal_score(score: float) -> bool:
        """Проверяет что скор сигнала в допустимом диапазоне"""
        return 0.0 <= score <= 100.0

# Контекстные менеджеры для тестов
class MockEnvironment:
    """Контекстный менеджер для изоляции тестового окружения"""
    
    def __init__(self, config: Dict = None):
        self.config = config or TEST_CONFIG
        self.patches = []
    
    def __enter__(self):
        # Мокаем конфигурацию
        config_patch = patch('utils.model_utils.load_config', return_value=self.config)
        self.patches.append(config_patch)
        config_patch.start()
        
        # Мокаем API вызовы
        api_patch = patch('requests.get')
        self.patches.append(api_patch)
        api_patch.start()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for patch_obj in self.patches:
            patch_obj.stop()

# Декораторы для тестов
def requires_api(func):
    """Декоратор для тестов, требующих API"""
    def wrapper(*args, **kwargs):
        if os.getenv('SKIP_API_TESTS'):
            pytest.skip("API tests skipped")
        return func(*args, **kwargs)
    return wrapper

def slow_test(func):
    """Декоратор для медленных тестов"""
    def wrapper(*args, **kwargs):
        if os.getenv('SKIP_SLOW_TESTS'):
            pytest.skip("Slow tests skipped")
        return func(*args, **kwargs)
    return wrapper
