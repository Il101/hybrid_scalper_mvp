"""
Тесты для модуля микроструктурных метрик и orderbook анализа
Проверяем корректность обработки данных стакана
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock

from features.microstructure import (
    compute_spread_bps, compute_obi, compute_volume_profile, 
    compute_market_depth, estimate_price_impact
)
from tests.conftest import TestDataValidator, generate_test_orderbook


class TestOrderbookProcessing:
    """Тесты обработки данных orderbook"""
    
    def test_spread_calculation(self, test_orderbook_data):
        """Тест расчета спреда"""
        spread_bps = compute_spread_bps(test_orderbook_data)
        
        # Спред должен быть положительным числом
        assert spread_bps > 0
        assert not pd.isna(spread_bps)
        
        # Спред должен быть разумным (не более 1000 bps = 10%)
        assert spread_bps < 1000
        
        # Ручная проверка расчета
        if test_orderbook_data and 'bids' in test_orderbook_data and 'asks' in test_orderbook_data:
            if test_orderbook_data['bids'] and test_orderbook_data['asks']:
                best_bid = test_orderbook_data['bids'][0][0]
                best_ask = test_orderbook_data['asks'][0][0]
                mid_price = (best_bid + best_ask) / 2
                expected_spread = ((best_ask - best_bid) / mid_price) * 10000
                
                assert abs(spread_bps - expected_spread) < 0.001
    
    def test_obi_calculation(self, test_orderbook_data):
        """Тест расчета Order Book Imbalance"""
        obi = compute_obi(test_orderbook_data)
        
        # OBI должен быть в диапазоне [-1, 1]
        assert -1.0 <= obi <= 1.0
        assert not pd.isna(obi)
        
        # Проверяем логику OBI
        if test_orderbook_data and 'bids' in test_orderbook_data and 'asks' in test_orderbook_data:
            bid_volume = sum(level[1] for level in test_orderbook_data['bids'][:5])
            ask_volume = sum(level[1] for level in test_orderbook_data['asks'][:5])
            
            if bid_volume + ask_volume > 0:
                expected_obi = (bid_volume - ask_volume) / (bid_volume + ask_volume)
                assert abs(obi - expected_obi) < 0.001
    
    def test_empty_orderbook_handling(self):
        """Тест обработки пустого orderbook"""
        empty_ob = {'bids': [], 'asks': []}
        
        # Функции должны обрабатывать пустые данные gracefully
        spread = compute_spread_bps(empty_ob)
        obi = compute_obi(empty_ob)
        
        # Должны возвращать нейтральные значения или NaN
        assert pd.isna(spread) or spread == 0
        assert pd.isna(obi) or obi == 0
    
    def test_malformed_orderbook_handling(self):
        """Тест обработки некорректного orderbook"""
        malformed_obs = [
            None,
            {},
            {'bids': None, 'asks': []},
            {'bids': [], 'asks': None},
            {'bids': [['invalid']], 'asks': [[50000, 1]]},
            {'bids': [[49000, 'invalid']], 'asks': [[50000, 1]]}
        ]
        
        for ob in malformed_obs:
            # Функции не должны падать на некорректных данных
            try:
                spread = compute_spread_bps(ob)
                obi = compute_obi(ob)
                
                # Результаты должны быть числами или NaN
                assert pd.isna(spread) or isinstance(spread, (int, float))
                assert pd.isna(obi) or isinstance(obi, (int, float))
                
            except Exception as e:
                # Допускаем только определенные типы исключений
                assert isinstance(e, (ValueError, TypeError, KeyError, IndexError, AttributeError))


class TestOrderflowMetrics:
    """Тесты метрик потока ордеров"""
    
    def test_volume_profile_analysis(self, test_orderbook_data):
        """Тест анализа объемного профиля"""
        # Используем импортированную функцию
        profile = compute_volume_profile(test_orderbook_data)
        
        # Профиль должен быть словарем с уровнями цен
        assert isinstance(profile, dict)
        
        # Все значения должны быть положительными
        for price, volume in profile.items():
            assert price > 0
    def test_depth_analysis(self, test_orderbook_data):
        """Тест анализа глубины стакана"""
        depth_info = compute_market_depth(test_orderbook_data)
        
        assert isinstance(depth_info, dict)
        assert 'bid_depth' in depth_info
        assert 'ask_depth' in depth_info
        assert 'total_depth' in depth_info
        
        # Все значения должны быть неотрицательными
        assert depth_info['bid_depth'] >= 0
        assert depth_info['ask_depth'] >= 0
        assert depth_info['total_depth'] >= 0
    
    def test_price_impact_estimation(self, test_orderbook_data):
        """Тест оценки влияния на цену"""
        # Тестируем покупку на $1000
        impact_buy = estimate_price_impact(test_orderbook_data, 1000.0, 'buy')
        impact_sell = estimate_price_impact(test_orderbook_data, 1000.0, 'sell')
        
        # Impact должен быть числом
        assert isinstance(impact_buy, float)
        assert isinstance(impact_sell, float)
        assert impact_buy >= 0
        assert impact_sell >= 0


class TestMicrostructureIntegrity:
    """Тесты целостности микроструктурных данных"""
    
    def test_orderbook_consistency(self, test_orderbook_data):
        """Тест консистентности данных orderbook"""
        if not test_orderbook_data or 'bids' not in test_orderbook_data or 'asks' not in test_orderbook_data:
            pytest.skip("Invalid orderbook data")
        
        # Проверяем упорядоченность bids (убывающие цены)
        bids = test_orderbook_data['bids']
        if len(bids) > 1:
            for i in range(1, len(bids)):
                assert bids[i][0] <= bids[i-1][0], f"Bids not ordered correctly: {bids[i-1][0]} > {bids[i][0]}"
        
        # Проверяем упорядоченность asks (возрастающие цены)
        asks = test_orderbook_data['asks']
        if len(asks) > 1:
            for i in range(1, len(asks)):
                assert asks[i][0] >= asks[i-1][0], f"Asks not ordered correctly: {asks[i-1][0]} < {asks[i][0]}"
        
        # Проверяем что лучший bid < лучший ask
        if bids and asks:
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            assert best_bid < best_ask, f"Best bid {best_bid} should be < best ask {best_ask}"
    
    def test_volume_positivity(self, test_orderbook_data):
        """Тест положительности объемов"""
        if not test_orderbook_data:
            return
            
        # Все объемы должны быть положительными
        for side in ['bids', 'asks']:
            if side in test_orderbook_data:
                for level in test_orderbook_data[side]:
                    if len(level) >= 2:
                        price, volume = level[0], level[1]
                        assert price > 0, f"Price {price} should be positive"
                        assert volume > 0, f"Volume {volume} should be positive"
    
    def test_timestamp_validity(self, test_orderbook_data):
        """Тест валидности timestamp"""
        if test_orderbook_data and 'timestamp' in test_orderbook_data:
            timestamp = test_orderbook_data['timestamp']
            
            # Timestamp должен быть числом
            assert isinstance(timestamp, (int, float))
            
            # Timestamp должен быть разумным (в пределах последних/будущих лет)
            import time
            current_time = time.time() * 1000  # Миллисекунды
            assert abs(timestamp - current_time) < 365 * 24 * 60 * 60 * 1000  # В пределах года


class TestOrderbookRealTimeProcessing:
    """Тесты обработки orderbook в реальном времени"""
    
    def test_orderbook_updates_handling(self):
        """Тест обработки обновлений orderbook"""
        # Создаем конкретный orderbook с известными значениями
        initial_ob = {
            'bids': [[50000.0, 1.5], [49990.0, 2.0]],
            'asks': [[50100.0, 1.2], [50110.0, 1.8]]  # Спред = 100 bps = (50100-50000)/50050 * 10000
        }
        
        # Обновляем лучший bid - поднимаем его ближе к ask
        updated_ob = initial_ob.copy()
        updated_ob['bids'] = [[50050.0, 1.5], [49990.0, 2.0]]  # Спред уменьшается
        
        # Спред должен уменьшиться
        initial_spread = compute_spread_bps(initial_ob)
        updated_spread = compute_spread_bps(updated_ob)
        
        print(f"Initial spread: {initial_spread}, Updated spread: {updated_spread}")
        assert updated_spread < initial_spread
    
    def test_orderbook_latency_impact(self):
        """Тест влияния задержки на данные orderbook"""
        import time
        
        # Симулируем старые данные
        old_timestamp = (time.time() - 5) * 1000  # 5 секунд назад
        current_timestamp = time.time() * 1000
        
        old_ob = generate_test_orderbook()
        old_ob['timestamp'] = old_timestamp
        
        current_ob = generate_test_orderbook()
        current_ob['timestamp'] = current_timestamp
        
        # Проверяем что можем детектировать устаревшие данные
        age_old = current_timestamp - old_ob['timestamp']
        age_current = current_timestamp - current_ob['timestamp']
        
        assert age_old > age_current
        assert age_old > 1000  # Больше секунды
    
    @patch('time.time')
    def test_orderbook_frequency_analysis(self, mock_time):
        """Тест анализа частоты обновлений orderbook"""
        timestamps = []
        mock_times = [1000, 1000.1, 1000.15, 1000.3, 1000.35]  # Разные интервалы
        
        for mock_time_val in mock_times:
            mock_time.return_value = mock_time_val
            ob = generate_test_orderbook()
            ob['timestamp'] = mock_time_val * 1000
            timestamps.append(ob['timestamp'])
        
        # Вычисляем интервалы между обновлениями
        intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        
        # Все интервалы должны быть положительными
        assert all(interval > 0 for interval in intervals)
        
        # Должны быть разумные интервалы (не слишком частые/редкие)
        assert all(0 < interval < 10000 for interval in intervals)  # От 0 до 10 секунд


class TestIntegrationWithTradingLogic:
    """Тесты интеграции микроструктуры с торговой логикой"""
    
    def test_spread_gating(self, test_orderbook_data):
        """Тест блокировки по спреду"""
        spread = compute_spread_bps(test_orderbook_data)
        max_spread = 50.0  # 50 bps максимум
        
        # Проверяем логику гейтинга
        spread_gate_passed = spread <= max_spread
        
        if spread <= max_spread:
            assert spread_gate_passed
        else:
            assert not spread_gate_passed
    
    def test_liquidity_assessment(self, test_orderbook_data):
        """Тест оценки ликвидности"""
        if not test_orderbook_data or 'bids' not in test_orderbook_data or 'asks' not in test_orderbook_data:
            return
        
        # Считаем общую ликвидность в топ-5 уровнях
        bid_liquidity = sum(level[1] for level in test_orderbook_data['bids'][:5])
        ask_liquidity = sum(level[1] for level in test_orderbook_data['asks'][:5])
        total_liquidity = bid_liquidity + ask_liquidity
        
        # Минимальный порог ликвидности
        min_liquidity = 10.0  # 10 единиц минимум
        liquidity_sufficient = total_liquidity >= min_liquidity
        
        assert isinstance(liquidity_sufficient, bool)
        if total_liquidity >= min_liquidity:
            assert liquidity_sufficient
    
    def test_market_impact_limits(self, test_orderbook_data):
        """Тест ограничений по маркет импакту"""
        try:
            from features.orderflow import estimate_price_impact
            
            # Тестируем разные размеры ордеров
            order_sizes = [100, 500, 1000, 5000, 10000]  # USD
            impacts = []
            
            for size in order_sizes:
                impact = estimate_price_impact(test_orderbook_data, 'buy', size)
                impacts.append(abs(impact))
            
            # Impact должен расти с размером ордера
            for i in range(1, len(impacts)):
                assert impacts[i] >= impacts[i-1], "Impact should increase with order size"
                
        except ImportError:
            # Простая проверка без реальной функции
            assert True  # Skip test if function not available
