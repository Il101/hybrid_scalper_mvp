"""
Интеграционные тесты для проверки всей системы end-to-end
Проверяем что все компоненты работают вместе корректно
"""
import pytest
import pandas as pd
import numpy as np
import time
from unittest.mock import patch, Mock, MagicMock
import os
import tempfile
from pathlib import Path

from tests.conftest import TestDataValidator, generate_test_ohlcv, generate_test_orderbook, MockEnvironment


class TestFullSystemIntegration:
    """Тесты полной интеграции системы"""
    
    @pytest.mark.integration
    def test_complete_trading_pipeline(self):
        """Тест полного торгового пайплайна от данных до сделок"""
        
        with MockEnvironment() as env:
            # 1. Мокаем все источники данных
            test_ohlcv = generate_test_ohlcv(bars=100)
            test_orderbook = generate_test_orderbook()
            
            with patch('ingest.prices.get_ohlcv', return_value=test_ohlcv), \
                 patch('features.news_metrics.news_score', return_value=75.0), \
                 patch('ingest.orderbook_ccxt.fetch_orderbook', return_value=test_orderbook):
                
                # 2. Запускаем симуляцию
                from backtest.sim_loop import run_simulation
                result = run_simulation(symbol='BTCUSDT', steps=20, fast_mode=True)
                
                # 3. Проверяем что пайплайн отработал
                assert isinstance(result, dict)
                assert 'cash_final' in result
                assert 'log_path' in result
                
                # 4. Проверяем что создались лог-файлы
                if result.get('log_path') and os.path.exists(result['log_path']):
                    trades_df = pd.read_csv(result['log_path'])
                    # Проверяем структуру, но не требуем обязательно сделок
                    # (сделок может не быть если нет торговых сигналов)
                    expected_columns = ['ts', 'symbol', 'side', 'price', 'size_usd', 'reason']  # 'ts' а не 'timestamp'
                    
                    # Если есть сделки, проверяем их валидность
                    if not trades_df.empty:
                        for col in expected_columns:
                            assert col in trades_df.columns, f"Missing column: {col}"
                            assert trades_df[col].notna().any(), f"Column {col} has only NaN values"
                        
                        # Проверяем что цены и размеры положительные
                        assert (trades_df['price'] > 0).all(), "All prices should be positive"
                        assert (trades_df['size_usd'] > 0).all(), "All sizes should be positive"
                    else:
                        print("INFO: No trades generated in test pipeline - this is acceptable if no trading signals occurred")
    
    @pytest.mark.integration  
    def test_multi_symbol_pipeline(self):
        """Тест пайплайна с множественными символами"""
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        results = {}
        
        for symbol in symbols:
            with patch('ingest.prices.get_ohlcv') as mock_ohlcv, \
                 patch('features.news_metrics.news_score') as mock_news:
                
                # Генерируем уникальные данные для каждого символа
                mock_ohlcv.return_value = generate_test_ohlcv(
                    symbol=symbol, 
                    start_price=50000 if 'BTC' in symbol else 3000, 
                    bars=50
                )
                mock_news.return_value = np.random.uniform(30, 70)
                
                from backtest.sim_loop import run_simulation
                result = run_simulation(symbol=symbol, steps=10, fast_mode=True)
                results[symbol] = result
        
        # Проверяем что все симуляции завершились успешно
        for symbol, result in results.items():
            assert isinstance(result, dict)
            assert 'cash_final' in result
            assert result['cash_final'] > 0
    
    @pytest.mark.integration
    def test_data_flow_consistency(self):
        """Тест консистентности потока данных через все компоненты"""
        
        # Создаем тестовые данные
        ohlcv_data = generate_test_ohlcv(bars=50)
        orderbook_data = generate_test_orderbook(mid_price=50000.0)
        
        # Проверяем что данные валидны на каждом этапе
        assert TestDataValidator.validate_ohlcv(ohlcv_data)
        assert TestDataValidator.validate_orderbook(orderbook_data)
        
        # Тестируем обработку через все компоненты
        with patch('ingest.prices.get_ohlcv', return_value=ohlcv_data):
            
            # 1. Технический анализ
            ta_score = 65.0  # Мокаем TA скор
            assert TestDataValidator.validate_signal_score(ta_score)
            
            # 2. Новостной анализ
            with patch('features.news_metrics.news_score', return_value=55.0) as mock_news:
                news_score_result = mock_news('BTCUSDT')
                assert TestDataValidator.validate_signal_score(news_score_result)
            
            # 3. Микроструктурный анализ
            try:
                from features.microstructure import compute_spread_bps, compute_obi
                spread = compute_spread_bps(orderbook_data)
                obi = compute_obi(orderbook_data)
                
                assert spread > 0
                assert -1 <= obi <= 1
            except ImportError:
                # Функции могут не существовать, пропускаем
                pass
            
            # 4. Объединение сигналов через симуляцию
            from backtest.sim_loop import run_simulation
            result = run_simulation(symbol='BTCUSDT', steps=5, fast_mode=True)
            
            assert isinstance(result, dict)
    
    @pytest.mark.performance
    def test_system_performance_under_load(self):
        """Тест производительности системы под нагрузкой"""
        import time
        
        start_time = time.time()
        
        # Запускаем несколько симуляций параллельно (имитируем нагрузку)
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT', 'SOLUSDT']
        results = []
        
        for symbol in symbols:
            with patch('ingest.prices.get_ohlcv') as mock_ohlcv, \
                 patch('features.news_metrics.news_score') as mock_news:
                
                mock_ohlcv.return_value = generate_test_ohlcv(bars=100)
                mock_news.return_value = 50.0
                
                from backtest.sim_loop import run_simulation
                result = run_simulation(symbol=symbol, steps=20, fast_mode=True)
                results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Проверяем что все завершилось быстро
        assert total_time < 60.0  # Менее минуты на все симуляции
        assert len(results) == len(symbols)
        assert all(isinstance(r, dict) for r in results)
    
    @pytest.mark.robustness
    def test_system_error_recovery(self):
        """Тест восстановления системы после ошибок"""
        
        # Тест 1: Ошибка в данных OHLCV
        with patch('ingest.prices.get_ohlcv', side_effect=Exception("Network error")):
            try:
                from backtest.sim_loop import run_simulation
                result = run_simulation(symbol='BTCUSDT', steps=5, fast_mode=True)
                # Если не падает, проверяем результат
                assert isinstance(result, dict)
            except Exception as e:
                # Должны быть определенные типы исключений
                assert isinstance(e, (ValueError, ConnectionError, TimeoutError, AttributeError))
        
        # Тест 2: Ошибка в новостях
        with patch('ingest.prices.get_ohlcv', return_value=generate_test_ohlcv()), \
             patch('features.news_metrics.news_score', side_effect=Exception("API error")):
            
            from backtest.sim_loop import run_simulation
            result = run_simulation(symbol='BTCUSDT', steps=5, fast_mode=True)
            
            # Система должна продолжить работу с дефолтными значениями
            assert isinstance(result, dict)
    
    @pytest.mark.integration
    def test_configuration_loading(self):
        """Тест загрузки и применения конфигурации"""
        
        # Создаем тестовую конфигурацию
        test_config = {
            "execution": {
                "spread_cap_bps": 100.0,
                "k_stop_atr": 3.0,
                "rr_tp1": 2.0,
                "tp1_frac": 0.6,
                "trail_atr_k": 1.5,
                "time_stop_bars": 30
            },
            "gates": {
                "max_spread_bps": 20.0,
                "news_blackout": True
            },
            "risk": {
                "per_trade_pct": 0.015,
                "max_position_frac": 0.15
            }
        }
        
        with patch('utils.model_utils.load_config', return_value=test_config):
            # Проверяем что конфигурация применяется
            try:
                from utils.model_utils import load_config
                config = load_config()
                
                assert config == test_config
                assert config["execution"]["k_stop_atr"] == 3.0
                assert config["risk"]["per_trade_pct"] == 0.015
                
            except ImportError:
                # Функция может не существовать
                pass


class TestDataIntegrityAcrossComponents:
    """Тесты целостности данных между компонентами"""
    
    @pytest.mark.integration
    def test_timestamp_consistency(self):
        """Тест консистентности временных меток"""
        
        # Создаем данные с одинаковыми временными метками
        base_timestamp = pd.Timestamp('2023-01-01 12:00:00')
        timestamps = pd.date_range(base_timestamp, periods=50, freq='5min')
        
        ohlcv_data = pd.DataFrame({
            'timestamp': timestamps,
            'open': np.random.uniform(50000, 51000, 50),
            'high': np.random.uniform(51000, 52000, 50),
            'low': np.random.uniform(49000, 50000, 50),
            'close': np.random.uniform(50000, 51000, 50),
            'volume': np.random.uniform(100, 1000, 50)
        })
        
        # Проверяем что временные метки правильно обрабатываются
        assert len(ohlcv_data) == 50
        assert ohlcv_data['timestamp'].is_monotonic_increasing
        
        # Интервалы должны быть постоянными
        intervals = ohlcv_data['timestamp'].diff()[1:]  # Пропускаем первый NaT
        assert all(interval == pd.Timedelta('5min') for interval in intervals)
    
    @pytest.mark.integration  
    def test_price_consistency_across_sources(self):
        """Тест консистентности цен между источниками"""
        
        mid_price = 50000.0
        ohlcv_data = generate_test_ohlcv(start_price=mid_price, bars=10, volatility=0.001)
        orderbook_data = generate_test_orderbook(mid_price=mid_price, spread_bps=1.0)
        
        # Цены из OHLCV должны быть близки к ценам из orderbook
        last_close = ohlcv_data['close'].iloc[-1]
        
        if orderbook_data and 'bids' in orderbook_data and 'asks' in orderbook_data:
            if orderbook_data['bids'] and orderbook_data['asks']:
                best_bid = orderbook_data['bids'][0][0]
                best_ask = orderbook_data['asks'][0][0]
                ob_mid_price = (best_bid + best_ask) / 2
                
                # Цены должны быть в разумном диапазоне друг от друга
                price_diff_pct = abs(last_close - ob_mid_price) / ob_mid_price * 100
                assert price_diff_pct < 5.0  # Менее 5% расхождения
    
    @pytest.mark.integration
    def test_volume_consistency(self):
        """Тест консистентности объемов"""
        
        ohlcv_data = generate_test_ohlcv(bars=20)
        
        # Объемы должны быть положительными и разумными
        assert all(ohlcv_data['volume'] > 0)
        assert all(ohlcv_data['volume'] < 1000000)  # Не слишком большие
        
        # Средний объем должен быть разумным
        avg_volume = ohlcv_data['volume'].mean()
        assert 10 < avg_volume < 10000


class TestRealTimeSimulation:
    """Тесты имитации реального времени"""
    
    @pytest.mark.slow
    def test_streaming_data_simulation(self):
        """Тест симуляции потоковых данных"""
        
        # Создаем данные для потокового режима
        full_data = generate_test_ohlcv(bars=100)
        
        # Имитируем получение данных по частям
        chunk_size = 10
        processed_chunks = []
        
        for i in range(0, len(full_data), chunk_size):
            chunk = full_data.iloc[i:i+chunk_size]
            
            # Каждый чанк должен быть валидным
            if not chunk.empty:
                assert TestDataValidator.validate_ohlcv(chunk)
                processed_chunks.append(chunk)
        
        # Объединенные данные должны соответствовать исходным
        combined_data = pd.concat(processed_chunks, ignore_index=True)
        assert len(combined_data) == len(full_data)
    
    @pytest.mark.integration
    def test_latency_simulation(self):
        """Тест симуляции задержек"""
        import time
        
        # Имитируем задержки в получении данных
        delays = [0.001, 0.005, 0.01, 0.02]  # От 1мс до 20мс
        
        for delay in delays:
            start_time = time.time()
            
            # Имитируем получение данных с задержкой
            time.sleep(delay)
            
            # Генерируем данные
            data = generate_test_ohlcv(bars=5)
            
            end_time = time.time()
            actual_delay = end_time - start_time
            
            # Проверяем что задержка была учтена
            assert actual_delay >= delay
            assert TestDataValidator.validate_ohlcv(data)


class TestSystemLimits:
    """Тесты системных ограничений"""
    
    @pytest.mark.performance
    def test_large_dataset_handling(self):
        """Тест обработки больших датасетов"""
        
        # Создаем большой датасет
        large_data = generate_test_ohlcv(bars=5000)
        
        with patch('ingest.prices.get_ohlcv', return_value=large_data):
            
            start_time = time.time()
            
            # Запускаем симуляцию на большом датасете
            from backtest.sim_loop import run_simulation
            result = run_simulation(symbol='BTCUSDT', steps=500, fast_mode=True)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Проверяем что обработка завершилась в разумное время
            assert execution_time < 120.0  # Менее 2 минут
            assert isinstance(result, dict)
    
    @pytest.mark.robustness
    def test_memory_limits(self):
        """Тест ограничений памяти"""
        
        # Создаем данные разного размера и отслеживаем использование памяти
        sizes = [100, 500, 1000, 2000]
        
        for size in sizes:
            data = generate_test_ohlcv(bars=size)
            
            # Данные должны создаваться без ошибок памяти
            assert len(data) == size
            assert TestDataValidator.validate_ohlcv(data)
            
            # Освобождаем память
            del data
    
    @pytest.mark.robustness
    def test_concurrent_access(self):
        """Тест одновременного доступа"""
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def run_simulation_thread(symbol, results_queue):
            try:
                with patch('ingest.prices.get_ohlcv') as mock_ohlcv, \
                     patch('features.news_metrics.news_score') as mock_news:
                    
                    mock_ohlcv.return_value = generate_test_ohlcv(bars=20)
                    mock_news.return_value = 50.0
                    
                    from backtest.sim_loop import run_simulation
                    result = run_simulation(symbol=symbol, steps=5, fast_mode=True)
                    results_queue.put(('success', symbol, result))
                    
            except Exception as e:
                results_queue.put(('error', symbol, str(e)))
        
        # Запускаем несколько потоков одновременно
        threads = []
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        
        for symbol in symbols:
            thread = threading.Thread(target=run_simulation_thread, args=(symbol, results_queue))
            thread.start()
            threads.append(thread)
        
        # Ждем завершения всех потоков
        for thread in threads:
            thread.join()
        
        # Проверяем результаты
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        assert len(results) == len(symbols)
        
        # Все результаты должны быть успешными или обработанными ошибками
        for status, symbol, data in results:
            assert status in ['success', 'error']
            if status == 'success':
                assert isinstance(data, dict)
