#!/usr/bin/env python3
import pandas as pd
import numpy as np
from datetime import datetime

# Анализ сделок
try:
    trades = pd.read_csv('logs/trades.csv')
    print('📊 АНАЛИЗ СДЕЛОК (trades.csv)')
    print('=' * 50)
    print(f'Всего сделок: {len(trades)}')
    
    if len(trades) > 0:
        # Временной диапазон
        trades['ts'] = pd.to_datetime(trades['ts'])
        start_time = trades['ts'].min()
        end_time = trades['ts'].max()
        duration = end_time - start_time
        print(f'Период торговли: {start_time} - {end_time}')
        print(f'Длительность: {duration}')
        
        # Символы
        symbols = trades['symbol'].value_counts()
        print(f'Торгуемые символы: {dict(symbols)}')
        
        # Стороны сделок
        sides = trades['side'].value_counts()
        print(f'Распределение сторон: {dict(sides)}')
        
        # Причины закрытия/действий
        if 'action' in trades.columns:
            actions = trades['action'].value_counts().head(10)
            print(f'Топ причин действий: {dict(actions)}')
        
        if 'reason' in trades.columns:
            reasons = trades['reason'].value_counts().head(10)
            print(f'Топ причины: {dict(reasons)}')
        
        # PnL анализ
        if 'pnl_usd' in trades.columns:
            total_pnl = trades['pnl_usd'].sum()
            winning_trades = trades[trades['pnl_usd'] > 0]
            losing_trades = trades[trades['pnl_usd'] < 0]
            
            print(f'Общий PnL: ${total_pnl:.2f}')
            print(f'Прибыльных сделок: {len(winning_trades)} (средний выигрыш: ${winning_trades["pnl_usd"].mean():.2f})')
            print(f'Убыточных сделок: {len(losing_trades)} (средний убыток: ${losing_trades["pnl_usd"].mean():.2f})')
            
            if 'equity' in trades.columns:
                final_equity = trades['equity'].iloc[-1]
                initial_equity = trades['equity'].iloc[0] if len(trades) > 0 else 10000
                print(f'Итоговый баланс: ${final_equity:.2f} (начальный: ${initial_equity:.2f})')
        
        # Размеры сделок
        if 'size_usd' in trades.columns:
            avg_size = trades['size_usd'].mean()
            max_size = trades['size_usd'].max()
            print(f'Средний размер сделки: ${avg_size:.2f}, максимальный: ${max_size:.2f}')
            
        # Комиссии
        if 'fees_usd' in trades.columns:
            total_fees = trades['fees_usd'].sum()
            print(f'Общие комиссии: ${total_fees:.2f}')
    
    print()
except Exception as e:
    print(f'Ошибка при чтении trades.csv: {e}')

# Анализ сигналов
try:
    signals = pd.read_csv('logs/signals.csv')
    print('📡 АНАЛИЗ СИГНАЛОВ (signals.csv)')
    print('=' * 50)
    print(f'Всего сигналов: {len(signals)}')
    
    if len(signals) > 0:
        # Временной диапазон
        signals['ts'] = pd.to_datetime(signals['ts'], unit='ms')
        start_time = signals['ts'].min()
        end_time = signals['ts'].max()
        duration = end_time - start_time
        print(f'Период сигналов: {start_time} - {end_time}')
        print(f'Длительность: {duration}')
        
        # Символы
        symbols = signals['symbol'].value_counts()
        print(f'Символы сигналов: {dict(symbols)}')
        
        # Направления сигналов
        if 'direction' in signals.columns:
            directions = signals['direction'].value_counts()
            print(f'Направления: {dict(directions)}')
        
        # Статистика скоров
        if 'score' in signals.columns:
            print(f'Скор - среднее: {signals["score"].mean():.2f}, медиана: {signals["score"].median():.2f}')
            print(f'Скор - мин: {signals["score"].min():.2f}, макс: {signals["score"].max():.2f}')
        
        # Спреды
        if 'spread_bps' in signals.columns:
            print(f'Спред (bps) - среднее: {signals["spread_bps"].mean():.2f}, медиана: {signals["spread_bps"].median():.2f}')
        
        # OBI анализ
        if 'obi' in signals.columns:
            print(f'OBI - среднее: {signals["obi"].mean():.3f}, медиана: {signals["obi"].median():.3f}')
        
        # Источники данных - проверяем последние колонки если структура неясна
        print(f'Колонки в signals: {list(signals.columns)}')
        
        # Анализ возрастов данных - проверяем наличие числовых колонок
        numeric_cols = signals.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 10:  # Если много колонок, покажем только основные
            print(f'Числовые колонки (есть {len(numeric_cols)}): {list(numeric_cols[:10])}...')
        
        # Частота сигналов
        signals_per_minute = len(signals) / (duration.total_seconds() / 60) if duration.total_seconds() > 0 else 0
        print(f'Частота сигналов: {signals_per_minute:.2f} сигналов/минуту')

except Exception as e:
    print(f'Ошибка при чтении signals.csv: {e}')

print('\n🏁 Анализ завершён')
