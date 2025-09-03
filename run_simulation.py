#!/usr/bin/env python3
"""
Удобный запуск симуляций трейдинг-бота

Использование:
    python run_simulation.py                       # Быстрая симуляция BTCUSDT, 5m, 50 шагов
    python run_simulation.py --symbol ETHUSDT      # Симуляция ETHUSDT
    python run_simulation.py --steps 100           # 100 шагов симуляции
    python run_simulation.py --slow                # Медленный режим с реальными API
    python run_simulation.py --timeframe 1m        # 1-минутный таймфрейм
"""

import argparse
import subprocess
import sys
from backtest.sim_loop import run_simulation

def main():
    parser = argparse.ArgumentParser(description='Запуск симуляции скальпинг-бота')
    
    parser.add_argument('--symbol', default='SOLUSDT', 
                        help='Торговая пара (по умолчанию: BTCUSDT)')
    parser.add_argument('--timeframe', default='5m', 
                        help='Таймфрейм (по умолчанию: 5m)')
    parser.add_argument('--steps', type=int, default=0, 
                        help='Количество шагов симуляции (0 = бесконечно, по умолчанию: 0)')
    parser.add_argument('--use-ws', action='store_true', 
                        help='Использовать WebSocket для данных')
    parser.add_argument('--use-ws-prices', action='store_true',
                        help='Использовать WebSocket для цен (с REST fallback)')
    parser.add_argument('--slow', action='store_true', 
                        help='Медленный режим с реальными API вызовами')
    parser.add_argument('--preset', type=str, default=None,
                        help='Имя пресета конфигурации (например: fast, live-sol, debug)')
    parser.add_argument('--list-presets', action='store_true', help='Показать доступные пресеты')
    parser.add_argument('--rotate-logs', action='store_true', help='Запустить ротацию логов перед стартом')

    # Make slow mode and WS (including WS prices) the default behavior
    parser.set_defaults(use_ws=True, use_ws_prices=True, slow=True)
    
    args = parser.parse_args()

    # Presets
    PRESETS = {
        'fast': {'symbol': 'SOLUSDT', 'timeframe': '5m', 'steps': 50, 'use_ws': False, 'use_ws_prices': False, 'slow': False},
        'live-sol': {'symbol': 'SOLUSDT', 'timeframe': '5m', 'steps': 0, 'use_ws': True, 'use_ws_prices': True, 'slow': True},
        'live-batch': {'symbol': 'BTCUSDT', 'timeframe': '1m', 'steps': 0, 'use_ws': True, 'use_ws_prices': False, 'slow': True},
        'debug': {'symbol': 'ETHUSDT', 'timeframe': '1m', 'steps': 10, 'use_ws': False, 'use_ws_prices': False, 'slow': False}
    }

    if args.list_presets:
        print("Available presets:")
        for k, v in PRESETS.items():
            print(f"  {k}: {v}")
        return

    # Apply preset if provided and individual args not overridden
    if args.preset:
        p = PRESETS.get(args.preset)
        if not p:
            print(f"Unknown preset '{args.preset}'. Use --list-presets to see available options.")
            return
        # Apply fields only if user didn't explicitly set them (i.e., they are default)
        if args.symbol == parser.get_default('symbol') and 'symbol' in p:
            args.symbol = p['symbol']
        if args.timeframe == parser.get_default('timeframe') and 'timeframe' in p:
            args.timeframe = p['timeframe']
        if args.steps == parser.get_default('steps') and 'steps' in p:
            args.steps = p['steps']
        if not args.use_ws and p.get('use_ws'):
            args.use_ws = p['use_ws']
        if not args.use_ws_prices and p.get('use_ws_prices'):
            args.use_ws_prices = p['use_ws_prices']
        if not args.slow and p.get('slow'):
            args.slow = p['slow']
    
    # Инвертируем логику: по умолчанию быстрый режим, --slow включает медленный
    fast_mode = not args.slow
    
    print(f"🚀 Запуск симуляции:")
    print(f"   Символ: {args.symbol}")
    print(f"   Таймфрейм: {args.timeframe}")
    print(f"   Шагов: {'Бесконечно' if args.steps == 0 else args.steps}")
    print(f"   Режим: {'Быстрый (мок-данные)' if fast_mode else 'Медленный (реальные API)'}")
    print(f"   WebSocket: {'Да' if args.use_ws else 'Нет'}")
    print(f"   WS Цены: {'Да' if args.use_ws_prices else 'Нет'}")
    print("-" * 50)
    
    try:
        # Optional: rotate logs before starting the simulation
        if args.rotate_logs:
            try:
                print('🔄 Ротация логов: запуск scripts/rotate_logs.py')
                subprocess.run([sys.executable, 'scripts/rotate_logs.py'], check=True)
                print('🔄 Ротация логов завершена')
            except Exception as e:
                print(f'⚠️ Ротация логов завершилась с ошибкой: {e}')

        result = run_simulation(
            symbol=args.symbol,
            timeframe=args.timeframe,
            steps=args.steps,
            use_ws=args.use_ws,
            use_ws_prices=args.use_ws_prices,
            fast_mode=fast_mode
        )
        
        print("-" * 50)
        print("✅ Симуляция завершена успешно!")
        print(f"   Итоговый баланс: ${result.get('cash_final', 'N/A')}")
        print(f"   Лог сделок: {result.get('log_path', 'N/A')}")
        print(f"\n📊 Для анализа результатов запустите дашборд:")
        print(f"   streamlit run dashboards/streamlit_app.py --server.port 8503")
        
    except Exception as e:
        print(f"❌ Ошибка при выполнении симуляции: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
