#!/usr/bin/env python3
"""
Удобный запуск дашборда для анализа результатов

Использование:
    python run_dashboard.py          # Запуск на порту 8503
    python run_dashboard.py --port 8504  # Запуск на другом порту
"""

import argparse
import subprocess
import sys
import webbrowser
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Запуск дашборда для анализа торговых результатов')
    parser.add_argument('--port', type=int, default=8503, 
                        help='Порт для запуска (по умолчанию: 8503)')
    parser.add_argument('--no-browser', action='store_true',
                        help='Не открывать браузер автоматически')
    
    args = parser.parse_args()
    
    # Проверим наличие файлов с результатами
    logs_dir = Path('logs')
    if not logs_dir.exists():
        print("⚠️  Папка logs не найдена. Возможно, нужно сначала запустить симуляцию.")
    elif not any(logs_dir.glob('*.csv')):
        print("⚠️  Файлы с результатами не найдены в logs/. Возможно, нужно сначала запустить симуляцию.")
    
    print(f"🎯 Запуск дашборда на порту {args.port}...")
    print(f"📊 URL: http://localhost:{args.port}")
    print("-" * 50)
    
    try:
        # Запуск Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "dashboards/streamlit_app.py", 
            "--server.port", str(args.port),
            "--server.headless", "true"
        ]
        
        print("Запуск команды:", " ".join(cmd))
        
        # Открыть браузер через несколько секунд
        if not args.no_browser:
            def open_browser():
                time.sleep(3)  # Подождать запуска сервера
                webbrowser.open(f'http://localhost:{args.port}')
            
            import threading
            browser_thread = threading.Thread(target=open_browser)
            browser_thread.daemon = True
            browser_thread.start()
        
        # Запустить Streamlit
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n👋 Дашборд остановлен пользователем")
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при запуске дашборда: {e}")
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")

if __name__ == "__main__":
    main()
