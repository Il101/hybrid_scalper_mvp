#!/usr/bin/env python3
"""
Скрипт для запуска всех тестов проекта hybrid_scalper_mvp
Поддерживает разные режимы запуска тестов
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import time

def install_test_dependencies():
    """Установка зависимостей для тестирования"""
    print("🔧 Установка зависимостей для тестирования...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "test_requirements.txt"
        ], check=True, capture_output=True, text=True)
        print("✅ Зависимости установлены успешно")
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка установки зависимостей: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False
    
    return True

def run_tests(test_type='all', verbose=False, coverage=False, parallel=False):
    """Запуск тестов с различными опциями"""
    
    cmd = [sys.executable, "-m", "pytest"]
    
    # Базовые опции
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Покрытие кода
    if coverage:
        cmd.extend(["--cov=features", "--cov=backtest", "--cov=exec", "--cov=ingest"])
        cmd.append("--cov-report=html")
        cmd.append("--cov-report=term-missing")
    
    # Параллельное выполнение
    if parallel:
        cmd.extend(["-n", "auto"])
    
    # Выбор типа тестов
    if test_type == 'unit':
        cmd.extend(["-m", "unit"])
    elif test_type == 'integration':
        cmd.extend(["-m", "integration"])
    elif test_type == 'performance':
        cmd.extend(["-m", "performance"])
    elif test_type == 'robustness':
        cmd.extend(["-m", "robustness"])
    elif test_type == 'fast':
        cmd.extend(["-m", "not slow"])
    elif test_type == 'slow':
        cmd.extend(["-m", "slow"])
    elif test_type == 'no-api':
        cmd.extend(["-m", "not api"])
    
    # Директория с тестами
    cmd.append("tests/")
    
    print(f"🧪 Запуск тестов: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=False)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"\n⏱️  Время выполнения: {execution_time:.2f} секунд")
        
        if result.returncode == 0:
            print("✅ Все тесты прошли успешно!")
        else:
            print(f"❌ Некоторые тесты не прошли (код завершения: {result.returncode})")
        
        return result.returncode == 0
        
    except KeyboardInterrupt:
        print("\n🛑 Тестирование прервано пользователем")
        return False
    except Exception as e:
        print(f"❌ Ошибка при запуске тестов: {e}")
        return False

def run_specific_test_file(test_file, verbose=False):
    """Запуск конкретного файла тестов"""
    
    test_path = Path("tests") / test_file
    
    if not test_path.exists():
        print(f"❌ Файл тестов {test_path} не найден")
        return False
    
    cmd = [sys.executable, "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    cmd.append(str(test_path))
    
    print(f"🧪 Запуск тестов из файла: {test_file}")
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Ошибка при запуске тестов: {e}")
        return False

def run_test_discovery():
    """Обнаружение и вывод списка всех тестов"""
    print("🔍 Обнаружение тестов...")
    
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests/"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("\n📋 Найденные тесты:")
        print(result.stdout)
        
        # Подсчет тестов
        lines = result.stdout.split('\n')
        test_count = sum(1 for line in lines if '::test_' in line)
        print(f"\n📊 Всего найдено тестов: {test_count}")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка обнаружения тестов: {e}")
        return False

def generate_test_report():
    """Генерация детального отчета о тестах"""
    print("📊 Генерация детального отчета...")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "--cov=features", "--cov=backtest", "--cov=exec", "--cov=ingest",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--html=tests/reports/test_report.html",
        "--self-contained-html",
        "tests/"
    ]
    
    # Создаем директорию для отчетов
    os.makedirs("tests/reports", exist_ok=True)
    
    try:
        result = subprocess.run(cmd, check=False)
        
        if result.returncode == 0:
            print("✅ Отчет сгенерирован успешно!")
            print("📁 HTML отчет: tests/reports/test_report.html")
            print("📁 Покрытие кода: htmlcov/index.html")
        else:
            print(f"⚠️  Отчет сгенерирован с ошибками (код: {result.returncode})")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Ошибка генерации отчета: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Запуск тестов для hybrid_scalper_mvp')
    
    parser.add_argument('--type', choices=['all', 'unit', 'integration', 'performance', 'robustness', 'fast', 'slow', 'no-api'],
                        default='all', help='Тип тестов для запуска')
    parser.add_argument('--file', help='Запустить конкретный файл тестов')
    parser.add_argument('--install-deps', action='store_true', help='Установить зависимости для тестов')
    parser.add_argument('--discover', action='store_true', help='Обнаружить и показать все тесты')
    parser.add_argument('--report', action='store_true', help='Сгенерировать детальный отчет')
    parser.add_argument('-v', '--verbose', action='store_true', help='Подробный вывод')
    parser.add_argument('--coverage', action='store_true', help='Включить анализ покрытия кода')
    parser.add_argument('--parallel', action='store_true', help='Запускать тесты параллельно')
    
    args = parser.parse_args()
    
    print("🚀 Тестирование проекта hybrid_scalper_mvp")
    print("=" * 50)
    
    # Установка зависимостей
    if args.install_deps:
        if not install_test_dependencies():
            return 1
    
    # Обнаружение тестов
    if args.discover:
        if not run_test_discovery():
            return 1
        return 0
    
    # Генерация отчета
    if args.report:
        if not generate_test_report():
            return 1
        return 0
    
    # Запуск конкретного файла
    if args.file:
        success = run_specific_test_file(args.file, args.verbose)
        return 0 if success else 1
    
    # Обычный запуск тестов
    success = run_tests(
        test_type=args.type,
        verbose=args.verbose,
        coverage=args.coverage,
        parallel=args.parallel
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
