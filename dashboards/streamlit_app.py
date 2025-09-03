import os
import sys
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def simple_kpis_from_trades(trades_df):
    """Simple KPI calculation from trades DataFrame"""
    if len(trades_df) == 0:
        return {
            'trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'pnl_total': 0
        }
    
    # Only count closed trades
    closed_trades = trades_df[trades_df['action'] == 'CLOSE']
    
    if len(closed_trades) == 0:
        return {
            'trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'pnl_total': 0
        }
    
    total_pnl = closed_trades['pnl_usd'].sum()
    winning_trades = len(closed_trades[closed_trades['pnl_usd'] > 0])
    losing_trades = len(closed_trades[closed_trades['pnl_usd'] < 0])
    total_trades = len(closed_trades)
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    gross_profit = closed_trades[closed_trades['pnl_usd'] > 0]['pnl_usd'].sum()
    gross_loss = abs(closed_trades[closed_trades['pnl_usd'] < 0]['pnl_usd'].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
    
    return {
        'trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'pnl_total': total_pnl
    }

def simple_equity_curve(trades_df):
    """Simple equity curve calculation"""
    if len(trades_df) == 0:
        return pd.DataFrame()
    
    # Sort by timestamp
    trades_sorted = trades_df.sort_values('ts')
    trades_sorted['cumulative_pnl'] = trades_sorted['pnl_usd'].cumsum()
    trades_sorted['equity'] = 10000 + trades_sorted['cumulative_pnl']  # Starting with 10k
    
    return trades_sorted[['ts', 'equity']]

st.set_page_config(page_title="🤖 Enhanced Scalper Dashboard", layout="wide")

st.title("⚡ Enhanced Scalper Trading Dashboard")
st.markdown("---")

# File paths
trades_path = st.sidebar.text_input("📁 Путь к логу сделок", "logs/trades.csv")
signals_path = st.sidebar.text_input("📡 Путь к логу сигналов", "logs/signals.csv")

# Refresh button
if st.sidebar.button("🔄 Обновить данные"):
    st.rerun()

# Check if files exist
trades_exist = os.path.exists(trades_path)
signals_exist = os.path.exists(signals_path)

# === ОСНОВНЫЕ МЕТРИКИ ===
st.header("📊 Торговая статистика")

if trades_exist:
    df_trades = pd.read_csv(trades_path)
    k = simple_kpis_from_trades(df_trades)
    
    # Основные метрики в колонках
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("💰 Общий PnL", f"${k['pnl_total']:.2f}")
    col2.metric("📈 Сделок закрыто", f"{k['trades']}")
    col3.metric("🎯 Win Rate", f"{k['win_rate']:.1f}%")
    col4.metric("⚡ Profit Factor", f"{k['profit_factor']:.2f}")
    
    # Дополнительные метрики
    if len(df_trades) > 0:
        last_equity = df_trades['equity'].iloc[-1] if 'equity' in df_trades.columns else 10000
        col5.metric("💼 Текущий Equity", f"${last_equity:.2f}")
        
        # Вторая строка метрик
        col6, col7, col8, col9, col10 = st.columns(5)
        
        # Анализ прибыльных/убыточных сделок
        profitable_trades = df_trades[df_trades['pnl_usd'] > 0] if 'pnl_usd' in df_trades.columns else pd.DataFrame()
        losing_trades = df_trades[df_trades['pnl_usd'] < 0] if 'pnl_usd' in df_trades.columns else pd.DataFrame()
        
        avg_win = profitable_trades['pnl_usd'].mean() if len(profitable_trades) > 0 else 0
        avg_loss = abs(losing_trades['pnl_usd'].mean()) if len(losing_trades) > 0 else 0
        
        col6.metric("💚 Средняя прибыль", f"${avg_win:.2f}")
        col7.metric("❌ Средний убыток", f"${avg_loss:.2f}")
        col8.metric("🔥 Выигрышей подряд", "0")  # TODO: реализовать
        col9.metric("❄️ Проигрышей подряд", "0")  # TODO: реализовать
        
        # Максимальные значения
        if 'pnl_usd' in df_trades.columns:
            max_win = df_trades['pnl_usd'].max()
            max_loss = df_trades['pnl_usd'].min()
            col10.metric("🚀 Макс прибыль", f"${max_win:.2f}")
else:
    st.warning("⚠️ Файл trades.csv не найден. Запустите симуляцию для получения данных.")
    st.stop()

st.markdown("---")

# === ГРАФИКИ ===
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📈 Equity Curve")
    eq = simple_equity_curve(df_trades)
    if not eq.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(eq["ts"], eq["equity"], linewidth=2, color='#1f77b4')
        ax.fill_between(eq["ts"], eq["equity"], alpha=0.3, color='#1f77b4')
        ax.set_xlabel("Время")
        ax.set_ylabel("Equity ($)")
        ax.set_title("Рост капитала")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Drawdown analysis
        equity_values = eq["equity"].values
        running_max = np.maximum.accumulate(np.array(equity_values))
        drawdown = (np.array(equity_values) - running_max) / running_max * 100
        max_dd = np.min(drawdown)
        st.metric("📉 Максимальная просадка", f"{max_dd:.2f}%")
    else:
        st.info("Пока нет данных для графика equity.")

with col_right:
    st.subheader("💹 Распределение PnL")
    if 'pnl_usd' in df_trades.columns and len(df_trades) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Гистограмма PnL
        profits = df_trades[df_trades['pnl_usd'] > 0]['pnl_usd']
        losses = df_trades[df_trades['pnl_usd'] < 0]['pnl_usd']
        
        ax.hist(profits, bins=20, alpha=0.7, color='green', label=f'Прибыли ({len(profits)})')
        ax.hist(losses, bins=20, alpha=0.7, color='red', label=f'Убытки ({len(losses)})')
        ax.set_xlabel("PnL ($)")
        ax.set_ylabel("Количество сделок")
        ax.set_title("Распределение результатов сделок")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("Нет данных PnL для отображения.")

st.markdown("---")

# === ТЕКУЩИЙ СТАТУС ТОРГОВЛИ ===
st.header("🎯 Текущий статус")

col_status1, col_status2, col_status3 = st.columns(3)

with col_status1:
    st.subheader("📊 Последние сделки")
    # Показываем последние 5 сделок с цветовой индикацией
    if len(df_trades) > 0:
        recent_trades = df_trades.tail(5).copy()
        
        # Добавляем цветовую индикацию
        def color_pnl(val):
            if pd.isna(val):
                return ''
            color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
            return f'color: {color}'
        
        if 'pnl_usd' in recent_trades.columns:
            styled_trades = recent_trades[['ts', 'side', 'action', 'price', 'pnl_usd']].style.map(
                color_pnl, subset=['pnl_usd']
            )
            st.dataframe(styled_trades, width="stretch")
        else:
            st.dataframe(recent_trades[['ts', 'side', 'action', 'price']], width="stretch")

with col_status2:
    st.subheader("⚡ Активность по символам")
    if 'symbol' in df_trades.columns:
        symbol_stats = df_trades.groupby('symbol').agg({
            'pnl_usd': ['count', 'sum'] if 'pnl_usd' in df_trades.columns else 'count'
        }).round(2)
        
        if isinstance(symbol_stats.columns, pd.MultiIndex):
            symbol_stats.columns = ['Сделки', 'PnL']
        else:
            symbol_stats.columns = ['Сделки']
        
        st.dataframe(symbol_stats, width="stretch")
    else:
        st.info("Нет данных по символам")

with col_status3:
    st.subheader("� Активность по времени")
    if len(df_trades) > 0 and 'ts' in df_trades.columns:
        # Конвертируем timestamp в datetime если нужно
        try:
            df_trades['datetime'] = pd.to_datetime(df_trades['ts'])
            df_trades['hour'] = df_trades['datetime'].dt.hour
            
            hourly_activity = df_trades.groupby('hour').size()
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(hourly_activity.index.values, np.array(hourly_activity.values), color='skyblue')
            ax.set_xlabel("Час дня")
            ax.set_ylabel("Количество сделок")
            ax.set_title("Активность по часам")
            st.pyplot(fig)
        except:
            st.info("Не удалось обработать временные данные")

st.markdown("---")

# === ДЕТАЛЬНЫЙ АНАЛИЗ ТОРГОВЫХ РЕШЕНИЙ ===
st.header("🧠 Анализ торговых решений")

# Анализ причин входов/выходов
col_analysis1, col_analysis2 = st.columns(2)

with col_analysis1:
    st.subheader("📝 Причины сделок")
    if 'reason' in df_trades.columns:
        reason_counts = df_trades['reason'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie(np.array(reason_counts.values), labels=list(reason_counts.index), autopct='%1.1f%%')
        ax.set_title("Распределение причин сделок")
        st.pyplot(fig)
        
        # Табличка с детализацией
        reason_detailed = df_trades.groupby('reason').agg({
            'pnl_usd': ['count', 'sum', 'mean'] if 'pnl_usd' in df_trades.columns else 'count'
        }).round(2)
        
        if isinstance(reason_detailed.columns, pd.MultiIndex):
            reason_detailed.columns = ['Количество', 'Общий PnL', 'Средний PnL']
        
        st.dataframe(reason_detailed, width="stretch")

with col_analysis2:
    st.subheader("⚖️ Long vs Short")
    if 'side' in df_trades.columns:
        side_analysis = df_trades.groupby('side').agg({
            'pnl_usd': ['count', 'sum', 'mean'] if 'pnl_usd' in df_trades.columns else 'count'
        }).round(2)
        
        if isinstance(side_analysis.columns, pd.MultiIndex):
            side_analysis.columns = ['Сделки', 'Общий PnL', 'Средний PnL']
        
        st.dataframe(side_analysis, width="stretch")
        
        # График сравнения
        if 'pnl_usd' in df_trades.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            long_pnl = df_trades[df_trades['side'] == 'long']['pnl_usd'].dropna()
            short_pnl = df_trades[df_trades['side'] == 'short']['pnl_usd'].dropna()
            
            ax.hist(long_pnl, alpha=0.7, label=f'Long ({len(long_pnl)})', color='green', bins=15)
            ax.hist(short_pnl, alpha=0.7, label=f'Short ({len(short_pnl)})', color='red', bins=15)
            ax.set_xlabel("PnL ($)")
            ax.set_ylabel("Количество")
            ax.set_title("Long vs Short результаты")
            ax.legend()
            st.pyplot(fig)

st.markdown("---")

# === ДЕТАЛЬНЫЕ ТАБЛИЦЫ ===
st.header("📋 Детальная информация")

tab1, tab2, tab3 = st.tabs(["🧾 Все сделки", "📡 Сигналы", "⚙️ Настройки"])

with tab1:
    st.subheader("История всех сделок")
    if len(df_trades) > 0:
        # Фильтры
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            if 'side' in df_trades.columns:
                side_filter = st.selectbox("Тип сделки:", ['Все'] + list(df_trades['side'].unique()))
            else:
                side_filter = 'Все'
                
        with col_filter2:
            if 'symbol' in df_trades.columns:
                symbol_filter = st.selectbox("Символ:", ['Все'] + list(df_trades['symbol'].unique()))
            else:
                symbol_filter = 'Все'
                
        with col_filter3:
            show_rows = st.selectbox("Показать строк:", [20, 50, 100, 'Все'], index=0)
        
        # Применяем фильтры
        filtered_trades = df_trades.copy()
        if side_filter != 'Все' and 'side' in df_trades.columns:
            filtered_trades = filtered_trades[filtered_trades['side'] == side_filter]
        if symbol_filter != 'Все' and 'symbol' in df_trades.columns:
            filtered_trades = filtered_trades[filtered_trades['symbol'] == symbol_filter]
            
        # Показываем таблицу
        if show_rows != 'Все':
            filtered_trades = filtered_trades.head(int(show_rows))
            
        st.dataframe(
            filtered_trades.sort_values('ts', ascending=False) if 'ts' in filtered_trades.columns else filtered_trades,
            width="stretch",
            height=400
        )
        
        # Экспорт данных
        csv = filtered_trades.to_csv(index=False)
        st.download_button(
            label="📥 Скачать отфильтрованные данные",
            data=csv,
            file_name=f"trades_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )

with tab2:
    st.subheader("Сигналы торговой системы")
    if signals_exist:
        try:
            # Handle mixed CSV formats (old vs new columns) by using error_bad_lines=False
            df_signals = pd.read_csv(signals_path, on_bad_lines='skip')
            
            # Show column info for debugging
            st.info(f"📊 Загружено {len(df_signals)} сигналов. Колонки: {', '.join(df_signals.columns.tolist())}")
            
            # Display recent signals
            display_signals = df_signals.sort_values('ts', ascending=False).head(100) if 'ts' in df_signals.columns else df_signals.head(100)
            st.dataframe(
                display_signals,
                width="stretch",
                height=400
            )
            
            # Show summary stats if we have score column
            if 'score' in df_signals.columns:
                col_sig1, col_sig2, col_sig3 = st.columns(3)
                with col_sig1:
                    avg_score = df_signals['score'].mean()
                    st.metric("📈 Средний счёт", f"{avg_score:.1f}")
                with col_sig2:
                    if 'direction' in df_signals.columns:
                        long_signals = len(df_signals[df_signals['direction'] == 'long'])
                        short_signals = len(df_signals[df_signals['direction'] == 'short'])
                        st.metric("📊 Long/Short", f"{long_signals}/{short_signals}")
                with col_sig3:
                    if 'ob_source' in df_signals.columns:
                        ws_count = len(df_signals[df_signals['ob_source'] == 'ws'])
                        rest_count = len(df_signals[df_signals['ob_source'] == 'rest'])
                        st.metric("🌐 WS/REST OB", f"{ws_count}/{rest_count}")
                        
        except Exception as e:
            st.error(f"❌ Ошибка чтения файла сигналов: {e}")
            st.markdown("**Возможные причины:**")
            st.markdown("- Смешанные форматы CSV (старые и новые колонки)")
            st.markdown("- Поврежденные строки в файле")
            st.markdown("- Несовместимые данные")
            
            # Show the problematic area
            if "line" in str(e):
                st.code(f"Ошибка: {e}")
                st.markdown("**Попробуйте:**")
                st.code("# Очистить логи и запустить симуляцию заново\nrm logs/signals.csv\npython run_simulation.py --steps 10")
    else:
        st.info("📡 Файл сигналов не найден. Это опциональный компонент.")
        st.markdown("**Что такое сигналы?**")
        st.markdown("- Сигналы показывают все расчёты торговой системы")
        st.markdown("- Включают оценки TA, новостей, smart money")
        st.markdown("- Показывают причины входов и выходов")

with tab3:
    st.subheader("⚙️ Настройки дашборда")
    st.markdown("**Пути к файлам:**")
    st.code(f"Сделки: {trades_path}")
    st.code(f"Сигналы: {signals_path}")
    
    st.markdown("**Статистика файлов:**")
    if trades_exist:
        trades_size = os.path.getsize(trades_path)
        st.success(f"✅ Файл сделок: {trades_size} байт")
    else:
        st.error("❌ Файл сделок не найден")
        
    if signals_exist:
        signals_size = os.path.getsize(signals_path)
        st.success(f"✅ Файл сигналов: {signals_size} байт")
    else:
        st.warning("⚠️ Файл сигналов не найден")
    
    st.markdown("---")
    st.markdown("**🚀 Для запуска симуляции:**")
    st.code("python -m backtest.sim_loop")
    st.markdown("**🌐 Для запуска API:**")
    st.code("uvicorn app:app --reload")

# === FOOTER ===
st.markdown("---")
st.markdown("**⚡ Enhanced Scalper Dashboard** | Обновляется автоматически при новых данных")

st.markdown("---")
