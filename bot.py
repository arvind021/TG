import telebot
import requests
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import io
import os

# Telegram Bot Token
API_TOKEN = '7664042669:AAFOuogg0eEhlEgvmQ3U6xUF7dITqCrfda0'
bot = telebot.TeleBot(API_TOKEN)

# Get Binance candles dynamically based on symbol and interval
def get_binance_candles(symbol, interval='1m'):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=100'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            closes = [float(candle[4]) for candle in data]
            volumes = [float(candle[5]) for candle in data]
            highs = [float(candle[2]) for candle in data]
            lows = [float(candle[3]) for candle in data]
            times = [datetime.fromtimestamp(candle[0]/1000) for candle in data]
            return closes, volumes, highs, lows, times
        else:
            print(f"Error fetching data: {response.status_code}")
            return None, None, None, None, None
    except Exception as e:
        print(f"Exception: {e}")
        return None, None, None, None, None

def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)

def calculate_macd(prices, fast=12, slow=26, signal=9):
    exp1 = np.convolve(prices, np.ones(fast)/fast, mode='valid')
    exp2 = np.convolve(prices, np.ones(slow)/slow, mode='valid')
    macd_line = exp1[-len(exp2):] - exp2
    signal_line = np.convolve(macd_line, np.ones(signal)/signal, mode='valid')
    if len(macd_line) > len(signal_line):
        macd_line = macd_line[-len(signal_line):]
    hist = macd_line - signal_line
    return macd_line[-1], signal_line[-1], hist[-1]

def calculate_vwap(closes, volumes):
    vwap = np.sum(np.array(closes) * np.array(volumes)) / np.sum(volumes)
    return round(vwap, 2)

def calculate_bollinger_bands(prices, period=20):
    sma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    upper = sma + 2 * std
    lower = sma - 2 * std
    return round(upper, 2), round(lower, 2)

def calculate_adx(highs, lows, closes, period=14):
    highs = np.array(highs)
    lows = np.array(lows)
    closes = np.array(closes)

    plus_dm = highs[1:] - highs[:-1]
    minus_dm = lows[:-1] - lows[1:]
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr1 = highs[1:] - lows[1:]
    tr2 = np.abs(highs[1:] - closes[:-1])
    tr3 = np.abs(lows[1:] - closes[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])

    atr = np.convolve(tr, np.ones(period)/period, mode='valid')
    plus_di_raw = np.convolve(plus_dm, np.ones(period)/period, mode='valid')[:len(atr)]
    minus_di_raw = np.convolve(minus_dm, np.ones(period)/period, mode='valid')[:len(atr)]

    plus_di = 100 * plus_di_raw / atr
    minus_di = 100 * minus_di_raw / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-6)

    adx = np.mean(dx[-period:])
    return round(adx, 2)

def calculate_stoch_rsi(prices, period=14):
    rsi_series = [calculate_rsi(prices[i - period:i]) if i >= period else 0 for i in range(len(prices))]
    rsi_series = np.array(rsi_series)
    min_rsi = np.min(rsi_series[-period:])
    max_rsi = np.max(rsi_series[-period:])
    stoch_rsi = (rsi_series[-1] - min_rsi) / (max_rsi - min_rsi) * 100 if max_rsi != min_rsi else 0
    return round(stoch_rsi, 2)

def predict_next_candle(candles):
    ma_short = np.mean(candles[-5:])
    ma_long = np.mean(candles[-20:])
    trend = 'UP' if ma_short > ma_long else 'DOWN'
    return trend, ma_short, ma_long

def generate_chart(symbol, times, closes, rsis, rsi_times):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(times, closes, label='Price', color='blue')
    ax1.set_title(f"{symbol} - Last 100 Candles")
    ax1.set_ylabel("Price")
    ax1.grid(True)

    ax2.plot(rsi_times, rsis, label='RSI', color='green')
    ax2.axhline(70, color='red', linestyle='--')
    ax2.axhline(30, color='red', linestyle='--')
    ax2.set_ylabel("RSI")
    ax2.set_xlabel("Time")
    ax2.grid(True)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

@bot.message_handler(commands=['predict'])
def handle_predict(message):
    try:
        parts = message.text.split()
        symbol = parts[1].upper() if len(parts) >= 2 else 'BTCUSDT'
        interval = parts[2] if len(parts) >= 3 else '1m'

        closes, volumes, highs, lows, times = get_binance_candles(symbol, interval)
        if closes:
            trend, ma5, ma20 = predict_next_candle(closes)
            rsi = calculate_rsi(closes)
            rsis = [calculate_rsi(closes[i-14:i]) for i in range(14, len(closes))]
            rsi_times = times[14:]
            vol_avg = round(np.mean(volumes[-10:]), 2)
            vwap = calculate_vwap(closes, volumes)
            macd, signal, hist = calculate_macd(closes)
            upper_bb, lower_bb = calculate_bollinger_bands(closes)
            adx = calculate_adx(highs, lows, closes)
            stoch_rsi = calculate_stoch_rsi(closes)
            confidence = 90 if trend == 'UP' and rsi < 70 and macd > signal and closes[-1] > vwap and adx > 20 else 60

            reply_text = (
                f"📊 {symbol} Prediction ({interval}):\n\n"
                f"• MA(5): {ma5:.2f}\n"
                f"• MA(20): {ma20:.2f}\n"
                f"• RSI(14): {rsi}\n"
                f"• Stoch RSI: {stoch_rsi}\n"
                f"• VWAP: {vwap}\n"
                f"• MACD: {macd:.2f}, Signal: {signal:.2f}\n"
                f"• Bollinger Bands: Upper {upper_bb}, Lower {lower_bb}\n"
                f"• ADX: {adx}\n"
                f"• Avg Volume (10 candles): {vol_avg}\n"
                f"• Trend: {'📈 UP' if trend == 'UP' else '📉 DOWN'}\n"
                f"• Confidence: {confidence}%\n\n"
                f"🔮 Expected next {interval} candle: {trend}"
            )

            chart = generate_chart(symbol, times, closes, rsis, rsi_times)
            bot.send_photo(message.chat.id, chart, caption=reply_text)
        else:
            bot.reply_to(message, f"❌ Failed to fetch candle data for {symbol}.")
    except Exception as e:
        bot.reply_to(message, f"⚠️ Error: {e}")

@bot.message_handler(commands=['symbols'])
def handle_symbols(message):
    try:
        url = "https://api.binance.com/api/v3/exchangeInfo"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            symbols = [s['symbol'] for s in data['symbols'] if s['quoteAsset'] == 'USDT']
            top = sorted(symbols)[:30]  # limit list to avoid spam
            reply = "📄 Available USDT Symbols:\n" + '\n'.join(top)
            bot.reply_to(message, reply)
        else:
            bot.reply_to(message, "⚠️ Could not fetch symbols.")
    except Exception as e:
        bot.reply_to(message, f"⚠️ Error: {e}")

@bot.message_handler(commands=['top'])
def handle_top(message):
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            sorted_data = sorted(data, key=lambda x: float(x['quoteVolume']), reverse=True)
            top_symbols = [x['symbol'] for x in sorted_data if x['symbol'].endswith('USDT')][:10]
            reply = "🔥 Top 10 Most Traded USDT Pairs:\n" + '\n'.join(top_symbols)
            bot.reply_to(message, reply)
        else:
            bot.reply_to(message, "⚠️ Failed to fetch top pairs.")
    except Exception as e:
        bot.reply_to(message, f"⚠️ Error: {e}")

@bot.message_handler(commands=['start', 'help'])
def handle_start(message):
    help_text = (
        "🤖 *Powerful Binary Candle Prediction Bot*\n\n"
        "Use the following command to predict:\n"
        "/predict [SYMBOL] [TIMEFRAME]\n"
        "Example: /predict BTCUSDT 5m or /predict ETHUSDT 15m\n\n"
        "Commands:\n"
        "/symbols - Show supported USDT pairs\n"
        "/top - Show top 10 traded USDT pairs\n"
        "Default is BTCUSDT 1m if no input provided."
    )
    bot.reply_to(message, help_text, parse_mode='Markdown')

bot.polling()

