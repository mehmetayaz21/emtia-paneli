import threading
import websocket
import json
import pandas as pd
import time
import requests

# Binance'den tüm sembolleri çek (SPOT)
def get_all_symbols():
    try:
        url = "https://api.binance.com/api/v3/exchangeInfo"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            symbols = []
            for symbol_info in data['symbols']:
                symbol = symbol_info['symbol'].lower()
                # Sadece USDT paritelerini al
                if symbol.endswith('usdt') and symbol_info['status'] == 'TRADING':
                    symbols.append(symbol)
            return symbols[:200]  # İlk 200 sembol
    except Exception as e:
        print(f"Sembol listesi alınamadı: {e}")
        # Fallback semboller
        return ['btcusdt', 'ethusdt', 'bnbusdt', 'adausdt', 'xrpusdt']

# Takip edilecek semboller (dinamik olarak alınacak)
SYMBOLS = get_all_symbols()
print(f"Toplam {len(SYMBOLS)} sembol takip ediliyor")

# Her sembol için veri saklanacak
symbol_data = {symbol: [] for symbol in SYMBOLS}
ema_results = {}

# Binance WebSocket endpoint (SPOT)
def get_ws_url(symbol):
    return f"wss://stream.binance.com:9443/ws/{symbol}@kline_1m"

def calculate_ema(prices, period=200):
    """Doğru EMA hesaplama"""
    if len(prices) < period:
        return None
    
    # İlk SMA hesapla
    sma = sum(prices[:period]) / period
    
    # Multiplier hesapla
    multiplier = 2 / (period + 1)
    
    # EMA hesapla
    ema = sma
    for price in prices[period:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
    
    return ema

def on_message(ws, message, symbol):
    try:
        data = json.loads(message)
        kline = data['k']
        close = float(kline['c'])
        close_time = int(kline['T']) // 1000
        is_closed = kline['x']  # Mum kapandı mı?
        
        # Her sembol için son 250 kapanışı sakla
        symbol_data[symbol].append({'close': close, 'time': close_time})
        if len(symbol_data[symbol]) > 250:
            symbol_data[symbol] = symbol_data[symbol][-250:]
        
        # EMA 200 hesapla (sadece kapalı mumlar için)
        if is_closed and len(symbol_data[symbol]) >= 200:
            closes = [item['close'] for item in symbol_data[symbol]]
            ema_200 = calculate_ema(closes, 200)
            
            if ema_200 is not None:
                fark = closes[-1] - ema_200
                ema_results[symbol] = {
                    'symbol': symbol.upper(),
                    'close': round(closes[-1], 4),
                    'ema200': round(ema_200, 4),
                    'fark': round(fark, 4),
                    'zaman': time.strftime('%Y-%m-%d %H:%M', time.localtime(close_time))
                }
                # Debug: Kaç sembol için EMA hesaplandığını göster
                if len(ema_results) % 10 == 0:  # Her 10 sembolde bir
                    print(f"EMA hesaplandı: {len(ema_results)} sembol")
    except Exception as e:
        print(f"Veri işleme hatası ({symbol}): {e}")

def on_error(ws, error):
    print(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    print(f"WebSocket closed: {close_status_code} {close_msg}")

def run_ws(symbol):
    try:
        ws = websocket.WebSocketApp(
            get_ws_url(symbol),
            on_message=lambda ws, msg: on_message(ws, msg, symbol),
            on_error=on_error,
            on_close=on_close
        )
        ws.run_forever()
    except Exception as e:
        print(f"WebSocket başlatma hatası ({symbol}): {e}")

def start_all_ws():
    print(f"WebSocket bağlantıları başlatılıyor...")
    for symbol in SYMBOLS:
        t = threading.Thread(target=run_ws, args=(symbol,), daemon=True)
        t.start()
        time.sleep(0.1)  # Rate limit için küçük gecikme
    print("Tüm WebSocket bağlantıları başlatıldı")

# Flask uygulaması başlatılmadan önce bu fonksiyon çağrılmalı 