from flask import Flask, render_template, request
import requests
import pandas as pd
import time
from database import db
import numpy as np
from datetime import datetime, timedelta
import threading

# Binance'ten çekilecek maksimum sembol sayısı
MAX_SYMBOLS = 300  # Önceden 100'dü; performansa göre değiştirilebilir

app = Flask(__name__)

# Global değişkenler
price_history = {}  # {symbol: [(timestamp, price), ...]}
alert_threshold = 0.4  # %0.4 değişim uyarısı
alerts = []  # Uyarıları saklayacak liste

def get_all_binance_symbols():
    """Binance'den tüm USDT paritelerini al"""
    try:
        url = "https://api.binance.com/api/v3/exchangeInfo"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            symbols = []
            
            for symbol_info in data['symbols']:
                symbol = symbol_info['symbol']
                # Sadece USDT paritelerini ve aktif olanları al
                if (symbol.endswith('USDT') and 
                    symbol_info['status'] == 'TRADING' and
                    symbol_info['isSpotTradingAllowed']):
                    symbols.append(symbol)
            
            # İlk MAX_SYMBOLS sembolü al
            symbols = symbols[:MAX_SYMBOLS]
            print(f"İlk {MAX_SYMBOLS} USDT paritesi alındı")
            return symbols
        else:
            print(f"API hatası: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"Sembol listesi alınamadı: {e}")
        # Fallback semboller
        return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT']

def calculate_zscore(data_list, value):
    """Z-Score hesapla"""
    if len(data_list) < 2:
        return 0
    
    mean = np.mean(data_list)
    std = np.std(data_list)
    
    if std == 0:
        return 0
    
    return (value - mean) / std

def calculate_momentum_roc(df, ema_len=200, ema_roc_len=10):
    df = df.copy()
    df['ema'] = df['close'].ewm(span=ema_len, adjust=False).mean()
    df['ema_roc'] = df['close'].ewm(span=ema_roc_len, adjust=False).mean()
    df['crossup'] = (df['close'] > df['ema']) & (df['close'].shift(1) <= df['ema'].shift(1))
    df['crossdown'] = (df['close'] < df['ema']) & (df['close'].shift(1) >= df['ema'].shift(1))
    momentum_roc_down = np.nan
    momentum_roc_up = np.nan
    if df['crossdown'].any():
        idx = df.index[df['crossdown']][-1]
        crossdown_close = df.loc[idx, 'close']
        crossdown_ema_roc = df.loc[idx, 'ema_roc']
        momentum_roc_down = ((crossdown_close / crossdown_ema_roc) - 1) * 100
    if df['crossup'].any():
        idx = df.index[df['crossup']][-1]
        crossup_close = df.loc[idx, 'close']
        crossup_ema_roc = df.loc[idx, 'ema_roc']
        momentum_roc_up = ((crossup_close / crossup_ema_roc) - 1) * 100
    return momentum_roc_down, momentum_roc_up

def get_binance_data(rf=0.01, period=200):
    """REST API ile Binance'den veri çek ve finansal oranları hesapla (1m sabit)"""
    try:
        # İlk MAX_SYMBOLS USDT paritesini al
        symbols = get_all_binance_symbols()
        
        results = []
        processed_count = 0
        all_fark_yuzde = []  # Tüm fark yüzdelerini topla
        SPIKE_THRESHOLD = 0.4  # 1m'de %0.4 üzeri artış (5m'deki %2'ye eşdeğer)
        spikes = []  # ani yükseliş yaşayan semboller
        
        for symbol in symbols:
            try:
                # 1 dakikalık son 'period' mum verisi + warm-up al
                warmup = 800  # TradingView ile yakınsama için ek veri
                limit_size = min(period + warmup, 1000)  # Binance limit 1000
                url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit={limit_size}"
                response = requests.get(url, timeout=3)
                
                if response.status_code == 200:
                    klines = response.json()
                    closes = [float(k[4]) for k in klines]
                    
                    if len(closes) >= period:
                        # Pandas tabanlı EMA (TradingView ile daha uyumlu) 
                        closes_np = np.array(closes, dtype=np.float64)
                        ema = pd.Series(closes_np).ewm(span=period, adjust=False).mean().values
                        last_close = closes_np[-1]
                        last_ema = ema[-1]
                        
                        # Momentum ROC fonksiyonu DataFrame istiyor; minimal df oluştur
                        df = pd.DataFrame({
                            'close': closes_np,
                            f'ema{period}': ema
                        })
                        
                        fark = last_close - last_ema
                        fark_yuzde = (fark / last_ema) * 100
                        
                        # Fark yüzdesini listeye ekle
                        all_fark_yuzde.append(fark_yuzde)
                        
                        last_time = int(klines[-1][6]) // 1000
                        # Sadece saat:dakika göster
                        zaman = time.strftime('%H:%M', time.localtime(last_time))
                        
                        # Finansal oranlar için getirileri hesapla
                        returns = np.diff(closes_np) / closes_np[:-1]
                        average_return = np.mean(returns) if len(returns) > 0 else 0
                        stddev_return = np.std(returns) if len(returns) > 0 else 0
                        sharpe_ratio = (average_return - rf) / stddev_return if stddev_return != 0 else np.nan
                        negative_returns = returns[returns < 0]
                        neg_deviation = np.std(negative_returns) if len(negative_returns) > 0 else np.nan
                        sortino_ratio = (average_return - rf) / neg_deviation if neg_deviation not in [0, np.nan] else np.nan
                        # Calmar Ratio
                        roll_max = np.maximum.accumulate(closes_np)
                        drawdowns = (roll_max - closes_np) / roll_max
                        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else np.nan
                        calmar_ratio = average_return / max_drawdown if max_drawdown not in [0, np.nan] else np.nan
                        # Omega Ratio
                        threshold = rf
                        num_omega = np.sum(returns[returns >= threshold] - threshold)
                        denom_omega = np.sum(threshold - returns[returns < threshold])
                        omega_ratio = num_omega / denom_omega if denom_omega != 0 else np.nan
                        # diff_ratio_avg
                        diff_ratio_avg = (
                            (0.40 * sortino_ratio if not np.isnan(sortino_ratio) else 0) +
                            (0.30 * omega_ratio if not np.isnan(omega_ratio) else 0) +
                            (0.20 * calmar_ratio if not np.isnan(calmar_ratio) else 0) +
                            (0.10 * sharpe_ratio if not np.isnan(sharpe_ratio) else 0)
                        )
                        # Momentum ROC hesapla
                        momentum_roc_down, momentum_roc_up = calculate_momentum_roc(df, ema_len=200, ema_roc_len=10)
                        
                        # Son 1 dakikalık değişim yüzdesi
                        last_change_pct = returns[-1] * 100 if len(returns) > 0 else 0
                        
                        results.append({
                            'symbol': symbol,
                            'close': round(last_close, 4),
                            'ema200': round(last_ema, 4),
                            'fark': round(fark_yuzde, 2),
                            'ratio': round(last_close / last_ema, 4),  # Ratio ekle
                            'zaman': zaman,
                            'sharpe': round(sharpe_ratio, 4) if not np.isnan(sharpe_ratio) else None,
                            'sortino': round(sortino_ratio, 4) if not np.isnan(sortino_ratio) else None,
                            'calmar': round(calmar_ratio, 4) if not np.isnan(calmar_ratio) else None,
                            'omega': round(omega_ratio, 4) if not np.isnan(omega_ratio) else None,
                            'diff_ratio_avg': round(diff_ratio_avg, 4),
                            'momentum_roc_down': round(momentum_roc_down, 4) if not np.isnan(momentum_roc_down) else None,
                            'momentum_roc_up': round(momentum_roc_up, 4) if not np.isnan(momentum_roc_up) else None
                        })
                        
                        # Ani yükseliş kontrolü
                        if last_change_pct >= SPIKE_THRESHOLD:
                            spikes.append({
                                'symbol': symbol,
                                'change_pct': round(last_change_pct, 2)
                            })
                        
                        update_price_history(symbol, last_close)  # Fiyat geçmişini güncelle
                        
                processed_count += 1
                if processed_count % 20 == 0:
                    print(f"İşlenen: {processed_count}/{len(symbols)} sembol")
                        
            except Exception as e:
                print(f"Hata ({symbol}): {e}")
                continue
        
        # Z-Score hesapla
        for result in results:
            zscore = calculate_zscore(all_fark_yuzde, result['fark'])
            result['zscore'] = round(zscore, 2)
        
        print(f"Toplam {len(results)} sembol için veri başarıyla alındı")
        return results, spikes
        
    except Exception as e:
        print(f"Genel hata: {e}")
        return [], []

def update_price_history(symbol, price):
    """Fiyat geçmişini günceller"""
    now = datetime.now()
    if symbol not in price_history:
        price_history[symbol] = []
    
    price_history[symbol].append((now, float(price)))
    
    # 1 saatten eski kayıtları temizle
    price_history[symbol] = [
        (ts, p) for ts, p in price_history[symbol] 
        if now - ts < timedelta(hours=1)
    ]

def check_price_alerts():
    """Fiyat değişimlerini kontrol eder ve uyarı oluşturur"""
    global alerts
    while True:
        current_alerts = {}
        now = datetime.now()
        
        for symbol in list(price_history.keys()):
            if not price_history[symbol]:
                continue
                
            # Son fiyat
            last_price = price_history[symbol][-1][1]
            
            # Son 5 dakikadaki en düşük ve en yüksek fiyatları bul
            five_min_ago = now - timedelta(minutes=5)
            prices = [p for ts, p in price_history[symbol] if ts >= five_min_ago]
            
            if len(prices) < 2:  # Yeterli veri yoksa atla
                continue
                
            min_price = min(prices)
            max_price = max(prices)
            current_price = prices[-1]
            
            # Yükseliş uyarısı
            if current_price >= min_price * (1 + alert_threshold/100):
                change_pct = ((current_price - min_price) / min_price) * 100
                current_alerts[symbol] = {
                    'symbol': symbol,
                    'type': 'YÜKSELİŞ',
                    'change': round(change_pct, 2),
                    'time': now.strftime('%H:%M:%S'),
                    'price': current_price,
                    'timestamp': now.timestamp()
                }
            # Düşüş uyarısı
            elif current_price <= max_price * (1 - alert_threshold/100):
                change_pct = ((max_price - current_price) / max_price) * 100
                current_alerts[symbol] = {
                    'symbol': symbol,
                    'type': 'DÜŞÜŞ',
                    'change': round(change_pct, 2),
                    'time': now.strftime('%H:%M:%S'),
                    'price': current_price,
                    'timestamp': now.timestamp()
                }
        
        # Sadece son uyarıları al (son 20 uyarıyı sakla)
        alerts = list(current_alerts.values())[-20:]
        time.sleep(5)  # 5 saniyede bir kontrol et

@app.route('/', methods=['GET', 'POST'])
def home():
    try:
        # Varsayılan değerler
        rff = 0.02
        period = 200
        if request.method == 'POST':
            try:
                rff = float(request.form.get('rff', rff))
            except:
                rff = 0.02
            try:
                period = int(request.form.get('period', period))
                if period < 2:
                    period = 2
                if period > 200:
                    period = 200
            except:
                period = 200
        # REST API ile veri çek
        all_data, spikes = get_binance_data(rf=rff, period=period)
        
        # Verileri veritabanına kaydet
        if all_data:
            db.save_data(all_data)
        
        # EMA 200'ün üzerinde olanları bul
        emtialar = [item for item in all_data if item['close'] > item['ema200']]
        # En yüksek farktan başlayarak ilk 300'ü göster (gerekirse artırılabilir)
        emtialar = sorted(emtialar, key=lambda x: x['fark'], reverse=True)[:300]
        
        # Grafik için en yüksek 100 fark yüzdesi (EMA200 üzeri)
        top100 = sorted(all_data, key=lambda x: x['fark'], reverse=True)[:100]
        chart_series = []
        series_len = 50  # son 50 mum
        for d in top100:
            sym = d['symbol']
            try:
                url = f"https://api.binance.com/api/v3/klines?symbol={sym}&interval=1m&limit={period}"
                resp = requests.get(url, timeout=3)
                if resp.status_code == 200:
                    kl = resp.json()
                    closes_small = np.array([float(k[4]) for k in kl], dtype=np.float64)
                    if closes_small.size < series_len:
                        continue
                    # EMA hesapla (aynı yöntem)
                    ema_small = np.empty_like(closes_small)
                    ema_small[0] = closes_small[0]
                    alpha = 2 / (period + 1)
                    for i in range(1, closes_small.size):
                        ema_small[i] = alpha * closes_small[i] + (1 - alpha) * ema_small[i - 1]
                    ratios = (closes_small / ema_small)[-series_len:]
                    chart_series.append({
                        'label': sym,
                        'data': [round(float(x), 4) for x in ratios]
                    })
            except Exception as _:
                continue
        
        x_axis = list(range(-series_len + 1, 1))  # e.g., -49..0 (relative bars)
        
        # Son güncelleme zamanı
        last_update = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Uyarıları türlerine göre ayır ve sırala
        rising_alerts = sorted(
            [a for a in alerts if a['type'] == 'YÜKSELİŞ'],
            key=lambda x: x['change'],
            reverse=True
        )
        
        falling_alerts = sorted(
            [a for a in alerts if a['type'] == 'DÜŞÜŞ'],
            key=lambda x: x['change'],
            reverse=True
        )
        
        # Son 20 uyarıyı al (yeniden eskiye doğru sırala)
        recent_alerts = sorted(alerts, key=lambda x: x['timestamp'], reverse=True)[:20]
        
        return render_template('index.html', 
                             emtialar=emtialar, 
                             last_update=last_update,
                             rff=rff, 
                             period=period, 
                             chart_series=chart_series, 
                             x_axis=x_axis,
                             spikes=spikes,
                             rising_alerts=rising_alerts,
                             falling_alerts=falling_alerts,
                             recent_alerts=recent_alerts)
    except Exception as e:
        print(f"Ana sayfa hatası: {e}")
        return str(e)

@app.route('/history/<symbol>')
def symbol_history(symbol):
    """Belirli bir sembolün geçmiş verilerini göster"""
    try:
        history = db.get_symbol_history(symbol, days=7)
        return render_template('history.html', symbol=symbol, history=history)
    except Exception as e:
        return f"Hata: {e}"

@app.route('/top_performers')
def top_performers():
    """En iyi performans gösteren sembolleri göster"""
    try:
        top_data = db.get_top_performers(10)
        return render_template('top_performers.html', performers=top_data)
    except Exception as e:
        return f"Hata: {e}"

if __name__ == '__main__':
    print("Flask başlatılıyor...")
    # Alarm kontrol thread'ini başlat
    alert_thread = threading.Thread(target=check_price_alerts, daemon=True)
    alert_thread.start()
    app.run(debug=True, host='127.0.0.1', port=8080)