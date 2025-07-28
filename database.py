import sqlite3
import pandas as pd
from datetime import datetime
import os

class Database:
    def __init__(self, db_name='crypto_data.db'):
        self.db_name = db_name
        self.init_database()
    
    def init_database(self):
        """Veritabanını ve tabloları oluştur"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Ana veri tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS crypto_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                close_price REAL NOT NULL,
                ema200 REAL NOT NULL,
                fark_yuzde REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Sembol indeksi oluştur
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
            ON crypto_data(symbol, timestamp)
        ''')
        
        conn.commit()
        conn.close()
        print(f"Veritabanı başlatıldı: {self.db_name}")
    
    def save_data(self, data_list):
        """Veri listesini veritabanına kaydet"""
        # Tablo yoksa oluştur (örneğin farklı klasörden çalıştırıldığında)
        self.init_database()
        
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for data in data_list:
            cursor.execute('''
                INSERT INTO crypto_data (symbol, close_price, ema200, fark_yuzde, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                data['symbol'],
                data['close'],
                data['ema200'],
                data['fark'],
                current_time
            ))
        
        conn.commit()
        conn.close()
        print(f"{len(data_list)} kayıt veritabanına kaydedildi")
    
    def get_latest_data(self, limit=20):
        """En son verileri getir"""
        conn = sqlite3.connect(self.db_name)
        
        query = '''
            SELECT symbol, close_price, ema200, fark_yuzde, timestamp
            FROM crypto_data 
            WHERE timestamp = (SELECT MAX(timestamp) FROM crypto_data)
            ORDER BY fark_yuzde DESC
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        # DataFrame'i dict listesine çevir
        results = []
        for _, row in df.iterrows():
            results.append({
                'symbol': row['symbol'],
                'close': row['close_price'],
                'ema200': row['ema200'],
                'fark': row['fark_yuzde'],
                'zaman': row['timestamp']
            })
        
        return results
    
    def get_symbol_history(self, symbol, days=7):
        """Belirli bir sembolün geçmiş verilerini getir"""
        conn = sqlite3.connect(self.db_name)
        
        query = '''
            SELECT symbol, close_price, ema200, fark_yuzde, timestamp
            FROM crypto_data 
            WHERE symbol = ? 
            AND timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp DESC
        '''.format(days)
        
        df = pd.read_sql_query(query, conn, params=(symbol,))
        conn.close()
        
        return df
    
    def get_top_performers(self, limit=10):
        """En iyi performans gösteren sembolleri getir"""
        conn = sqlite3.connect(self.db_name)
        
        query = '''
            SELECT symbol, AVG(fark_yuzde) as avg_performance, COUNT(*) as data_count
            FROM crypto_data 
            WHERE timestamp >= datetime('now', '-1 day')
            GROUP BY symbol
            HAVING data_count >= 10
            ORDER BY avg_performance DESC
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        return df

# Global veritabanı instance'ı
db = Database() 