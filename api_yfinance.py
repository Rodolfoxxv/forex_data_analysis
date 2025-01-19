import duckdb
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import gaussian_kde

# Função para obter os dados do Yahoo Finance
def get_data():
    df = yf.download('EURUSD=X', start='2000-01-01', end=pd.to_datetime('today').strftime('%Y-%m-%d'))
    
    # Corrigir o formato do DataFrame e renomear as colunas
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df.reset_index(inplace=True)
    
    return df

# Função para calcular volatilidade, mudança de preço e frequência de preços
def calculate_metrics(df):
    df['Volatility'] = df['High'] - df['Low']
    df['Price_Change'] = df['Close'].diff().abs()

    # Proxy de atividade no mercado como frequência de preços
    df['Activity'] = (df['Volatility'] + df['Price_Change']).rolling(window=5).mean()
    return df

# Função para identificar níveis de preços com maior densidade
def calculate_price_density(df):
    prices = df['Close'].dropna()
    kde = gaussian_kde(prices)
    df['Price_Density'] = kde(prices)
    return df

# Função para registrar requisição na tabela api_requests
def log_api_request(conn):
    today = pd.Timestamp.now().date()
    # Verificar se já há registro para o dia atual
    result = conn.execute(""" 
        SELECT COUNT(*) FROM api_requests WHERE request_date = ? 
    """, (today,)).fetchone()[0]
    
    if result == 0:
        # Inserir novo registro para o dia
        conn.execute(""" 
            INSERT INTO api_requests (request_date, request_count)
            VALUES (?, 1)
        """, (today,))
    else:
        # Incrementar o contador de requisições do dia
        conn.execute(""" 
            UPDATE api_requests
            SET request_count = request_count + 1
            WHERE request_date = ? 
        """, (today,))

# Função para atualizar o banco de dados
def update_database():
    # Conectar ao banco DuckDB
    conn = duckdb.connect('forex_data.duckdb')
    
    # Criar a tabela de dados, se ainda não existir (incluindo novos campos)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS eur_usd_yf (
            date DATE PRIMARY KEY,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            volatility DOUBLE,
            price_change DOUBLE,
            activity DOUBLE,
            price_density DOUBLE
        );
    """)
    
    # Criar a tabela para contar requisições, se ainda não existir
    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_requests (
            request_date DATE PRIMARY KEY,
            request_count INTEGER
        );
    """)

    # Adicionar as colunas ausentes, se necessário
    try:
        conn.execute("ALTER TABLE eur_usd_yf ADD COLUMN activity DOUBLE;")
    except duckdb.BinderException:
        print("A coluna 'activity' já existe.")

    try:
        conn.execute("ALTER TABLE eur_usd_yf ADD COLUMN price_density DOUBLE;")
    except duckdb.BinderException:
        print("A coluna 'price_density' já existe.")
    
    # Registrar a requisição à API
    log_api_request(conn)
    
    # Obter os dados da API e calcular métricas
    new_data = get_data()
    new_data = calculate_metrics(new_data)
    new_data = calculate_price_density(new_data)
    
    # Tornar o índice de 'new_data' timezone-naive
    new_data['Date'] = pd.to_datetime(new_data['Date']).dt.date
    
    # Apagar dados antigos da tabela antes de carregar novos
    conn.execute("DELETE FROM eur_usd_yf;")
    
    # Inserir os novos dados na tabela
    for index, row in new_data.iterrows():
        conn.execute("""
            INSERT INTO eur_usd_yf (date, open, high, low, close, volume, volatility, price_change, activity, price_density)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (date) DO NOTHING;
        """, tuple([row['Date'], row['Open'], row['High'], row['Low'], row['Close'], row['Volume'], row['Volatility'], row['Price_Change'], row['Activity'], row['Price_Density']]))
    
    print(f"{len(new_data)} novos registros adicionados ao banco de dados.")
    
    conn.close()

if __name__ == "__main__":
    update_database()
