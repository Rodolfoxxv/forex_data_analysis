import duckdb
import pandas as pd
from joblib import load
import numpy as np

# Carregar o modelo salvo
modelo = load('modelo_random_forest_final.joblib')

# Função para carregar dados
def carregar_dados(conn, start_date, end_date):
    query = f"""
    SELECT date, open, high, low, close, volume, volatility, price_change, activity, price_density
    FROM eur_usd_yf
    WHERE date BETWEEN '{start_date}' AND '{end_date}';
    """
    return conn.execute(query).fetch_df()

# Função para preprocessar os dados
def preprocessar_dados(data):
    data['date'] = pd.to_datetime(data['date'])
    data['day_of_week'] = data['date'].dt.dayofweek
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year
    
    # Criar variáveis de lag e médias móveis
    data['close_lag1'] = data['close'].shift(1)
    data['close_rolling_mean5'] = data['close'].rolling(window=5).mean()

    data.dropna(inplace=True)  # Remover linhas com valores NaN

    return data

# Função para gerar previsões para o dia 20 de janeiro de 2025
def prever_para_20_jan(dados_janeiro):
    # Filtrar dados até o dia 19 de janeiro (ignorando finais de semana)
    dados_janeiro = dados_janeiro[~dados_janeiro['date'].dt.weekday.isin([5, 6])]  # Remover finais de semana
    dados_janeiro = dados_janeiro[dados_janeiro['date'] <= '2025-01-19']

    # Preprocessar os dados
    dados_janeiro = preprocessar_dados(dados_janeiro)

    # Selecionar as variáveis preditoras
    X = dados_janeiro[['open', 'high', 'low', 'close', 'volume', 'volatility', 'price_change', 'activity', 'price_density',
                       'day_of_week', 'month', 'year', 'close_lag1', 'close_rolling_mean5']]

    # Gerar previsões para os últimos dias (utilizando o histórico até o dia 19)
    prev_serie = modelo.predict(X)

    # Obter o valor de fechamento do dia 19 e desvio padrão
    ultimo_dado = dados_janeiro.tail(1)
    valor_fechamento = ultimo_dado['close'].values[0]
    
    # Calcular o desvio padrão para os dados de fechamento
    desvio_padrao = np.std(dados_janeiro['close'].values)

    # Prever o fechamento para o dia 20 (com base nas previsões da série)
    previsao_fechar_20 = valor_fechamento + (np.mean(prev_serie) * desvio_padrao)

    # Gerar a previsão final (fechamento estimado e desvio padrão)
    previsao = pd.DataFrame({
        'date': ['2025-01-20'],
        'close': [previsao_fechar_20],  # Previsão ajustada para o fechamento
        'desvio_padrao': [desvio_padrao],
        'previsao': [np.mean(prev_serie)],  # Média das previsões
        'lower_bound': [previsao_fechar_20 - desvio_padrao],  # Limite inferior (-1 desvio padrão)
        'upper_bound': [previsao_fechar_20 + desvio_padrao]   # Limite superior (+1 desvio padrão)
    })

    # Salvar as previsões em um arquivo CSV
    previsao.to_csv('previsao_janeiro_2025.csv', index=False)

    print("Previsões salvas em 'previsao_janeiro_2025.csv'")

# Função principal
def main():
    # Conectar ao banco DuckDB
    conn = duckdb.connect('forex_data.duckdb')

    # Carregar dados de janeiro de 2025 (até o dia 19)
    dados_janeiro = carregar_dados(conn, '2025-01-01', '2025-01-19')

    # Gerar previsões para o dia 20 de janeiro de 2025
    prever_para_20_jan(dados_janeiro)

if __name__ == "__main__":
    main()
