import duckdb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump
from datetime import timedelta
from sklearn.model_selection import cross_val_score

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

    # Variável alvo: aumento/diminuição do preço
    data['Target'] = (data['close'].diff() > 0).astype(int)
    data.dropna(inplace=True)  # Remover linhas com valores NaN

    return data

# Função para treinar o modelo com warm_start
def treinar_modelo(data, modelo_salvo=None):
    # Dividir os dados em preditores (X) e alvo (y)
    X = data[['open', 'high', 'low', 'close', 'volume', 'volatility', 'price_change', 'activity', 'price_density',
              'day_of_week', 'month', 'year', 'close_lag1', 'close_rolling_mean5']]
    y = data['Target']

    # Criar e treinar o modelo Random Forest com warm_start para aprendizado progressivo
    if modelo_salvo is None:
        model = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=10, min_samples_split=5, 
                                       min_samples_leaf=4, n_estimators=100, warm_start=True)
    else:
        model = modelo_salvo

    # Se já estiver treinando com warm_start, incrementar n_estimators para adicionar novas árvores
    model.n_estimators += 50  # Incrementa o número de estimadores em 50 a cada iteração
    model.fit(X, y)

    # Usar validação cruzada para avaliar o modelo
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Validação Cruzada (Acurácia Média): {cv_scores.mean():.4f}")
    
    return model

# Função para avaliar o modelo
def avaliar_modelo(model, data):
    X = data[['open', 'high', 'low', 'close', 'volume', 'volatility', 'price_change', 'activity', 'price_density',
              'day_of_week', 'month', 'year', 'close_lag1', 'close_rolling_mean5']]
    y = data['Target']
    y_pred = model.predict(X)

    # Calcular a acurácia
    acuracia = accuracy_score(y, y_pred)
    print(f"Acurácia: {acuracia:.4f}")
    
    # Exibir matriz de confusão
    print("Matriz de Confusão:")
    print(confusion_matrix(y, y_pred))
    
    # Exibir o relatório de classificação
    print("\nRelatório de Classificação:")
    print(classification_report(y, y_pred))

# Função para dividir os dados em períodos homogêneos (intervalos com aproximadamente 3 anos)
def dividir_periodos_homogeneos(data):
    # Ordenar os dados pela data
    data = data.sort_values(by='date')
    
    # Calcular a data inicial e final
    start_date = data['date'].min()
    end_date = data['date'].max()

    # Calcular a quantidade total de anos
    total_anos = (end_date - start_date).days / 365.25
    
    # Calcular a quantidade de períodos (dividindo a duração total por 3 anos)
    num_periodos = total_anos / 3  # Pode não ser um valor inteiro
    
    # Criar lista de intervalos
    periodos = []
    current_start = start_date

    for _ in range(int(num_periodos)):
        # Calcula o final do período (pode não ser exatamente 3 anos)
        current_end = current_start + timedelta(days=365.25 * 3)
        if current_end > end_date:
            current_end = end_date
        periodos.append((current_start, current_end))
        current_start = current_end

    # Adicionar o último período para garantir que todos os dados sejam usados
    if current_start < end_date:
        periodos.append((current_start, end_date))
    
    return periodos

# Função principal
def main():
    # Conectar ao banco DuckDB
    conn = duckdb.connect('forex_data.duckdb')
    
    # Carregar os dados completos (todo o conjunto)
    dados_completos = carregar_dados(conn, '2003-12-01', '2025-01-01')
    dados_completos = preprocessar_dados(dados_completos)

    # Dividir os dados em períodos homogêneos
    periodos = dividir_periodos_homogeneos(dados_completos)
    
    modelo_atual = None  # Iniciar o modelo como None
    
    for i, (start_date, end_date) in enumerate(periodos):
        print(f"\nTreinando modelo para o período {start_date.date()} a {end_date.date()}")
        
        # Filtrar os dados para o período atual
        dados_periodo = dados_completos[(dados_completos['date'] >= start_date) & (dados_completos['date'] <= end_date)]
        
        # Treinar o modelo para o período atual com aprendizado progressivo
        modelo_atual = treinar_modelo(dados_periodo, modelo_salvo=modelo_atual)
        
        # Avaliar o modelo
        avaliar_modelo(modelo_atual, dados_periodo)

    # Salvar o modelo final após o treinamento de todos os períodos
    dump(modelo_atual, 'modelo_random_forest_final.joblib')
    print(f"Modelo final salvo como 'modelo_random_forest_final.joblib'")

if __name__ == "__main__":
    main()
