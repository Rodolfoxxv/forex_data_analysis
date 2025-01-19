# Previsão de Câmbio com Random Forest

Este projeto visa prever o comportamento do mercado Forex (EUR/USD) utilizando aprendizado de máquina, especificamente o modelo **Random Forest**, para prever a direção dos preços. O processo envolve treinar o modelo com dados históricos e, em seguida, realizar previsões de fechamento de preços com base nesses dados.

## Estrutura do Projeto

Este projeto contém três scripts principais:

1. **`ml_model/modelo_forex_rf.py`**: Script de treinamento do modelo **Random Forest** para prever a direção dos preços.
2. **`scripts/modelo_previsao_forex.py`**: Script para carregar o modelo treinado e fazer previsões com base em dados recentes.
3. **`api_yfinance.py`**: Script auxiliar para obter dados históricos do mercado Forex através da API do Yahoo Finance (caso queira obter dados mais atualizados).
4. **`LICENSE`**: Arquivo de licença (MIT License).
5. **`README.md`**: Este arquivo, que explica o projeto.

## Como Usar

### 1. **Clone o repositório**

Primeiro, clone o repositório para seu ambiente local:

```bash
git clone <https://github.com/Rodolfoxxv/forex_data_analysis.git>
