import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import yfinance as yf
import os
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Criar pasta de graficos se nao existir
if not os.path.exists('graficos'):
    os.makedirs('graficos')

# Configuracoes de estilo (Clean e Profissional)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F8F9FA",
    "axes.grid":        True,
    "grid.color":       "white",
    "font.family":      "sans-serif",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# 1. Coleta de Dados Reais
tickers = [
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "WEGE3.SA",
    "RENT3.SA", "MGLU3.SA", "ABEV3.SA", "BBAS3.SA", "^BVSP"
]

print("Baixando dados do Yahoo Finance...")
df = yf.download(tickers, start="2022-01-01", end="2024-12-31")["Close"]

# Renomear colunas para ficar mais limpo
df.columns = [c.replace(".SA", "").replace("^BVSP", "BOVA11") for c in df.columns]
df = df.dropna()

# 2. Calculos de Retorno e Risco
retornos = df.pct_change().dropna()
retorno_acumulado = (1 + retornos).cumprod() - 1

vol_anual = retornos.std() * np.sqrt(252) * 100
retorno_total = retorno_acumulado.iloc[-1] * 100
sharpe = (retornos.mean() * 252) / (retornos.std() * np.sqrt(252))

# 3. Geracao dos Graficos
PALETTE = ["#1F5C99","#E05C2A","#2E9E6B","#9B59B6","#E8A020","#2980B9","#C0392B","#27AE60","#8E44AD","#D35400"]

# Grafico 1: Retorno Acumulado
plt.figure(figsize=(12, 6))
for i, col in enumerate(df.columns):
    if col == "BOVA11":
        plt.plot(retorno_acumulado[col] * 100, color="black", lw=3, label="Ibovespa (BOVA11)", ls="--")
    else:
        plt.plot(retorno_acumulado[col] * 100, label=col, alpha=0.8, lw=1.5)

plt.title("Performance Acumulada: Ativos B3 vs Ibovespa", fontsize=14, weight='bold')
plt.ylabel("Retorno (%)")
plt.legend(loc="upper left", ncol=2)
plt.savefig("graficos/01_retorno_acumulado.png", dpi=150)
print("Gerado: 01_retorno_acumulado.png")

# Grafico 2: Risco x Retorno
plt.figure(figsize=(10, 6))
for i, txt in enumerate(df.columns):
    plt.scatter(vol_anual[i], retorno_total[i], s=100)
    plt.annotate(txt, (vol_anual[i], retorno_total[i]), xytext=(5,5), textcoords='offset points')

plt.axvline(vol_anual["BOVA11"], color="red", ls=":", alpha=0.5)
plt.title("Relação Risco x Retorno (2022-2024)")
plt.xlabel("Volatilidade Anualizada (%)")
plt.ylabel("Retorno Total (%)")
plt.savefig("graficos/02_risco_retorno.png")
print("Gerado: 02_risco_retorno.png")

# Grafico 3: Heatmap de Correlacao
plt.figure(figsize=(10, 8))
sns.heatmap(retornos.corr(), annot=True, cmap="RdYlGn", fmt=".2f", linewidths=0.5)
plt.title("Matriz de Correlação dos Ativos")
plt.savefig("graficos/03_correlacao.png")
print("Gerado: 03_correlacao.png")

print("\nProcesso finalizado com sucesso!")