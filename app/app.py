import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# 1. Configuração da Página
st.set_page_config(page_title="Análise de Homicídios", layout="wide")
st.title("Análise Exploratória e Regressão: Dados de Homicídios (UNODC)")
st.write("Trabalho da disciplina Tópicos Especiais para Computação - Prof. Iális Cavalcante")

# 2. Carregamento de Dados (com cache para o app não ficar lento)
@st.cache_data
def load_data():
    # Lendo o arquivo CSV
    df_counts = pd.read_csv('data_cts_intentional_homicide.csv', skiprows=2, low_memory=False)

    # Criando o DataFrame limpo (Filtra apenas contagens absolutas e evita dupla contagem)
    df_limpo = df_counts[
        (df_counts['Unit of measurement'] == 'Counts') &
        (df_counts['Dimension'] == 'Total') &
        (df_counts['Category'] == 'Total') &
        (df_counts['Age'] == 'Total')
    ].copy()

    # Garante que a coluna 'VALUE' é tratada como número
    df_limpo['VALUE'] = pd.to_numeric(df_limpo['VALUE'], errors='coerce')   
        
    return df_limpo

df_counts = load_data()

# 3. Seção de Análise Exploratória 
st.header("1. Análise Exploratória (EDA)")

st.subheader("Pergunta 1: Quais países apresentam os 10 maiores índices de homicídios nos últimos 5 anos?")
df_ultimos_5_anos = df_counts[(df_counts['Year'] >= 2018) & (df_counts['Year'] <= 2022) & (df_counts['Sex'] == 'Total')]
top_10_paises = df_ultimos_5_anos.groupby('Country')['VALUE'].sum().nlargest(10).reset_index()
top_10_paises.columns = ['País', 'Total de Homicídios (2018-2022)']
st.bar_chart(top_10_paises.set_index('País')) # Exibe o gráfico de barras
st.write("---")

st.subheader("Pergunta 2: 10 maiores índices de homicídios de mulheres em 2022")
df_mulheres_2022 = df_counts[(df_counts['Year'] == 2022) & (df_counts['Sex'].str.contains('Female', na=False, case=False))]
top_10_mulheres_2022 = df_mulheres_2022.groupby('Country')['VALUE'].sum().nlargest(10)
st.bar_chart(top_10_mulheres_2022)
st.write("---")

st.subheader("Pergunta 3: as regiões com mais homicídios")
df_totais = df_counts[df_counts['Sex'] == 'Total']
regioes_maiores = df_totais.groupby('Region')['VALUE'].sum().sort_values(ascending=False).reset_index()
regioes_maiores.columns = ['Região', 'Total Histórico de Homicídios']
st.bar_chart(regioes_maiores.set_index('Região'))
st.write("---")

st.subheader("Pergunta 4: países com menor número de homicídios em cada subregião")
df_totais = df_counts[df_counts['Sex'] == 'Total']
# Agrupamos por subregião e país para ter a soma total de cada país
somas_subregiao = df_totais.groupby(['Subregion', 'Country'])['VALUE'].sum().reset_index()
# Encontramos o índice da linha com o menor valor dentro de cada subregião
indices_menores = somas_subregiao.groupby('Subregion')['VALUE'].idxmin()
# Filtramos a tabela original usando esses índices
menores_por_subregiao = somas_subregiao.loc[indices_menores].reset_index(drop=True)
menores_por_subregiao.columns = ['Subregião', 'País', 'Total de Homicídios']
st.dataframe(menores_por_subregiao)
st.write("---")

st.subheader("Pergunta 5: países com menor número de morte de mulheres")
df_mulheres = df_counts[df_counts['Sex'].str.contains('Female', na=False, case=False)]
menor_mulheres = df_mulheres.groupby('Country')['VALUE'].sum().nsmallest(5)
st.dataframe(menor_mulheres)
st.write("---")

st.subheader("Pergunta 6: as subregiões com maior número de homicídios")
subregioes_maiores = df_totais.groupby('Subregion')['VALUE'].sum().nlargest(5).reset_index()
subregioes_maiores.columns = ['Subregião', 'Total Histórico de Homicídios']
st.bar_chart(subregioes_maiores.set_index('Subregião'))
st.write("---")

st.subheader("Pergunta 7: Identifique o país com maior número de homicídios em cada continente em 2020")
# Filtramos o ano de 2020 E a trava de Sexo='Total'
df_2020 = df_counts[(df_counts['Year'] == 2020) & (df_counts['Sex'] == 'Total')]
# Agrupamos por Região e País para ter as somas
somas_2020 = df_2020.groupby(['Region', 'Country'])['VALUE'].sum().reset_index()
# Identificamos os índices dos maiores valores por região
indices_maiores = somas_2020.groupby('Region')['VALUE'].idxmax()
maiores_por_regiao = somas_2020.loc[indices_maiores].reset_index(drop=True)
maiores_por_regiao.columns = ['Continente (Região)', 'País', 'Homicídios em 2020']
st.dataframe(maiores_por_regiao)
st.write("---")

st.subheader("Pergunta 8: o país mais violento para as mulheres em 2021")
# Filtramos o ano de 2021 e o sexo feminino
df_mulheres_2021 = df_counts[(df_counts['Year'] == 2021) & (df_counts['Sex'] == 'Female')]

# Somamos os valores por país e pegamos o 1º maior. 
# O .reset_index() transforma a resposta de uma "Série" para um "DataFrame" estruturado.
pais_mais_violento_2021 = df_mulheres_2021.groupby('Country')['VALUE'].sum().nlargest(1).reset_index()

# Renomeamos as colunas para a tabela ficar com uma apresentação mais profissional no app
pais_mais_violento_2021.columns = ['País', 'Total de Vítimas (Mulheres em 2021)']

# Exibimos o resultado como tabela na tela
st.dataframe(pais_mais_violento_2021)
st.write("---")

st.subheader("Pergunta 9: o país com maior valor de 'indicador: Victims of intentional homicide'")
df_indicador = df_counts[(df_counts['Indicator'].str.contains('Victims of intentional homicide', na=False, case=False)) & (df_counts['Sex'] == 'Total')]
maior_indicador_pais = df_indicador.groupby('Country')['VALUE'].sum().nlargest(1).reset_index()
maior_indicador_pais.columns = ['País', 'Total de Vítimas (Homicídio Intencional)']
st.dataframe(maior_indicador_pais)
st.write("---")

st.subheader("Pergunta 10: a média de homicídios no Brasil nos últimos 10 anos")
# Filtramos o país Brasil E apenas a categoria 'Total' na coluna Sexo para evitar dupla contagem
df_brasil = df_counts[(df_counts['Country'] == 'Brazil') & (df_counts['Sex'] == 'Total') & (df_counts['Year'] >= 2013) & (df_counts['Year'] <= 2022)]
media_brasil = df_brasil.groupby('Year')['VALUE'].sum().mean()
# Construímos a tabela 
tabela_media_brasil = pd.DataFrame({
    'País': ['Brasil'],
    'Média Anual de Homicídios (2013-2022)': [media_brasil]
})
st.dataframe(tabela_media_brasil)
st.write("---")




# 4. Seção do Modelo de Regressão
st.header("2. Modelo de Regressão (Predição 2023-2026)")

# Agrupando dados globais por ano, garantindo que pegamos apenas a contagem total
df_tendencia = df_counts[
    (df_counts['Sex'] == 'Total') & 
    (df_counts['Year'] <= 2022) 
].groupby('Year')['VALUE'].sum().reset_index()

# Preparando as variáveis independentes (X) e dependentes (y)
X = df_tendencia[['Year']]
y = df_tendencia['VALUE']

# Treinando o modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Criando os anos futuros para predição
anos_futuros = pd.DataFrame({'Year': [2023, 2024, 2025, 2026]})
predicoes = modelo.predict(anos_futuros)

# Configurando o gráfico
fig, ax = plt.subplots(figsize=(10, 5))

# Plotando os dados históricos e reais 
ax.plot(df_tendencia['Year'], df_tendencia['VALUE'], label='Histórico Global (Até 2022)', marker='o', color='royalblue')

# Plotando a predição gerada pelo Machine Learning 
ax.plot(anos_futuros['Year'], predicoes, label='Predição ML (2023-2026)', marker='x', color='crimson', linestyle='--')

ax.set_title("Tendência Global de Homicídios e Predição Futura")
ax.set_xlabel("Ano")
ax.set_ylabel("Total de Homicídios Absolutos")
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend()

# Exibindo o gráfico no aplicativo web
st.pyplot(fig)