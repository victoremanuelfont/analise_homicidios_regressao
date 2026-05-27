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

# 3. Navegação Lateral (Sidebar)
st.sidebar.header("Navegação")
menu = [
    "Início",
    "Pergunta 1", "Pergunta 2", "Pergunta 3", "Pergunta 4", "Pergunta 5",
    "Pergunta 6", "Pergunta 7", "Pergunta 8", "Pergunta 9", "Pergunta 10",
    "Modelo de Regressão"
]
escolha = st.sidebar.selectbox("Ir para:", menu)

# --- Conteúdo Principal ---

if escolha == "Início":
    st.header("Equipe do Projeto")
    st.write("""
    * Alisson dos Santos Nascimento
    * Arthur Vieira de Lacerda
    * Eduardo Santos de Castro
    * Jonathan Fernandes da Costa
    * Marina Paula Fontenele 
    * Ryan Gomes Magalhães Lima
    * Victor Emanuel Fontenele Lima
    """)
    
    st.divider()
    
    st.header("Bem-vindo!")
    st.write("""
    Utilize o menu lateral para navegar entre as 10 questões da Análise Exploratória de Dados (EDA) 
    e visualizar o Modelo de Regressão Linear com as predições para 2023-2026.
    """)
    st.info("Dica: A navegação lateral permite acesso rápido a qualquer parte do trabalho.")

elif escolha == "Pergunta 1":
    st.subheader("Pergunta 1: Quais países apresentam os 10 maiores índices de homicídios nos últimos 5 anos?")
    df_ultimos_5_anos = df_counts[(df_counts['Year'] >= 2018) & (df_counts['Year'] <= 2022) & (df_counts['Sex'] == 'Total')]
    top_10_paises = df_ultimos_5_anos.groupby('Country')['VALUE'].sum().nlargest(10).reset_index()
    top_10_paises.columns = ['País', 'Total de Homicídios (2018-2022)']
    st.bar_chart(top_10_paises.set_index('País'))

elif escolha == "Pergunta 2":
    st.subheader("Pergunta 2: 10 maiores índices de homicídios de mulheres em 2022")
    df_mulheres_2022 = df_counts[(df_counts['Year'] == 2022) & (df_counts['Sex'].str.contains('Female', na=False, case=False))]
    top_10_mulheres_2022 = df_mulheres_2022.groupby('Country')['VALUE'].sum().nlargest(10)
    st.bar_chart(top_10_mulheres_2022)

elif escolha == "Pergunta 3":
    st.subheader("Pergunta 3: as regiões com mais homicídios")
    df_totais = df_counts[df_counts['Sex'] == 'Total']
    regioes_maiores = df_totais.groupby('Region')['VALUE'].sum().sort_values(ascending=False).reset_index()
    regioes_maiores.columns = ['Região', 'Total Histórico de Homicídios']
    st.bar_chart(regioes_maiores.set_index('Região'))

elif escolha == "Pergunta 4":
    st.subheader("Pergunta 4: países com menor número de homicídios em cada subregião")
    df_totais = df_counts[df_counts['Sex'] == 'Total']
    somas_subregiao = df_totais.groupby(['Subregion', 'Country'])['VALUE'].sum().reset_index()
    indices_menores = somas_subregiao.groupby('Subregion')['VALUE'].idxmin()
    menores_por_subregiao = somas_subregiao.loc[indices_menores].reset_index(drop=True)
    menores_por_subregiao.columns = ['Subregião', 'País', 'Total de Homicídios']
    st.dataframe(menores_por_subregiao, use_container_width=True)

elif escolha == "Pergunta 5":
    st.subheader("Pergunta 5: países com menor número de morte de mulheres")
    df_mulheres = df_counts[df_counts['Sex'].str.contains('Female', na=False, case=False)]
    menor_mulheres = df_mulheres.groupby('Country')['VALUE'].sum().nsmallest(5).reset_index()
    menor_mulheres.columns = ['País', 'Total de Homicídios (Feminino)']
    st.dataframe(menor_mulheres, use_container_width=True)

elif escolha == "Pergunta 6":
    st.subheader("Pergunta 6: as subregiões com maior número de homicídios")
    df_totais = df_counts[df_counts['Sex'] == 'Total']
    subregioes_maiores = df_totais.groupby('Subregion')['VALUE'].sum().nlargest(5).reset_index()
    subregioes_maiores.columns = ['Subregião', 'Total Histórico de Homicídios']
    st.bar_chart(subregioes_maiores.set_index('Subregião'))

elif escolha == "Pergunta 7":
    st.subheader("Pergunta 7: Identifique o país com maior número de homicídios em cada continente em 2020")
    df_2020 = df_counts[(df_counts['Year'] == 2020) & (df_counts['Sex'] == 'Total')]
    somas_2020 = df_2020.groupby(['Region', 'Country'])['VALUE'].sum().reset_index()
    indices_maiores = somas_2020.groupby('Region')['VALUE'].idxmax()
    maiores_por_regiao = somas_2020.loc[indices_maiores].reset_index(drop=True)
    maiores_por_regiao.columns = ['Continente (Região)', 'País', 'Homicídios em 2020']
    st.dataframe(maiores_por_regiao, use_container_width=True)

elif escolha == "Pergunta 8":
    st.subheader("Pergunta 8: o país mais violento para as mulheres em 2021")
    df_mulheres_2021 = df_counts[(df_counts['Year'] == 2021) & (df_counts['Sex'] == 'Female')]
    pais_mais_violento_2021 = df_mulheres_2021.groupby('Country')['VALUE'].sum().nlargest(1).reset_index()
    pais_mais_violento_2021.columns = ['País', 'Total de Vítimas (Mulheres em 2021)']
    st.dataframe(pais_mais_violento_2021, use_container_width=True)

elif escolha == "Pergunta 9":
    st.subheader("Pergunta 9: o país com maior valor de 'indicador: Victims of intentional homicide'")
    df_indicador = df_counts[(df_counts['Indicator'].str.contains('Victims of intentional homicide', na=False, case=False)) & (df_counts['Sex'] == 'Total')]
    maior_indicador_pais = df_indicador.groupby('Country')['VALUE'].sum().nlargest(1).reset_index()
    maior_indicador_pais.columns = ['País', 'Total de Vítimas (Homicídio Intencional)']
    st.dataframe(maior_indicador_pais, use_container_width=True)

elif escolha == "Pergunta 10":
    st.subheader("Pergunta 10: a média de homicídios no Brasil nos últimos 10 anos")
    df_brasil = df_counts[(df_counts['Country'] == 'Brazil') & (df_counts['Sex'] == 'Total') & (df_counts['Year'] >= 2013) & (df_counts['Year'] <= 2022)]
    media_brasil = df_brasil.groupby('Year')['VALUE'].sum().mean()
    tabela_media_brasil = pd.DataFrame({
        'País': ['Brasil'],
        'Média Anual de Homicídios (2013-2022)': [media_brasil]
    })
    st.dataframe(tabela_media_brasil, use_container_width=True)

elif escolha == "Modelo de Regressão":
    st.header("2. Modelo de Regressão (Predição 2023-2026)")

    # Dados Históricos
    df_tendencia = df_counts[
        (df_counts['Sex'] == 'Total') & 
        (df_counts['Year'] <= 2022) 
    ].groupby('Year')['VALUE'].sum().reset_index()

    X = df_tendencia[['Year']]
    y = df_tendencia['VALUE']

    # Modelo
    modelo = LinearRegression()
    modelo.fit(X, y)

    # Predição
    anos_futuros = [2023, 2024, 2025, 2026]
    df_futuro = pd.DataFrame({'Year': anos_futuros})
    predicoes = modelo.predict(df_futuro)

    # Preparando dados para o gráfico interativo do Streamlit
    # Criamos um DataFrame que une o Histórico e a Predição
    df_historico_plot = df_tendencia.copy()
    df_historico_plot['Tipo'] = 'Histórico'

    df_predicao_plot = pd.DataFrame({
        'Year': anos_futuros,
        'VALUE': predicoes,
        'Tipo': 'Predição'
    })

    # Unindo as tabelas
    df_final_plot = pd.concat([df_historico_plot, df_predicao_plot])

    # No Streamlit, st.line_chart funciona melhor com o índice sendo o eixo X
    chart_data = df_final_plot.pivot(index='Year', columns='Tipo', values='VALUE')

    # Definindo cores personalizadas: Azul para Histórico, Laranja/Vermelho para Predição
    st.line_chart(chart_data, color=["#1f77b4", "#ff7f0e"])

    st.write("O gráfico acima mostra a tendência histórica (em azul) e a projeção calculada pelo modelo de Regressão Linear para os próximos anos (em laranja).")