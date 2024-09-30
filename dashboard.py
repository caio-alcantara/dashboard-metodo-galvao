import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Função para carregar o modelo de anomalias
@st.cache_resource
def load_anomaly_model():
    return joblib.load('iso_forest_tuning_model.pkl')

# Função para escalar os dados de entrada
@st.cache_data
def scale_data(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled

# Função para fazer predições de anomalias
def predict_anomalies(model, df_scaled):
    labels = model.predict(df_scaled)
    return pd.Series(labels).map({1: 0, -1: 1})

# Interface da Dashboard com Abas

st.image("galvao.png", width=250)

st.markdown("<h1 style='text-align: left;'><span style='color: #c9a487;'>Método Galvão</span> para identificação de anomalias e previsão de consumo de gás</h1>", unsafe_allow_html=True)
st.markdown("### Utilize as abas abaixo para conhecer os objetivos da dashboard, analisar anomalias ou fazer predições de consumo.")

# Definindo abas
tab1, tab2, tab3 = st.tabs(["Apresentação", "Detecção de Anomalias", "Predição de Consumo"])

# Adicionando estilo CSS para centralizar as abas
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Aba 1: Apresentação
with tab1:
    st.header("Objetivos da Dashboard")
    st.write("""
    Esta dashboard foi desenvolvida com a finalidade de tornar mais fácil a identificação de anomalias no consumo de gás de clientes e a realização de previsões de consumo para clientes individuais.                      
    
    **Objetivos:**
    - **Detecção de Anomalias**: Utilizar o modelo Isolation Forest para identificar comportamentos anômalos nos dados de consumo de gás.
    - **Predição de Consumo**: Utilizar o modelo Holt-Winters para prever o consumo de gás de um cliente individual com base no histórico de consumo.
    
    **Aviso Importante**: Os arquivos CSV que devem ser subidos nessa dashboard devem ser aqueles exportados ao final da execução do notebook do projeto.
    
    Navegue pelas abas acima para explorar as funcionalidades da dashboard.
    """)

    st.header("Sobre o Método Galvão")
    st.write("""
    O Método Galvão é uma técnica de análise de dados que utiliza modelos de Machine Learning para identificar anomalias e prever o consumo de gás de clientes.
    Foi desenvolvido pela equipe Galvão & Associados Gases e Dados, com o objetivo de melhorar a eficiência na gestão de consumo de gás.
    
    A equipe é composta por Caio de Alcantara Santos, Cecília Beatriz Melo Galvão, Pablo de Azevedo, Lucas Cozzolino Tort, Nataly de Souza Cunha, Kethlen Martins da Silva, Mariella Sayumi Mercado Kamezawa.
    """)

# Aba 2: Detecção de Anomalias
import matplotlib.pyplot as plt

# Aba 2: Detecção de Anomalias
with tab2:
    st.header("Análise de Anomalias com Isolation Forest")

    st.write("""
    Esta ferramenta usa o modelo **Isolation Forest** para detectar anomalias nos dados de consumo de gás. 
    Faça o upload do arquivo CSV disponibilizado após executar o notebook do projeto (df_sample_selecionado.csv).
    """)

    # Carregar o modelo de anomalias
    iso_forest_model = load_anomaly_model()

    # Upload de arquivo
    uploaded_file = st.file_uploader("Carregar arquivo CSV", type="csv", key="anomalias")

    if uploaded_file is not None:
        # Ler o arquivo CSV
        df_new = pd.read_csv(uploaded_file)

        # Remover colunas que não devem ser usadas no modelo
        df_filtered = df_new.drop(columns=['clientCode', 'clientIndex', 'clientCode_encoded', 'delta_time', 'consumo_horarizado'])

        st.write("Visualização dos primeiros 3 dados carregados:")
        st.dataframe(df_filtered.head(3))

        # Selecionar automaticamente todas as colunas numéricas para análise
        selected_columns = df_filtered.select_dtypes(include=["number"]).columns

        if len(selected_columns) > 0:
            # Escalando os dados selecionados
            df_selected = df_filtered[selected_columns]
            df_scaled = scale_data(df_selected)

            # Fazendo as predições
            df_new['anomaly'] = predict_anomalies(iso_forest_model, df_scaled)

            # Mostrando o resultado das predições
            num_anomalias = df_new[df_new['anomaly'] == 1].shape[0]
            num_normais = df_new[df_new['anomaly'] == 0].shape[0]

            st.markdown(f"<span style='color: red;'>**Número de anomalias detectadas:** {num_anomalias}</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='color: green;'>**Número de amostras normais:** {num_normais}</span>", unsafe_allow_html=True)

            # Tabela com Instalações com Dados Anômalos
            st.subheader("Instalações com Dados Anômalos")

            # Filtrar apenas as linhas com anomalias
            df_anomalies = df_new[df_new['anomaly'] == 1][['clientCode', 'clientCode_encoded', 'clientIndex', 'delta_time', 'consumo_horarizado', 'anomaly']]

            # Remover instalações duplicadas com anomalias
            df_anomalies_unique = df_anomalies.drop_duplicates()

            st.write("Tabela de instalações que possuem dados anômalos:")
            st.dataframe(df_anomalies_unique)

            # Visualização com gráficos
            st.subheader("Visualização de Anomalias")
            st.write("Gráfico de contagem de amostras normais vs anomalias:")

            # Plotando a contagem de anomalias
            st.bar_chart(df_new['anomaly'].value_counts())

            # Gráfico Scatterplot de Anomalias vs Delta Time
            st.subheader("Scatterplot de Anomalias vs Delta Time")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                x='delta_time', y='consumo_horarizado', hue='anomaly', data=df_new, palette={0: 'blue', 1: 'red'}, alpha=0.7, ax=ax
            )
            ax.set_title('Anomalias vs Delta Time')
            ax.set_xlabel('Delta Time (horas)')
            ax.set_ylabel('Variação de Consumo por Hora')
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=['Normal', 'Anomalia'], title='Classificação')
            ax.grid(True)
            st.pyplot(fig)

            
        else:
            st.warning("Não foram encontradas colunas numéricas no arquivo carregado.")
    else:
        st.warning("Por favor, carregue um arquivo CSV.")


# Aba 3: Predição de Consumo de um Cliente (Somente Visual)
with tab3:
    st.header("Predição de Consumo de Gás para um Cliente (Em Breve)")
    st.write("""
    Esta seção será usada no futuro para prever o consumo de gás de um cliente com base em características específicas.
    """)
    
    uploaded_file = st.file_uploader("Carregar arquivo CSV", type="csv", key="previsao")
