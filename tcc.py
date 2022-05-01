#---------------------------------#
# Importação das Bibliotecas

#importando streamlit para execução do código Python na web
import streamlit as st

# importando o numpy para manipulação dos dados 
import numpy as np

# importando o pandas para manipulação do dataset
import pandas as pd

# importando todas as funções específicas de seleção de atributos do scikit-learn
from sklearn.feature_selection import * 

# imputando valores de média, mediana e moda para valores faltantes
from sklearn.impute import SimpleImputer 

# utilizado para o split entre treinamento e teste
from sklearn.model_selection import train_test_split

# importando RandomForest para regressão
from sklearn.ensemble import RandomForestRegressor

# importando KNN para regressão
from sklearn.neighbors import KNeighborsRegressor 

# importando LinearRegression para regressão
from sklearn.linear_model import LinearRegression 

# importando # SVM para regressão
from sklearn.svm import SVR 

# utilizado para que todas as entradas estejam na mesma escala numérica
# utilizado para preparar as colunas do tipo texto
from sklearn.preprocessing import RobustScaler, LabelEncoder 

# utilizando metricas para verificar acurácia
from sklearn.metrics import mean_squared_error, r2_score

# importando datasets padrão como exemplo
from sklearn.datasets import load_diabetes, load_boston

#---------------------------------#
# Layout da página
## Página se ajusta automaticamente a largura da página
st.set_page_config(page_title='TCC Final: Machine Learning App',
    layout='wide')

#---------------------------------#
# Construção do Modelo
def construir_modelo(df):
    X = df.iloc[:,:-1] # Usa toda as colunas exceto a última coluna como X
    Y = df.iloc[:,-1] # Seleciona a última coluna como Y


    # Label Encoder
    if exemplo is False:
        le = LabelEncoder()
        le.fit(X['Country'])
        X['Country'] = pd.DataFrame(le.transform(X['Country']))
        le.fit(X['Status'])
        X['Status'] = pd.DataFrame(le.transform(X['Status']))

    st.markdown("**1.3. Tratando as colunas de texto com o método LabelEncoder**")
    st.write(X)

    # RobustScaler
    st.markdown("**1.4. Normalizando os dados com RobustScaler:**")
    df_robust = RobustScaler().fit_transform(X)
    normal_df = pd.DataFrame(df_robust)
    st.write(normal_df)

    
    # Variance Thereshold
    st.markdown('**1.5. Variance Thereshold para seleção de atributos:**')
    sel = VarianceThreshold(param_threshold)
    st.write('Aplicando a técnica Variance Thereshold:')
    df_threshold = pd.DataFrame(sel.fit_transform(normal_df), columns=X.columns[sel.get_support()])
    st.info(list(df_threshold.columns))
    st.info( str(df_threshold.shape[1]) + " atributos selecionados")

    # Voltando dataset original para tratamento de valores ausentes
    X = df_threshold
    X['Life expectancy'] = Y
    st.write(X)

    # Mostrando dados ausentes NaN
    st.markdown('Dados ausentes')
    st.write((X.isna().sum()))        

    # Simple Imputer para valores vizinhos
    st.markdown('**1.6. Simple Imputer para valores ausentes:**')
    imputer = SimpleImputer(strategy=param_imputer)
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    st.markdown('Dados ausentes após Simple Imputer')
    st.write((X.isna().sum()))

        
    # Preparando dataset para treino e teste
    Y = X["Life expectancy"]
    X = X.drop(['Life expectancy'], axis=1)
        
    # Dividindo os dados para treino e teste
    X_train, X_test, Y_train, Y_test = train_test_split(X, # aqui são informados os atributos de X sem o Y
                                                        Y, # aqui são informados os valores atributos Y
                                                        test_size=(100-split_size)/100, # porcentagem de divisão da base. 
                                                        # Geralmente é algo entre 20% (0.20) a 35% (0.35)
                                                        random_state=0)  # valor aleatório e usado para que alguns 
                                                        # algoritmos iniciem de forma aleatória a sua divisão.

    st.subheader('2. Dividindo os dados')
    st.write('Treinamento dos Dados')
    st.info(X_train.shape)
    st.write('Teste dos Dados')
    st.info(X_test.shape)

    st.write('Variável X')
    st.info(list(X.columns))
    st.write('Variável Y')
    st.info(Y.name)

    #Mostrando os modelos
    st.subheader('3. Performance dos Modelos')

    # KNN para regressão
    st.markdown('**3.1. Modelo KNN para regressão:**')
    modelo_knn = KNeighborsRegressor().fit(X_train, Y_train)
    st.markdown('Score:')
    st.info(modelo_knn.score(X_test, Y_test))
    st.markdown('Score:')
    

    # SVM para regressão
    st.markdown('**3.2. Modelo SVM para regressão:**')
    modelo_svm = SVR().fit(X_train, Y_train)
    st.markdown('Score:')
    st.info(modelo_svm.score(X_test, Y_test))
    
    # Regressão linear
    st.markdown('**3.3. Modelo LinearRegression para regressão:**')
    modelo_lr = LinearRegression().fit(X_train, Y_train)
    st.markdown('Score:')
    st.info(modelo_lr.score(X_test, Y_test))

    # RandomForestRegressor
    st.markdown('**3.4. Modelo RandomForest para regressão:**')
    modelo_random = RandomForestRegressor().fit(X_train, Y_train)
    st.markdown('Score:')
    st.info(modelo_random.score(X_test, Y_test))


    # Mostrando as previsões para todas as técnicas
    st.markdown('**3.5. Mostrando as previsões para todas as técnicas:**')
    aux = list(X_test.columns)
    prev_mod = X_test.set_index(aux[0])
    # st.write(aux)
    #prev_mod = pd.DataFrame(X_test) 
    prev_mod['Life Expectancy_Real'] = Y_test.values
    prev_mod['Life Expectancy_KNN'] = modelo_knn.predict(X_test)
    prev_mod['Life Expectancy_SVM'] = modelo_svm.predict(X_test)
    prev_mod['Life Expectancy_Linear'] = modelo_lr.predict(X_test)
    prev_mod['Life Expectancy_Random'] = modelo_random.predict(X_test)
    st.write(prev_mod)


    df = df.dropna()
    X = df.iloc[:,:-1] # Usa toda as colunas exceto a última coluna como X
    X = X.drop(['Country', 'Status', 'Population', 'Income_resource' ], axis=1)
    #X = X.set_index('Year')
    Y = df.iloc[:,-1] # Seleciona a última coluna como Y
    
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-split_size)/100, random_state=0) 


    def entrada_variaveis():
        #Country = st.sidebar.slider('Country', X.Country.min(), X.Country.max(), X.Country.mean())
        Year = st.sidebar.slider('Year', int(X.Year.min()), int(X.Year.max()), int(X.Year.mean()))
        Adult_Mortality = st.sidebar.slider('Adult_Mortality', int(X.Adult_Mortality.min()), int(X.Adult_Mortality.max()), int(X.Adult_Mortality.mean()))
        infant_deaths = st.sidebar.slider('infant_deaths', int(X.infant_deaths.min()), int(X.infant_deaths.max()), int(X.infant_deaths.mean()))
        alcool = st.sidebar.slider('alcool', int(X.alcool.min()), int(X.alcool.max()), int(X.alcool.mean()))
        percentage_expenditure = st.sidebar.slider('percentage_expenditure', int(X.percentage_expenditure.min()), int(X.percentage_expenditure.max()), int(X.percentage_expenditure.mean()))
        Hepatitis_B = st.sidebar.slider('Hepatitis_B', int(X.Hepatitis_B.min()), int(X.Hepatitis_B.max()), int(X.Hepatitis_B.mean()))
        Measles = st.sidebar.slider('Measles', int(X.Measles.min()), int(X.Measles.max()), int(X.Measles.mean()))
        BMI = st.sidebar.slider('BMI', int(X.BMI.min()), int(X.BMI.max()), int(X.BMI.mean()))
        under_five_deaths = st.sidebar.slider('under_five_deaths', int(X.under_five_deaths.min()), int(X.under_five_deaths.max()), int(X.under_five_deaths.mean()))
        Polio = st.sidebar.slider('Polio', int(X.Polio.min()), int(X.Polio.max()), int(X.Polio.mean()))
        Total_expenditure = st.sidebar.slider('Total_expenditure', int(X.Total_expenditure.min()), int(X.Total_expenditure.max()), int(X.Total_expenditure.mean()))
        Diphtheria = st.sidebar.slider('Diphtheria', int(X.Diphtheria.min()), int(X.Diphtheria.max()), int(X.Diphtheria.mean()))
        Aids = st.sidebar.slider('Aids', int(X.Aids.min()), int(X.Aids.max()), int(X.Aids.mean()))
        Gdp = st.sidebar.slider('Gdp', int(X.Gdp.min()), int(X.Gdp.max()), int(X.Gdp.mean()))
        #Population = st.sidebar.slider('Population', float(X.Population.min()), float(X.Population.max()), float(X.Population.mean()))
        thinnessTwenty = st.sidebar.slider('thinnessTwenty', int(X.thinnessTwenty.min()), int(X.thinnessTwenty.max()), int(X.thinnessTwenty.mean()))
        thinnessNine = st.sidebar.slider('thinnessNine', int(X.thinnessNine.min()), int(X.thinnessNine.max()), int(X.thinnessNine.mean()))
        #Income_resource = st.sidebar.slider('Income_resource', int(X.Income_resource.min()), int(X.Income_resource.max()), int(X.Income_resource.mean()))
        Schooling = st.sidebar.slider('Schooling', int(X.Schooling.min()), int(X.Schooling.max()), int(X.Schooling.mean()))

        data = {#'Country': Country,
                'Year': Year,
                'Adult_Mortality': Adult_Mortality,
                'infant_deaths': infant_deaths,
                'alcool': alcool,
                'percentage_expenditure': percentage_expenditure,
                'Hepatitis_B': Hepatitis_B,
                'Measles': Measles,
                'BMI' : BMI,
                'under_five_deaths': under_five_deaths,
                'Polio': Polio,
                'Total_expenditure': Total_expenditure,
                'Diphtheria': Diphtheria,
                'Aids' : Aids,
                'Gdp' : Gdp,
                #'Population' : Population,
                'thinnessTwenty' : thinnessTwenty,
                'thinnessNine' : thinnessNine,
                #'Income_resource' : Income_resource,
                'Schooling' : Schooling
                }
        modelo = pd.DataFrame(data, index=[0])
        return modelo

    simul = entrada_variaveis()

    st.subheader('4. Previsão de Y')
    st.markdown('**4.1. Entre com os valores das variáveis nos sidebars esquerdo:**')
    st.write(simul)


    # Simulando com Modelo de Regressão RandomForest
    model = RandomForestRegressor()
    model.fit(X, Y)

    # Aplicando modelo para previsão
    predicao = model.predict(simul)

    st.markdown('**4.2. Predição de Life Expectancy**')
    st.info(str(int(predicao)) + ' anos.')
    st.write('---')
    
    

#---------------------------------#
# Breve descrição do APP
st.write("""
# TCC Final: WebApp Machine Learning
TCC do curso de Engenharia de Software: Este aplicativo utiliza técnicas de Machine Learning para predição de valores.
""")

#---------------------------------#
# Barra Lateral - Coleta configurações de entrada do usuário no dataframe
with st.sidebar.header('1. Carregue seu arquivo .CSV'):
    carregar_arquivo = st.sidebar.file_uploader("Carregue um arquivo do tipo CSV", type=["csv"])
    st.sidebar.markdown("""
[Exemplo de como criar um Arquivo CSV](https://rockcontent.com/br/blog/csv/)
""")


# Barra Lateral - Especificação e Configuração dos Parâmetros
with st.sidebar.header('2. Configuração dos Parâmetros'):
    split_size = st.sidebar.slider('Divisão dos Dados (% para Treinamento dos Dados)', 10, 90, 70, 5)


# Barra Lateral - Configuração do Variance Thereshold (seleção de atributos) 
with st.sidebar.subheader('2.1 Thereshold (seleção de atributos:'):
    param_threshold = st.sidebar.slider('Aplicando threshold', 0.0, 0.4, 0.3, 0.1)


# Barra Lateral - Configuração do Imputer (valores vizinhos) 
with st.sidebar.subheader('2.2 Imputer (média, mediana ou moda):'):
    param_imputer = st.sidebar.select_slider('Aplicando imputer para valores NaN', options=['mean', 'median', 'most_frequent'])

st.sidebar.subheader('2.3 Valores de Simulação:')


#---------------------------------#
# Painel Geral

# Mostrando o dataset
st.subheader('1. Dataset')

if carregar_arquivo is not None:
    df = pd.read_csv(carregar_arquivo)
    
    st.markdown('**1.1. Visão Geral do dataset (Cinco Primeiras linhas):**')
    st.write(df.head(5))
    st.markdown('**1.2. Medidas de tendência central e dispersão:**')
    st.write(df.describe())

    exemplo = False
    construir_modelo(df)

else:
    st.info('Esperando carregamento de Arquivo CSV')
    if st.button('Pressione para usar um dataset de exemplo'):
        # Diabetes dataset
        #diabetes = load_diabetes()
        #X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        #Y = pd.Series(diabetes.target, name='response')
        #df = pd.concat( [X,Y], axis=1 )

        #st.markdown('The Diabetes dataset is used as the example.')
        #st.write(df.head(5))

        # Boston housing dataset
        boston = load_boston()
        X = pd.DataFrame(boston.data, columns=boston.feature_names)
        Y = pd.Series(boston.target, name='Resposta')
        df = pd.concat( [X,Y], axis=1 )

        st.markdown('O dataset The Boston housing está sendo usado. Exibindo as cinco primeiras linhas:')
        st.write(df.head(5))
        st.markdown('Medidas de tendência central e dispersão:')
        st.write(df.describe())
        exemplo = True



        #df_robust = RobustScaler().fit_transform(df)
        #normal_df = pd.DataFrame(df_robust)
        
        colunas = list(df.columns)
        construir_modelo(df)