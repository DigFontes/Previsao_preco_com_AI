#!/usr/bin/env python
# coding: utf-8

# # Projeto Ciência de Dados - Previsão de Preços
# 
# - Nosso desafio é conseguir prever o preço de barcos que vamos vender baseado nas características do barco, como: ano, tamanho, tipo de barco, se é novo ou usado, qual material usado, etc.
# 
# - Base de Dados: https://drive.google.com/drive/folders/1o2lpxoi9heyQV1hIlsHXWSfDkBPtze-V?usp=share_link

# ### Passo a Passo de um Projeto de Ciência de Dados
# 
# - Passo 1: Entendimento do Desafio
# - Passo 2: Entendimento da Área/Empresa
# - Passo 3: Extração/Obtenção de Dados
# - Passo 4: Ajuste de Dados (Tratamento/Limpeza)
# - Passo 5: Análise Exploratória
# - Passo 6: Modelagem + Algoritmos (Aqui que entra a Inteligência Artificial, se necessário)
# - Passo 7: Interpretação de Resultados

# ![title](tabelas.png)

# In[36]:


# importação da biblioteca para manipulação dos dados
import pandas as pd


# In[37]:


# Criando variável para receber o arquivo com base de dados
tabela = pd.read_csv(
    r'C:\Users\Virtual Office\Python\Módulo 50 - Python aplicação no mercado de trabalho\Aula 4\Aula 4\barcos_ref.csv'
)
# Visualizando a base de dados
display(tabela)


# In[38]:


# Com esse comando consigo verificar o tipo de dados que estou manipulando, isso me auxiliar na forma como  vou trabalhar com
#cada tipo de dado. Cada tipo há uma maneira de manipular e tratar. 
print(tabela.info())


# In[39]:


# Com esse comando consigo verificar correção que informações tem entre si, dessa forma consigo identificar dados que estão
#correlacionados e não correlacionados.
correlacao = tabela.corr()[['Preco']]
display(tabela.corr()[['Preco']])


# In[40]:


# importando as bibliotecas de criação de gráficos.
import seaborn  as sns 
import matplotlib.pyplot as plt


# In[41]:


# Com o gráfico mapa de calor consigo de forma visual, notar os fatores que estão mais e menos correlacionados
# criação do gráfico 
sns.heatmap(correlacao, cmap = 'Reds', annot = True)
#exibição do gráfico
plt.show()


# In[42]:


# Os dados que serão previstos
y = tabela['Preco']
# As características do barco a serem análisadas para construção da previsão de preço
x = tabela.drop('Preco', axis = 1)

# Train Test Split 
from sklearn.model_selection import train_test_split
# x_treino e x_teste são os dados das característica para a IA treinar e testar
# y_treino e y_teste são os dados de preço para a IA testar e treinar
# A partir dos treinos e testes na base de características, a IA compreenderá o que influcia o preço e então fazer as previsão
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size= 0.3)


# In[43]:


# Importação da inteligência artificial
    # - RegressaoLinear  e ArvoredeDecisão
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
# Criação da inteligência artificial
modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()
# Treinamento da inteligência artificial
modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)


# In[44]:


# Escolha do melhor modelo de previsão através do R2( 0 -> 100%)
from sklearn.metrics import r2_score

previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

print(r2_score(y_teste, previsao_regressaolinear))
print(r2_score(y_teste, previsao_arvoredecisao))


# In[45]:


# Com o gráfico de linhas consigo visualizar a precisão de cada modelo de previsão.
tabela_auxiliar = pd.DataFrame()
tabela_auxiliar['y_teste'] = y_teste
tabela_auxiliar['ArvoreDecisao'] = previsao_arvoredecisao
tabela_auxiliar['RegressaoLinear'] = previsao_regressaolinear
plt.figure(figsize=(10,5))
sns.lineplot(data = tabela_auxiliar)
plt.show()


# In[48]:


tabela_nova = pd.read_csv(
    r'C:\Users\Virtual Office\Python\Módulo 50 - Python aplicação no mercado de trabalho\Aula 4\Aula 4\novos_barcos.csv'
)

display(tabela_nova)

previsao = modelo_arvoredecisao.predict(tabela_nova)

print(previsao)

