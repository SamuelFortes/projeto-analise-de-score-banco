# Projeto Python IA: Inteligência Artificial e Previsões
# Case: Score de Crédito dos Clientes

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Passo 1: Importar base de dados
def carregar_dados(caminho_arquivo):
    """Carrega os dados do arquivo CSV"""
    return pd.read_csv(caminho_arquivo)

# Passo 2: Preparar base de dados para a IA
def preparar_dados(tabela):
    """Prepara os dados convertendo variáveis categóricas em numéricas"""
    # Lista de colunas categóricas para codificar
    colunas_categoricas = ["profissao", "mix_credito", "comportamento_pagamento"]
    
    # Aplicar LabelEncoder para cada coluna categórica
    codificadores = {}
    for coluna in colunas_categoricas:
        codificador = LabelEncoder()
        tabela[coluna] = codificador.fit_transform(tabela[coluna])
        codificadores[coluna] = codificador
    
    return tabela, codificadores

# Passo 3: Criar e treinar modelos
def treinar_modelos(x_treino, y_treino):
    """Treina e retorna dois modelos de classificação"""
    # Modelo 1: Random Forest
    modelo_arvore = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo_arvore.fit(x_treino, y_treino)
    
    # Modelo 2: K-Nearest Neighbors
    modelo_knn = KNeighborsClassifier()
    modelo_knn.fit(x_treino, y_treino)
    
    return modelo_arvore, modelo_knn

# Passo 4: Avaliar modelos
def avaliar_modelos(modelo_arvore, modelo_knn, x_teste, y_teste):
    """Avalia os modelos e retorna suas acurácias"""
    previsao_arvore = modelo_arvore.predict(x_teste)
    previsao_knn = modelo_knn.predict(x_teste)
    
    acuracia_arvore = accuracy_score(y_teste, previsao_arvore)
    acuracia_knn = accuracy_score(y_teste, previsao_knn)
    
    return acuracia_arvore, acuracia_knn

# Passo 5: Fazer previsões em novos dados
def prever_novos_dados(modelo, tabela_nova, codificadores):
    """Prepara e faz previsões em novos dados"""
    # Aplicar a mesma codificação usada nos dados de treino
    for coluna, codificador in codificadores.items():
        # Para lidar com categorias não vistas durante o treino
        tabela_nova[coluna] = tabela_nova[coluna].apply(
            lambda x: x if x in codificador.classes_ else codificador.classes_[0]
        )
        tabela_nova[coluna] = codificador.transform(tabela_nova[coluna])
    
    # Fazer previsões
    previsoes = modelo.predict(tabela_nova)
    tabela_nova['score_credito'] = previsoes
    
    return tabela_nova

# Função principal
def main():
    print("Iniciando análise de score de crédito...")
    
    # 1. Carregar dados
    print("Carregando dados de clientes...")
    tabela = carregar_dados("clientes.csv")
    
    # Exibir informações básicas
    print(f"Formato dos dados: {tabela.shape}")
    print("\nPrimeiras 5 linhas:")
    print(tabela.head())
    print("\nInformações do dataset:")
    print(tabela.info())
    
    # 2. Preparar dados
    print("\nPreparando dados para análise...")
    tabela_preparada, codificadores = preparar_dados(tabela.copy())
    
    # 3. Separar features e target
    y = tabela_preparada["score_credito"]
    x = tabela_preparada.drop(columns=["score_credito", "id_cliente"])
    
    # 4. Dividir em conjuntos de treino e teste
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    
    print(f"Tamanho do conjunto de treino: {x_treino.shape[0]} amostras")
    print(f"Tamanho do conjunto de teste: {x_teste.shape[0]} amostras")
    
    # 5. Treinar modelos
    print("\nTreinando modelos...")
    modelo_arvore, modelo_knn = treinar_modelos(x_treino, y_treino)
    
    # 6. Avaliar modelos
    acuracia_arvore, acuracia_knn = avaliar_modelos(
        modelo_arvore, modelo_knn, x_teste, y_teste
    )
    
    print(f"\nAcurácia do modelo Random Forest: {acuracia_arvore:.4f}")
    print(f"Acurácia do modelo KNN: {acuracia_knn:.4f}")
    
    # 7. Escolher o melhor modelo
    if acuracia_arvore >= acuracia_knn:
        melhor_modelo = modelo_arvore
        print("\nMelhor modelo: Random Forest")
    else:
        melhor_modelo = modelo_knn
        print("\nMelhor modelo: KNN")
    
    # 8. Fazer previsões em novos dados
    print("\nCarregando e processando novos clientes...")
    try:
        tabela_nova = carregar_dados("novos_clientes.csv")
        resultado = prever_novos_dados(melhor_modelo, tabela_nova, codificadores)
        
        print("\nPrevisões para novos clientes:")
        print(resultado[['mes', 'idade', 'profissao', 'salario_anual', 'score_credito']])
        
        # Salvar resultados
        resultado.to_csv("resultado_previsoes.csv", index=False)
        print("\nResultados salvos em 'resultado_previsoes.csv'")
        
    except FileNotFoundError:
        print("Arquivo 'novos_clientes.csv' não encontrado. Pulando etapa de previsão.")
    
    print("\nAnálise concluída!")

if __name__ == "__main__":
    main()