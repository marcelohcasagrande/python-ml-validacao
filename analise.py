# Pacotes.
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, KFold, StratifiedKFold, GroupKFold, GridSearchCV, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint
import time


    #                      #
    # Lendo banco de dados # 
    #                      #

# Lendo base de dados.
uri = 'https://gist.githubusercontent.com/guilhermesilveira/e99a526b2e7ccc6c3b70f53db43a87d2/raw/1605fc74aa778066bf2e6695e24d53cf65f2f447/machine-learning-carros-simulacao.csv'
dados = pd.read_csv( uri ).drop( columns = [ 'Unnamed: 0' ], axis = 1 )
dados.head()

# Separando features e resposta.
x = dados[ [ 'preco', 'idade_do_modelo', 'km_por_ano' ] ]
y = dados[ 'vendido' ]


    #                                                  #  
    # Rodando validação cruzada com k-fold = 3, 5 e 10 #
    #                                                  # 

# A importância da validação cruzada está em retirar possível viés na seleção do treino e do teste permitindo 
# que o resultado de performance encontrado seja o mais fiel possível ao que seria de fato obtido no mundo real.
# Observação: o default não embaralha os dados. O split é feito na ordem que os dados estão.

# Fixando semente.
SEED = 301
np.random.seed( SEED )

modelo = DecisionTreeClassifier( max_depth = 2 )
results = cross_validate( modelo, x, y, cv = 3, return_train_score = False )
media = results[ 'test_score' ].mean()
desvio_padrao = results[ 'test_score' ].std()
print( 'Acurácia com cross validation, 3 = [%.2f, %.2f]' % ( ( media - 2 * desvio_padrao ) * 100, ( media + 2 * desvio_padrao ) * 100 ) )

# Fixando semente.
SEED = 301
np.random.seed( SEED )

modelo = DecisionTreeClassifier( max_depth = 2 )
results = cross_validate( modelo, x, y, cv = 5, return_train_score = False )
media = results[ 'test_score' ].mean()
desvio_padrao = results[ 'test_score' ].std()
print( 'Acurácia com cross validation, 5 = [%.2f, %.2f]' % ( ( media - 2 * desvio_padrao ) * 100, ( media + 2 * desvio_padrao ) * 100 ) )

# Fixando semente.
SEED = 301
np.random.seed( SEED )

modelo = DecisionTreeClassifier( max_depth = 2 )
results = cross_validate( modelo, x, y, cv = 10, return_train_score = False )
media = results[ 'test_score' ].mean()
desvio_padrao = results[ 'test_score' ].std()
print( 'Acurácia com cross validation, 10 = [%.2f, %.2f]' % ( ( media - 2 * desvio_padrao ) * 100, ( media + 2 * desvio_padrao ) * 100 ) )


    #        #   
    # Função #
    #        #

# Imprime resultados.    
def imprime_resultados( results ):
    media = results[ 'test_score' ].mean()
    desvio_padrao = results[ 'test_score' ].std()
    print( 'Acurácia média: %.2f' % ( media * 100 ) )
    print( 'Acurácia (IC): [%.2f, %.2f]' % ( ( media - 2 * desvio_padrao ) * 100, ( media + 2 * desvio_padrao ) * 100 ) )
    

    #                           #
    # Usando KFold embaralhando #
    #                           #

# A importância de embaralhar está em retirar possível viés 
# na ordem em que os dados estão em seu banco de dados.

# Fixando semente.
SEED = 301
np.random.seed( SEED )

cv = KFold( n_splits = 10, shuffle = False ) # 10 splits, mas sem embaralhar.
modelo = DecisionTreeClassifier( max_depth = 2 )
results = cross_validate( modelo, x, y, cv = cv, return_train_score = False )
imprime_resultados( results )

# Fixando semente.
SEED = 301
np.random.seed( SEED )

cv = KFold( n_splits = 10, shuffle = True ) # 10 splits e embaralhando.
modelo = DecisionTreeClassifier( max_depth = 2 )
results = cross_validate( modelo, x, y, cv = cv, return_train_score = False )
imprime_resultados( results )


    #                        #
    # Usando StratifiedKFold #
    #                        #

# Esta abordagem mantém a mesma proporção das categorias 
# da variável resposta em cada fold, evitando desbalanceamentos.

# Fixando semente.
SEED = 301
np.random.seed( SEED )

cv = StratifiedKFold( n_splits = 10, shuffle = True ) # 10 splits e embaralhando.
modelo = DecisionTreeClassifier( max_depth = 2 )
results = cross_validate( modelo, x, y, cv = cv, return_train_score = False )
imprime_resultados( results )


    #                   #
    # Usando GroupKFold #
    #                   #

# Abordagem importante para o caso de algum grupo ter características muito
# diferentes dos demais, permitindo avaliar a performance mesmo neste cenário.

# Gerando uma variável de modelo do carro de forma aleatória.
np.random.seed( SEED )
dados[ 'modelo' ] = dados.idade_do_modelo + np.random.randint( -2, 3, size = 10000 )
dados.modelo = dados.modelo + abs( dados.modelo.min() ) + 1
dados.head()

# Valores únicos criados.
dados.modelo.unique()

# Frequência de cada modelo aleatório criado.
dados.modelo.value_counts()

# Fixando semente.
SEED = 301
np.random.seed( SEED )

cv = GroupKFold( n_splits = 10 ) # ele vai levar em consideração os grupos passados para criar os k-folds.
modelo = DecisionTreeClassifier( max_depth = 2 )
results = cross_validate( modelo, x, y, cv = cv, groups = dados.modelo, return_train_score = False ) # é passado o modelo do carro como grupo dentro do processo de validação cruzada.
imprime_resultados( results )


    #          #
    # Pipeline #
    #          # 

# Usar o pipeline permite a padronização dos dados de treino (para fit) em cada etapa de k-fold da validação cruzada.
# Além disso, permite criar um mecanismo completo do processo de ML.
# Observação: o modelo de SVC é ideal que os dados estejam padronizados. Vou usar o StandardScaler.

# Fixando semente.
SEED = 301
np.random.seed( SEED )

# Escalador e modelo.
scaler = StandardScaler()
modelo = SVC()

# Criando pipeline e atribuindo nomes ao escalador e ao modelo.
pipeline = Pipeline( [ ( 'transformacao', scaler ), ( 'estimador', modelo ) ] )

# Validação cruzada usando 10 splits e os modelos dos carros como grupos.
cv = GroupKFold( n_splits = 10 )
results = cross_validate( pipeline, x, y, cv = cv, groups = dados.modelo, return_train_score = False )
imprime_resultados( results )


    #                               #
    # Otimização de Hiperparâmetros #
    #                               #


# Criando graficamente a árvore com profundidade 2.
SEED = 301
np.random.seed( SEED )

cv = GroupKFold( n_splits = 10 )
modelo = DecisionTreeClassifier( max_depth = 2 )
results = cross_validate( modelo, x, y, cv = cv, groups = dados.modelo, return_train_score = False )
imprime_resultados( results )

modelo.fit( x, y )
features = x.columns
dot_data = export_graphviz( modelo, out_file = None, filled = True, rounded = True, class_names = [ 'não', 'sim' ], feature_names = features )
graph = graphviz.Source( dot_data )
graph

# Criando graficamente a árvore com profundidade 3.
SEED = 301
np.random.seed( SEED )

cv = GroupKFold( n_splits = 10 )
modelo = DecisionTreeClassifier( max_depth = 3 )
results = cross_validate( modelo, x, y, cv = cv, groups = dados.modelo, return_train_score = False )
imprime_resultados( results )

modelo.fit( x, y )
features = x.columns
dot_data = export_graphviz( modelo, out_file = None, filled = True, rounded = True, class_names = [ 'não', 'sim' ], feature_names = features )
graph = graphviz.Source( dot_data )
graph


    #                                             #  
    # Explorando hiper parâmetros em uma dimensão #
    #                                             # 

def roda_arvore_de_decisao( max_depth ):
    SEED = 301
    np.random.seed( SEED )
    cv = GroupKFold( n_splits = 10 )
    modelo = DecisionTreeClassifier( max_depth = max_depth )
    results = cross_validate( modelo, x, y, cv = cv, groups = dados.modelo, return_train_score = True )
    train_score = results[ 'train_score' ].mean() * 100
    test_score = results[ 'test_score' ].mean() * 100
    print( 'Árvore max_depth = %d, treino = %.2f, teste = %.2f' % ( max_depth, train_score, test_score ) )
    tabela = [ max_depth, train_score, test_score ]
    return tabela
  
resultados = [ roda_arvore_de_decisao( i ) for i in range( 1,33 ) ]
resultados = pd.DataFrame( resultados, columns = [ 'max_depth', 'train', 'test' ] )
resultados.head()

# Resultado para treino.
sns.lineplot( x = 'max_depth', y = 'train', data = resultados ) 
plt.show()

# Resultado para treino e teste.
sns.lineplot( x = 'max_depth', y = 'train', data = resultados )
sns.lineplot( x = 'max_depth', y = 'test', data = resultados )
plt.legend( [ 'Treino', 'Teste' ] )
plt.show()

# Ordenando performance por teste.
resultados.sort_values( 'test', ascending = False ).head()


    #                                            # 
    # Explorando hiper parâmetros em 2 dimensões # 
    #                                            #

def roda_arvore_de_decisao( max_depth, min_samples_leaf ):
    SEED = 301
    np.random.seed( SEED )
    cv = GroupKFold( n_splits = 10 )
    modelo = DecisionTreeClassifier( max_depth = max_depth, min_samples_leaf = min_samples_leaf )
    results = cross_validate( modelo, x, y, cv = cv, groups = dados.modelo, return_train_score = True )
    train_score = results[ 'train_score' ].mean() * 100
    test_score = results[ 'test_score' ].mean() * 100
    print( 'Árvore max_depth = %d, min_samples_leaf = %d, treino = %.2f, teste = %.2f' % ( max_depth, min_samples_leaf, train_score, test_score ) )
    tabela = [ max_depth, min_samples_leaf, train_score, test_score ]
    return tabela

def busca():
    resultados = []
    for max_depth in range( 1,33 ):
        for min_samples_leaf in [ 32, 64, 128, 256 ]:
            tabela = roda_arvore_de_decisao( max_depth, min_samples_leaf )
            resultados.append( tabela )
    resultados = pd.DataFrame( resultados, columns = [ 'max_depth', 'min_samples_leaf', 'train', 'test' ] )
    return resultados

# Resultados.
resultados = busca() # realizando busca.
resultados.head() # exibindo.

# Ordenando.
resultados.sort_values( 'test', ascending = False ).head() # ordenando pelo melhor para teste.

# Correlações.
corr = resultados.corr() # correlações.
corr

# Exibindo correlações via heatmap.
sns.heatmap( corr )
plt.show()


    #                                            #    
    # Explorando 3 dimensões de hiper parâmetros #
    #                                            #

def roda_arvore_de_decisao( max_depth, min_samples_leaf, min_samples_split ):
    SEED = 301
    np.random.seed( SEED )
    cv = GroupKFold( n_splits = 10 )
    modelo = DecisionTreeClassifier( max_depth = max_depth, min_samples_leaf = min_samples_leaf, min_samples_split = min_samples_split )
    results = cross_validate( modelo, x, y, cv = cv, groups = dados.modelo, return_train_score = True )
    fit_time = results[ 'fit_time' ].mean()
    score_time = results[ 'score_time' ].mean()
    train_score = results[ 'train_score' ].mean() * 100
    test_score = results[ 'test_score' ].mean() * 100
    tabela = [ max_depth, min_samples_leaf, min_samples_split, train_score, test_score, fit_time, score_time ]
    return tabela

def busca():
    resultados = []
    for max_depth in range( 1, 33 ):
        for min_samples_leaf in [ 32, 64, 128, 256 ]:
            for min_samples_split in [ 32, 64, 128, 256 ]:
                tabela = roda_arvore_de_decisao( max_depth, min_samples_leaf, min_samples_split )
                resultados.append( tabela )
    resultados = pd.DataFrame( resultados, columns = [ 'max_depth', 'min_samples_leaf', 'min_samples_split', 'train', 'test', 'fit_time', 'score_time' ] )
    return resultados

# Exibindo a busca.
resultados = busca()
resultados.head()

# Correlações.
corr = resultados.corr()


    #              # 
    # GridSearchCV #
    #              #
    
SEED = 301
np.random.seed( SEED )

espaco_de_parametros = {
    'max_depth': [ 3, 5 ],
    'min_samples_split': [ 32, 64, 128 ],
    'min_samples_leaf': [ 32, 64, 128 ],
    'criterion': [ 'gini', 'entropy' ]
}

# Montando o grid e a busca.
busca = GridSearchCV( DecisionTreeClassifier(),
                      espaco_de_parametros,
                      cv = GroupKFold( n_splits = 10 ) )

# Modelando.
busca.fit( x, y, groups = dados.modelo ) # fittando.
resultados = pd.DataFrame( busca.cv_results_ ) # resultados.
resultados.head()

print( busca.best_params_ ) # melhor combinação.
print( busca.best_score_ * 100 ) # melhor performance.

melhor = busca.best_estimator_ # armazenando melhor estimador.
melhor

# Evitar essa abordagem pois estará sendo otimista.

# Predições.
predicoes = melhor.predict( x )
accuracy = accuracy_score( predicoes, y ) * 100
print( 'Acurácia para os dados foi %.2f%%' % accuracy )
   
# Como ter uma estimativa sem esse vício nos dados que eu já vi?
# No caso de cross validation com busca de hiper parâmetros, fazemos uma nova validação cruzada. Chama-se nested cross validation.

# Scores via validação cruzada.
# scores = cross_val_score( busca, x, y, cv = GroupKFold( n_splits = 10 ), groups = dados.modelo ) 

# Infelizmente como o Pandas não suporta nested validation com group K Fold não conseguimos prever o resultado para novos grupos.

SEED = 301
np.random.seed( SEED )

espaco_de_parametros = {
    'max_depth': [ 3, 5 ],
    'min_samples_split': [ 32, 64, 128 ],
    'min_samples_leaf': [ 32, 64, 128 ],
    'criterion': [ 'gini', 'entropy' ]
}

busca = GridSearchCV( DecisionTreeClassifier(),
                      espaco_de_parametros,
                      cv = KFold( n_splits = 5, shuffle = True ) )

busca.fit( x, y )
resultados = pd.DataFrame( busca.cv_results_ )
resultados.head()

scores = cross_val_score( busca, x, y, cv = KFold( n_splits = 5, shuffle = True ) )
scores

def imprime_score( scores ):
  media = scores.mean() * 100
  desvio = scores.std() * 100
  print( 'Acurácia média %.2f' % media )
  print( 'Intervalo [%.2f, %.2f]' % ( media - 2 * desvio, media + 2 * desvio ) )

imprime_score( scores )

melhor = busca.best_estimator_
print( melhor )

features = x.columns
dot_data = export_graphviz( melhor, out_file = None, filled = True, rounded = True, class_names = [ 'não', 'sim' ], feature_names = features )
graph = graphviz.Source( dot_data )
graph


    #                               # 
    # Busca aleatória: RandomSearch #
    #                               # 

# Mais rápido e testa n_iter possibilidades aleatórias.
SEED = 301
np.random.seed( SEED )

espaco_de_parametros = {
    'max_depth': [ 3, 5 ],
    'min_samples_split': [ 32, 64, 128 ],
    'min_samples_leaf': [ 32, 64, 128 ],
    'criterion': [ 'gini', 'entropy' ]
}

busca = RandomizedSearchCV( DecisionTreeClassifier(),
                            espaco_de_parametros,
                            n_iter = 16,
                            cv = KFold( n_splits = 5, shuffle = True ),
                            random_state = SEED )

busca.fit( x, y )
resultados = pd.DataFrame( busca.cv_results_ )
resultados.head()

scores = cross_val_score( busca, x, y, cv = KFold( n_splits = 5, shuffle = True ) )
imprime_score( scores )

melhor = busca.best_estimator_
print( melhor )

features = x.columns
dot_data = export_graphviz( melhor, out_file = None, filled = True, rounded = True, class_names = [ 'não', 'sim' ], feature_names = features )
graph = graphviz.Source( dot_data )
graph


    #                                           # 
    # Customizando o espaço de hiper parâmetros #
    #                                           #

SEED = 301
np.random.seed( SEED )

espaco_de_parametros = {
    'max_depth': [ 3, 5, 10, 15, 20, 30, None ],
    'min_samples_split': randint( 32, 128 ),
    'min_samples_leaf': randint( 32, 128 ),
    'criterion': [ 'gini', 'entropy' ]
}

busca = RandomizedSearchCV( DecisionTreeClassifier(),
                            espaco_de_parametros,
                            n_iter = 16,
                            cv = KFold( n_splits = 5, shuffle = True ),
                            random_state = SEED )

busca.fit( x, y )
resultados = pd.DataFrame( busca.cv_results_ )
resultados.head()

scores = cross_val_score( busca, x, y, cv = KFold( n_splits = 5, shuffle = True ) )
imprime_score( scores )
melhor = busca.best_estimator_
print( melhor )

resultados_ordenados_pela_media = resultados.sort_values( 'mean_test_score', ascending = False )
for indice, linha in resultados_ordenados_pela_media.iterrows():
    print( '%.3f +-(%.3f) %s' % ( linha.mean_test_score, linha.std_test_score * 2, linha.params ) )


    #                                                #
    # Uma exploração mais a fundo de forma aleatória #
    #                                                #

SEED = 564
np.random.seed( SEED )

espaco_de_parametros = {
    'max_depth': [ 3, 5, 10, 15, 20, 30, None ],
    'min_samples_split': randint( 32, 128 ),
    'min_samples_leaf': randint( 32, 128 ),
    'criterion': [ 'gini', 'entropy' ]
}

busca = RandomizedSearchCV( DecisionTreeClassifier(),
                            espaco_de_parametros,
                            n_iter = 64,
                            cv = KFold( n_splits = 5, shuffle = True ),
                            random_state = SEED )

busca.fit( x, y )
resultados = pd.DataFrame( busca.cv_results_ )
resultados.head()

resultados_ordenados_pela_media = resultados.sort_values( 'mean_test_score', ascending = False )
for indice, linha in resultados_ordenados_pela_media.iterrows():
    print( '%.3f +-(%.3f) %s' % ( linha.mean_test_score, linha.std_test_score * 2, linha.params ) )

scores = cross_val_score( busca, x, y, cv = KFold( n_splits = 5, shuffle = True ) ) # adicionando validação cruzada.
imprime_score( scores )
melhor = busca.best_estimator_
print( melhor )


    #                                                             #
    # Comparando GridSearchCV com RandomizedSearch (1 comparação) #
    #                                                             #

SEED = 301
np.random.seed( SEED ) 

espaco_de_parametros = {
    'n_estimators': [ 10, 100 ],
    'max_depth': [ 3, 5 ],
    'min_samples_split': [ 32, 64, 128 ],
    'min_samples_leaf': [ 32, 64, 128 ],
    'bootstrap': [ True, False ],
    'criterion': [ 'gini', 'entropy' ]
}

tic = time.time()
busca = GridSearchCV( RandomForestClassifier(),
                      espaco_de_parametros,
                      cv = KFold( n_splits = 5, shuffle = True ) )

busca.fit( x, y )
tac = time.time()
tempo_que_passou = tac - tic
print( 'Tempo %.2f segundos' % tempo_que_passou ) # tempo do GridSearch para um RandomForestClassifier.

resultados = pd.DataFrame( busca.cv_results_ )
resultados.head()

resultados_ordenados_pela_media = resultados.sort_values( 'mean_test_score', ascending = False ) 
for indice, linha in resultados_ordenados_pela_media[:5].iterrows():
    print( '%.3f +-(%.3f) %s' % ( linha.mean_test_score, linha.std_test_score * 2, linha.params ) )

tic = time.time()
scores = cross_val_score( busca, x, y, cv = KFold( n_splits = 5, shuffle = True ) ) # tempo da validação cruzada.
tac = time.time()
tempo_passado = tac - tic
print( 'Tempo %.2f segundos' % tempo_passado )

imprime_score( scores )
melhor = busca.best_estimator_
print( melhor )

SEED = 301
np.random.seed( SEED )

espaco_de_parametros = {
    'n_estimators': [ 10, 100 ],
    'max_depth': [ 3, 5 ],
    'min_samples_split': [ 32, 64, 128 ],
    'min_samples_leaf': [ 32, 64, 128 ],
    'bootstrap': [ True, False ],
    'criterion': [ 'gini', 'entropy' ]
}

tic = time.time()
busca = RandomizedSearchCV( RandomForestClassifier(),
                            espaco_de_parametros,
                            n_iter = 20,
                            cv = KFold( n_splits = 5, shuffle = True ) )

busca.fit( x, y )
tac = time.time()
tempo_que_passou = tac - tic
print( 'Tempo %.2f segundos' % tempo_que_passou ) # tempo do RandomizedSearch para um RandomForestClassifier.

resultados = pd.DataFrame( busca.cv_results_ )
resultados.head()

resultados_ordenados_pela_media = resultados.sort_values( 'mean_test_score', ascending = False )
for indice, linha in resultados_ordenados_pela_media[:5].iterrows():
    print( '%.3f +-(%.3f) %s' % ( linha.mean_test_score, linha.std_test_score * 2, linha.params ) )

tic = time.time()
scores = cross_val_score( busca, x, y, cv = KFold( n_splits = 5, shuffle = True ) ) # tempo da validação cruzada.
tac = time.time()
tempo_passado = tac - tic
print( 'Tempo %.2f segundos' % tempo_passado )

imprime_score( scores )
melhor = busca.best_estimator_
print( melhor )

SEED = 301
np.random.seed( SEED )

espaco_de_parametros = {
    'n_estimators': randint( 10, 101 ),
    'max_depth': randint( 3, 6 ),
    'min_samples_split': randint( 32, 129 ),
    'min_samples_leaf': randint( 32, 129 ),
    'bootstrap': [ True, False ],
    'criterion': [ 'gini', 'entropy' ]
}

tic = time.time()
busca = RandomizedSearchCV( RandomForestClassifier(),
                            espaco_de_parametros,
                            n_iter = 80,
                            cv = KFold( n_splits = 5, shuffle = True ) )

busca.fit( x, y )
tac = time.time()
tempo_que_passou = tac - tic
print( 'Tempo %.2f segundos' % tempo_que_passou )

resultados = pd.DataFrame( busca.cv_results_ )
resultados.head()

resultados_ordenados_pela_media = resultados.sort_values( 'mean_test_score', ascending = False )
for indice, linha in resultados_ordenados_pela_media[:5].iterrows():
    print( '%.3f +-(%.3f) %s' % ( linha.mean_test_score, linha.std_test_score * 2, linha.params ) )


    #                                                      #
    # Se eu não posso ou não consigo usar cross validation #
    #                                                      #

# 0.6 treino     => treino
# 0.2 teste      => dev teste
# 0.2 validacao  => validacao

SEED = 301
np.random.seed( SEED )

x_treino_teste, x_validacao, y_treino_teste, y_validacao = train_test_split( x, y, test_size = 0.2, shuffle = True, stratify = y )
print( x_treino_teste.shape )
print( x_validacao.shape )
print( y_treino_teste.shape )
print( y_validacao.shape )

espaco_de_parametros = {
    'n_estimators': randint( 10, 101 ),
    'max_depth': randint( 3, 6 ),
    'min_samples_split': randint( 32, 129 ),
    'min_samples_leaf': randint( 32, 129 ),
    'bootstrap': [ True, False ],
    'criterion': [ 'gini', 'entropy' ]
}

split = StratifiedShuffleSplit( n_splits = 1, test_size = 0.25 )

tic = time.time()
busca = RandomizedSearchCV( RandomForestClassifier(),
                            espaco_de_parametros,
                            n_iter = 5,
                            cv = split )

busca.fit( x_treino_teste, y_treino_teste )
tac = time.time()
tempo_que_passou = tac - tic
print( 'Tempo %.2f segundos' % tempo_que_passou )

resultados = pd.DataFrame( busca.cv_results_ )
resultados.head()

tic = time.time()
scores = cross_val_score( busca, x_validacao, y_validacao, cv = split )
tac = time.time()
tempo_passado = tac - tic
print( 'Tempo %.2f segundos' % tempo_passado )
scores

