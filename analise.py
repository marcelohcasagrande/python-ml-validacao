# Pacotes.
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


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
def imprime_resultados( results) :
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



