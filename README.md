# titanic-ml

## Titanic: Machine Learning from Disaster: minha solução

Minha solução para a competição "Titanic: Machine Learning from Disaster" no [Kaggle](https://www.kaggle.com/`c/titanic). Neste desafio, é disponibilizado aos competidores um formulário com dados (como nome, idade, preço do ticket, etc) de 891 passageiros (*dataset* de treino) que estavam a bordo do Titanic e seu destino no acidente (falaceu/sobreviveu). O objetivo é criar modelos que  corretamente prevejam o destino dos 418 passageiros restantes (*dataset* de teste).

### Código

Esta foi minha primeira competição no Kaggle, então o código não está 100% claro e otimizado. Podem ser usados os seguintes scripts (pasta *code*):

* `base.py`: imports e algumas funções gerais
* `hyperopt_search_spaces.py`: definição de espaços de busca para otimização de hiperparâmetros
* `titanic-explore.py`: gráficos e dados sobre o conjunto de dados
* `titanic-feat-eng.py`: criação de novas variáveis (*feature engineering*)
* `titanic-train.py`: ajuste de algoritmos, otimização de hiperparâmetros
* `titanic.r`: ajuste do algoritmo *conditional inference trees*, melhor solução que obtive
   
### Descrição

Este desafio é como um "Hello World" no mundo dos Kagglers, onde muita gente começa com a ciência de dados. O dataset é pequeno e fácil de lidar.

Meu foco foi na criação de novas variáveis (mais na próxima seção) e na otimização de hiperparâmetros (usando o pacote [hyperopt](https://github.com/hyperopt/hyperopt)). 

Infelizmente não foi possível obter uma CV robusta. Os resultados na validação (utilzando diversas estratégias mudando o número de folds e repetições) se mostraram muito diferentes do que na leaderboard, provavelmente em razão da pequena quantidade de exemplos de teste e treino.

No final utilizei uma floresta de conditional inference trees com as novas variáveis que extraí, uma estratégia utilizada por diversos outros competidores. 

O resultado final foi uma acurácia de 0.80861, que ficaria entre o top 10%~25% da competição (a LB se atualiza com tempo, então não é possível dizer com certeza).

### Dados, variáveis, e visualização

Talvez a parte mais interessante da competição foi visualizar os dados dos passageiros que estavam envolvidos na tragédia do titanic. Pelo processamento dos dados, foi possível obter outras variáveis interessantes, além das já disponíveis (idade, sexo, etc):

* Número de pessoas em uma família
* Título (Mr, Mrs, etc)
* Deck (A, B, C ...)*
* Posição no barco (frente, meio, trás)*
* Sobrenome (indica pessoas da mesma família)

*em muitos exemplos esta informação não estava disponível, então foi necessário extrapolar os dados com um modelo específico 

Algumas visualizações interessantes podem ser feitas com estes dados:

#### Sobrevivência vs. Idade e Preço do ticket

Pessoas com maior poder aquisitivo tinham maior probabilidade de sobrevivência.

![](https://github.com/gdmarmerola/titanic-ml/blob/master/plots/fare-age.png)

#### Sobrevivência vs. Sexo e Idade

A taxa de mortalidade entre mulheres (codificadas como 0) foi muito menor que a entre homens (codificados como 1).

![](https://github.com/gdmarmerola/titanic-ml/blob/master/plots/age-sex.png)
 
#### Sobrevivência vs. Sexo e Preço do ticket 

Os dois padrões mostrados anteriormente se manifestam aqui.

![](https://github.com/gdmarmerola/titanic-ml/blob/master/plots/fare-sex.png)

#### Sobrevivência vs. Tamanho da família e Idade 

Famílias maiores e solteiros tiveram maior dificuldade para sobreviver.

![](https://github.com/gdmarmerola/titanic-ml/blob/master/plots/age-fsize.png)

#### Sobrevivência vs. Tamanho da família e Preço do ticket

Novamente, pessoas com maior poder aquisitivo tinham melhores chances de sobrevivência.

![](https://github.com/gdmarmerola/titanic-ml/blob/master/plots/fare-fsize.png)

#### Sobrevivência vs. Título e Idade

Mostrando novamente a tendência de mulheres sobreviverem mais que homens. O título "Master" em particular é dado para crianças, e por isso mostra maior número de sobreviventes.

![](https://github.com/gdmarmerola/titanic-ml/blob/master/plots/age-title.png)

#### Sobrevivência vs. Título e Preço do ticket

Idem ao anterior, mas mostrando a influência do poder aquisitivo dos passageiros.

![](https://github.com/gdmarmerola/titanic-ml/blob/master/plots/fare-title.png)

#### Sobrevivência vs. Deck e Idade

Mostra a taxa de sobrevivência dos passageiros alojados em cada andar do navio. Muitos pontos são decorrentes do ajuste de um modelo, e portanto não são, de fato, registros dos dados reais.

![](https://github.com/gdmarmerola/titanic-ml/blob/master/plots/age-deck.png)

#### Sobrevivência vs. Deck e Preço do ticket

Idem ao anterior.

![](https://github.com/gdmarmerola/titanic-ml/blob/master/plots/fare-deck.png)

#### Sobrevivência vs. Sobrenome e Idade

Aqui é possível observar a sobrevivência de elementos de cada família em particular (linhas com mais pontos representam casais ou solteiros). Em muitos casos todos os elementos da família faleceram ou sobreviveram juntos, poucas vezes existindo a situação de somente um membro sobreviver enquanto outros não.

![](https://github.com/gdmarmerola/titanic-ml/blob/master/plots/age-surname.png)
