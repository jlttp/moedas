# Detecção e Classificação de Moedas

O programa faz a detecção das moedas, as classifica e faz o somatório delas.
Apenas 3 classes foram treinadas: 5 centavos, 10 centavos e 50 centavos.

# Uso do modelo treinado

O modelo salvo em "moedas.label" foi treinado usando as imagens que estão em "dataset" e já está pronto para uso:
```
python classify.py --model moedas.model --labelbin lb.pickle --image ./exemplos/img1.jpg
```
substituindo o path "./exemplos/img1.jpg" pelo path da imagem desejada.

Nota: as imagens de teste que estão no diretório "exemplos" foram obtidas em ambiente controlado e o nível de detecção é de 100%. É possível que o programa não detecte as moedas em outras imagens de forma satisfatória a depender da iluminação, resolução, distância focal, etc.

# Treinar novo modelo

```
python train.py --dataset dataset --model moedas.model --labelbin lb.pickle
```
- dataset: é o path onde se encontra o dataset para treinamento.
- moedas.model: é o path onde será salvo o modelo treinado.
- lb.pickle: é o path onde serão salvas as labels das classes.
