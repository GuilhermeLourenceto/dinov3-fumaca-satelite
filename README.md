# Análise de fumaça em imagens de satélite com DINOv3

Código de iniciação científica para inspecionar features de patch do DINOv3 ViT-L/16 SAT-493M em imagens de satélite. O repositório separa três perguntas:

1. Como os patches de **uma imagem** se agrupam? `analise_colab.py` mostra K-means, dendrograma com ligação média e distância cosseno, e o mapa de grupos.
2. Para onde o token CLS dirige atenção? O mesmo arquivo compara a atenção direta da última camada com attention rollout. Esses mapas **não são máscaras de fumaça**.
3. Uma máscara de fumaça em uma imagem ajuda a segmentar outras? `insid3_colab.py` usa o código do INSID3 com um adaptador experimental para o DINOv3 carregado pelo Hugging Face.

## Uso no Google Colab

Ative uma GPU, monte o Drive e entre em uma conta Hugging Face com acesso aprovado a [`facebook/dinov3-vitl16-pretrain-sat493m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m). As células estão separadas por `# %%` nos arquivos `.py`; copie cada seção para uma célula do Colab, na ordem.

Instale as dependências da análise antes de executar `analise_colab.py`:

```python
!pip install -q "transformers==5.5.1" huggingface_hub scipy scikit-learn matplotlib pillow
```

Na seção **Configurações**, ajuste `pasta`, `imagem_principal` e `imagens_alvo`. Os números das imagens são índices da lista ordenada de arquivos; o catálogo os exibe. Nenhum arquivo `_annotations.coco.json` é lido ou alterado.

Para a etapa opcional do INSID3, execute as instruções do início de `insid3_colab.py` após a análise. Confira a máscara de referência antes de interpretar as previsões. O INSID3 oficial usa o backbone da implementação Meta; o adaptador deste projeto preserva o requisito de carregar os pesos pelo Hugging Face. Essa integração e os parâmetros para imagens de 256 px ainda precisam de validação experimental.

## Interpretação inicial

No exemplo com índices 493, 887 e 1183, o INSID3 acompanhou boa parte das plumas, mas produziu falsos positivos no índice 887 e limites imprecisos. A atenção direta e o rollout ficaram dispersos. Esses resultados visuais não constituem medida quantitativa de qualidade; uma avaliação exige máscaras de referência verificadas e imagens de teste separadas.

## Referências

- [DINOv3 no Hugging Face](https://huggingface.co/docs/transformers/model_doc/dinov3)
- [INSID3: código e instruções oficiais](https://github.com/visinf/INSID3)
- [Attention rollout: Abnar e Zuidema (2020)](https://aclanthology.org/2020.acl-main.385/)


