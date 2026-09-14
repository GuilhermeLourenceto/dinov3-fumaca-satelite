# %% Importações e autenticação
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from transformers import AutoImageProcessor, AutoModel
from huggingface_hub import notebook_login
from google.colab import drive

drive.mount("/content/drive")
notebook_login()
dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
print("Dispositivo:", dispositivo)


# %% Configurações: altere somente esta seção
config = {
    "modelo": "facebook/dinov3-vitl16-pretrain-sat493m",
    "pasta": "/content/drive/MyDrive/Dataset_David_2024/FireDatasetClass/Train/Fire",
    "imagem_principal": 493,
    "imagens_alvo": [887, 1183],
    "tamanho": 256,
    "media": [0.430, 0.411, 0.296],
    "desvio": [0.213, 0.156, 0.143],
    "quantidade_kmeans": 2,
    "metodo_dendrograma": "average",
    "distancia": "cosine",
    "corte_dendrograma": 0.4,
}


# %% Catálogo de imagens
pasta = Path(config["pasta"])
assert pasta.is_dir(), f"Pasta não encontrada: {pasta}"
extensoes = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
arquivos = sorted(
    caminho for caminho in pasta.iterdir()
    if caminho.is_file() and caminho.suffix.lower() in extensoes
)
assert arquivos, "Nenhuma imagem encontrada."
print("Total de imagens:", len(arquivos))

indices_catalogo = np.linspace(0, len(arquivos) - 1, min(18, len(arquivos)), dtype=int)
figura, eixos = plt.subplots(3, 6, figsize=(15, 8))
for eixo in eixos.flat:
    eixo.axis("off")
for eixo, indice in zip(eixos.flat, indices_catalogo):
    with Image.open(arquivos[indice]) as arquivo:
        eixo.imshow(ImageOps.exif_transpose(arquivo).convert("RGB"))
    eixo.set_title(f"Índice {indice}")
plt.tight_layout()
plt.show()


# %% Abrir imagens selecionadas
def carregar_imagem(indice):
    with Image.open(arquivos[indice]) as arquivo:
        imagem = ImageOps.exif_transpose(arquivo).convert("RGB")
    return imagem.resize((config["tamanho"], config["tamanho"]), Image.Resampling.BILINEAR)


indices_escolhidos = list(dict.fromkeys([config["imagem_principal"], *config["imagens_alvo"]]))
assert all(0 <= indice < len(arquivos) for indice in indices_escolhidos)
imagens = {indice: carregar_imagem(indice) for indice in indices_escolhidos}
figura, eixos = plt.subplots(1, len(indices_escolhidos), figsize=(4 * len(indices_escolhidos), 4), squeeze=False)
for eixo, indice in zip(eixos.flat, indices_escolhidos):
    eixo.imshow(imagens[indice])
    eixo.set_title(f"Índice {indice}")
    eixo.axis("off")
plt.tight_layout()
plt.show()


# %% Carregar DINOv3 diretamente do Hugging Face
processador = AutoImageProcessor.from_pretrained(
    config["modelo"], image_mean=config["media"], image_std=config["desvio"]
)
modelo = AutoModel.from_pretrained(
    config["modelo"], attn_implementation="eager"
).to(dispositivo).eval()
tamanho_patch = int(modelo.config.patch_size)
assert config["tamanho"] % tamanho_patch == 0
grade = config["tamanho"] // tamanho_patch
inicio_patches = 1 + int(modelo.config.num_register_tokens)


def preparar_entrada(imagem):
    return processador(
        images=imagem, do_resize=False, do_center_crop=False, return_tensors="pt"
    ).to(dispositivo)


def extrair_patches(imagem):
    with torch.inference_mode():
        saida = modelo(**preparar_entrada(imagem))
    tokens = saida.last_hidden_state[0, inicio_patches:].float()
    tokens = torch.nn.functional.normalize(tokens, dim=-1)
    assert len(tokens) == grade * grade
    return tokens.cpu().numpy()


indice_principal = config["imagem_principal"]
patches = extrair_patches(imagens[indice_principal])
print("Patch tokens:", patches.shape, "| grade:", grade, "×", grade)


# %% K-means em uma imagem (cores são IDs, não classes semânticas)
kmeans = KMeans(n_clusters=config["quantidade_kmeans"], random_state=42, n_init="auto")
mapa_kmeans = kmeans.fit_predict(patches).reshape(grade, grade)
plt.figure(figsize=(6, 6))
plt.imshow(mapa_kmeans, cmap="viridis", interpolation="nearest")
plt.title(f"K-means — índice {indice_principal}")
plt.colorbar(label="Grupo")
plt.axis("off")
plt.show()


# %% Dendrograma e mapa hierárquico de uma imagem
Z = linkage(patches, method=config["metodo_dendrograma"], metric=config["distancia"])
plt.figure(figsize=(18, 6))
dendrogram(Z, no_labels=True, color_threshold=config["corte_dendrograma"])
plt.axhline(config["corte_dendrograma"], color="black", linestyle="--")
plt.title(f"Dendrograma dos patches — índice {indice_principal}")
plt.xlabel("Patches")
plt.ylabel("Distância cosseno")
plt.show()

grupos = fcluster(Z, t=config["corte_dendrograma"], criterion="distance")
mapa_grupos = grupos.reshape(grade, grade)
print("Número de grupos:", len(np.unique(grupos)))
plt.figure(figsize=(6, 6))
plt.imshow(mapa_grupos, cmap="viridis", interpolation="nearest")
plt.title(f"Grupos do dendrograma — índice {indice_principal}")
plt.colorbar(label="Grupo")
plt.axis("off")
plt.show()


# %% Atenção direta do CLS e rollout (não são máscaras de fumaça)
with torch.inference_mode():
    saida = modelo(**preparar_entrada(imagens[indice_principal]), output_attentions=True)
atencoes = saida.attentions
assert atencoes is not None and all(a is not None for a in atencoes)

ultima = atencoes[-1][0].float().mean(dim=0)
atencao_direta = ultima[0, inicio_patches:].reshape(grade, grade).cpu().numpy()
identidade = torch.eye(atencoes[0].shape[-1], device=atencoes[0].device)
resultado = identidade.clone()
for atencao in atencoes:
    media = atencao[0].float().mean(dim=0) + identidade
    media = media / media.sum(dim=-1, keepdim=True)
    resultado = media @ resultado
rollout = resultado[0, inicio_patches:].reshape(grade, grade).cpu().numpy()

figura, eixos = plt.subplots(1, 3, figsize=(15, 5))
eixos[0].imshow(imagens[indice_principal])
eixos[0].set_title("Imagem original")
eixos[1].imshow(atencao_direta, cmap="inferno")
eixos[1].set_title("Atenção direta do CLS")
eixos[2].imshow(rollout, cmap="inferno")
eixos[2].set_title("Attention rollout")
for eixo in eixos:
    eixo.axis("off")
plt.tight_layout()
plt.show()

