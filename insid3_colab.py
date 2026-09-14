# Execução opcional após todas as seções de analise_colab.py.
# Instale o INSID3 em uma célula Colab antes deste arquivo:
# !git clone -q https://github.com/visinf/INSID3.git /content/INSID3
# !pip install -q -r /content/INSID3/requirements.txt

# %% Máscara de referência da imagem principal
from PIL import ImageDraw

# Coordenadas provisórias em uma imagem 256 × 256. Confira visualmente.
pontos_fumaca = [
    (204, 35), (218, 56), (224, 84), (232, 110),
    (243, 140), (250, 178), (255, 215), (255, 252),
    (223, 239), (214, 198), (209, 165), (201, 131),
    (195, 101), (193, 66),
]

mascara_referencia = Image.new("L", (config["tamanho"], config["tamanho"]), 0)
ImageDraw.Draw(mascara_referencia).polygon(pontos_fumaca, fill=255)

plt.figure(figsize=(6, 6))
plt.imshow(imagens[indice_principal])
mascara_visivel = np.ma.masked_where(
    np.array(mascara_referencia) == 0, np.array(mascara_referencia)
)
plt.imshow(mascara_visivel, cmap="Reds", alpha=0.45, vmin=0, vmax=255)
plt.title("A máscara deve cobrir somente a fumaça")
plt.axis("off")
plt.show()


# %% Adaptador experimental: INSID3 usando os pesos HF já carregados
import sys

sys.path.insert(0, "/content/INSID3")
from models.insid3 import INSID3


class AdaptadorDINO(torch.nn.Module):
    def __init__(self, modelo_hf):
        super().__init__()
        self.modelo_hf = modelo_hf

        def vetor(valores):
            return torch.tensor(valores).view(1, 3, 1, 1)

        self.register_buffer("media_insid3", vetor([0.485, 0.456, 0.406]))
        self.register_buffer("desvio_insid3", vetor([0.229, 0.224, 0.225]))
        self.register_buffer("media_sat", vetor(config["media"]))
        self.register_buffer("desvio_sat", vetor(config["desvio"]))

    def get_intermediate_layers(self, x, n=1, reshape=True):
        # INSID3 normaliza como ImageNet; este checkpoint SAT-493M usa outros valores.
        pixels = x * self.desvio_insid3 + self.media_insid3
        pixels = (pixels - self.media_sat) / self.desvio_sat
        tokens = self.modelo_hf(pixel_values=pixels).last_hidden_state[:, inicio_patches:, :]
        altura = x.shape[-2] // tamanho_patch
        largura = x.shape[-1] // tamanho_patch
        assert tokens.shape[1] == altura * largura
        features = tokens.transpose(1, 2).reshape(x.shape[0], tokens.shape[-1], altura, largura)
        return (features,)


adaptador = AdaptadorDINO(modelo).to(dispositivo).eval()
segmentador = INSID3(
    encoder=adaptador,
    image_size=config["tamanho"],
    svd_components=64,
    device=dispositivo,
).eval()
print("INSID3 preparado com DINOv3 do Hugging Face.")


# %% Segmentação das imagens-alvo
for indice_alvo in config["imagens_alvo"]:
    segmentador.set_reference(imagens[indice_principal], mascara_referencia)
    segmentador.set_target(imagens[indice_alvo])
    mascara_prevista = segmentador.segment().cpu().numpy().astype(bool)

    figura, eixos = plt.subplots(1, 2, figsize=(10, 5))
    eixos[0].imshow(imagens[indice_alvo])
    eixos[0].set_title(f"Imagem original — índice {indice_alvo}")
    eixos[1].imshow(imagens[indice_alvo])
    eixos[1].imshow(
        np.ma.masked_where(~mascara_prevista, mascara_prevista),
        cmap="Reds", alpha=0.55, vmin=0, vmax=1,
    )
    eixos[1].set_title("Fumaça prevista pelo INSID3")
    for eixo in eixos:
        eixo.axis("off")
    plt.tight_layout()
    plt.show()

