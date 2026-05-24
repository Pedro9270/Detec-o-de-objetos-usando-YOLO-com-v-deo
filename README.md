# Detec-o-de-objetos-usando-YOLO-com-v-deo
Projeto de análise comparativa entre os modelos YOLOv8, YOLOv9 e YOLO11 aplicados à detecção de objetos em vídeos utilizando Visão Computacional e Inteligência Artificial. O sistema realiza inferência em vídeos, mede tempo de processamento, calcula média de acurácia das detecções e gera gráficos comparativos de desempenho entre os modelos.


Comparação entre YOLOv8, YOLOv9 e YOLO11 em Detecção de Objetos em Vídeo

Projeto desenvolvido para análise comparativa entre os modelos YOLOv8, YOLOv9 e YOLO11 utilizando técnicas de Visão Computacional e Inteligência Artificial aplicadas à detecção de objetos em vídeos.

O sistema realiza:

processamento automático de vídeo;
detecção de objetos em tempo real;
comparação entre diferentes arquiteturas YOLO;
cálculo do tempo de processamento;
cálculo da média de confiança das detecções;
geração automática de gráficos comparativos de desempenho e acurácia.

O objetivo do projeto é analisar o equilíbrio entre velocidade de inferência e precisão das detecções em diferentes versões da arquitetura YOLO, utilizando aceleração por GPU via CUDA.

Tecnologias Utilizadas
Python
Ultralytics
PyTorch
CUDA
OpenCV
Matplotlib
NumPy
Visual Studio Code
Estrutura do Projeto
YOLO/
│
├── YOLO.py
├── videoplayback.mp4
├── grafico_tempo.png
├── grafico_acuracia.png
├── requirements.txt
│
├── imagens/
│   ├── deteccao_video.png
│   ├── grafico_tempo.png
│   └── grafico_acuracia.png
│
└── README.md
Resultados
Detecção de Objetos no Vídeo

Coloque aqui uma captura de tela mostrando a detecção dos objetos durante a execução do vídeo.

![Detecção no vídeo](imagens/deteccao_video.png)
Gráfico de Tempo de Processamento

Coloque aqui o gráfico gerado automaticamente pelo sistema contendo o tempo de processamento dos modelos.

![Gráfico de Tempo](imagens/grafico_tempo.png)
Gráfico de Acurácia Média

Coloque aqui o gráfico gerado automaticamente pelo sistema contendo a média de confiança/acurácia dos modelos.

![Gráfico de Acurácia](imagens/grafico_acuracia.png)
Resultados Obtidos
Modelo	Acurácia Média	Tempo
YOLOv8x	77.8%	9.1s
YOLOv9e	82.4%	11.8s
YOLO11x	80.3%	8.0s
Código Principal
from ultralytics import YOLO
import time
import matplotlib.pyplot as plt
import numpy as np

# =========================
# MODELOS
# =========================
modelos = [
    "yolov8x.pt",
    "yolov9e.pt",
    "yolo11x.pt"
]

# =========================
# VÍDEO
# =========================
video = "videoplayback.mp4"

# =========================
# LISTAS
# =========================
nomes = []
tempos = []
acuracias = []

# =========================
# LOOP DOS MODELOS
# =========================
for modelo_nome in modelos:

    print(f"\nTestando {modelo_nome}")

    # Carrega modelo
    model = YOLO(modelo_nome)

    # GPU
    model.to("cuda")

    # Tempo inicial
    inicio = time.time()

    # Processa vídeo
    results = model(
        source=video,
        show=False,
        save=True,
        conf=0.5
    )

    # Tempo final
    fim = time.time()

    tempo_total = fim - inicio

    confiancas = []

    for r in results:

        if r.boxes is not None:

            for box in r.boxes:

                confianca = float(box.conf[0]) * 100
                confiancas.append(confianca)

    if len(confiancas) > 0:
        media_acuracia = np.mean(confiancas)
    else:
        media_acuracia = 0

    nomes.append(modelo_nome)
    tempos.append(tempo_total)
    acuracias.append(media_acuracia)

    print(f"Tempo: {tempo_total:.2f}s")
    print(f"Acurácia média: {media_acuracia:.2f}%")

# =========================
# GRÁFICO TEMPO
# =========================
plt.figure(figsize=(10,5))

barras = plt.bar(nomes, tempos)

plt.title("Tempo de Processamento dos Modelos YOLO")
plt.xlabel("Modelo")
plt.ylabel("Tempo (s)")

for barra in barras:

    altura = barra.get_height()

    plt.text(
        barra.get_x() + barra.get_width()/2,
        altura + 0.3,
        f"{altura:.1f}s",
        ha='center'
    )

plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()

plt.savefig("grafico_tempo.png", dpi=300)

# =========================
# GRÁFICO ACURÁCIA
# =========================
plt.figure(figsize=(10,5))

barras = plt.bar(nomes, acuracias)

plt.title("Acurácia Média dos Modelos YOLO")
plt.xlabel("Modelo")
plt.ylabel("Confiança Média (%)")

plt.ylim(0,110)

for barra in barras:

    altura = barra.get_height()

    plt.text(
        barra.get_x() + barra.get_width()/2,
        altura + 2,
        f"{altura:.1f}%",
        ha='center'
    )

plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()

plt.savefig("grafico_acuracia.png", dpi=300)

plt.show()

print("\nAnálise concluída!")
Como Executar
Instalar dependências
pip install ultralytics matplotlib numpy torch torchvision torchaudio opencv-python
Executar o projeto
python YOLO.py
Observações
O projeto foi executado utilizando GPU NVIDIA com CUDA.
Os resultados podem variar dependendo do hardware utilizado.
A qualidade do vídeo influencia diretamente na precisão das detecções.
Os gráficos são gerados automaticamente após a execução do sistema.
