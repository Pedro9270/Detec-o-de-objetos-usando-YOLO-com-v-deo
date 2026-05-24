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
        show=False,   # IMPORTANTE
        save=True,
        conf=0.5
    )

    # Tempo final
    fim = time.time()

    tempo_total = fim - inicio

    # =========================
    # CONFIANÇAS
    # =========================
    confiancas = []

    for r in results:

        if r.boxes is not None:

            for box in r.boxes:

                confianca = float(box.conf[0]) * 100
                confiancas.append(confianca)

    # Média
    if len(confiancas) > 0:
        media_acuracia = np.mean(confiancas)
    else:
        media_acuracia = 0

    # Salva dados
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

print("Gráfico de tempo salvo!")

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

print("Gráfico de acurácia salvo!")

# =========================
# MOSTRAR NO FINAL
# =========================
plt.show()

print("\nAnálise concluída!")