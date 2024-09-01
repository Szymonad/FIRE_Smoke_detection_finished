import cv2
import torch
import torch.nn.functional as F
from model import ConvNet
from torchvision import transforms
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.transforms import v2
import datetime

start = datetime.datetime.now()
# Konfiguracja urządzenia
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parametry modelu (muszą być takie same jak podczas treningu)
input_size = 480
hidden_size = 1024
num_layers = 1
num_classes = 3
proj_size = hidden_size
i = 0
u = 0
# Wczytanie modelu
model = ConvNet(num_classes)
model.load_state_dict(torch.load('model.pth'))
model = model.to(device)
model.eval()



transform = transforms.Compose([
        v2.ToPILImage(),
        v2.Resize((input_size, input_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])




def predict_single(frame):
    # Transformacja klatki wideo i zmiana kształtu tensora
    frame_transformed = transform(frame).to(device)
    frame_transformed = frame_transformed.unsqueeze(0)  # Dodanie wymiaru batch_size

    outputs = model(frame_transformed)
    probabilities = F.softmax(outputs, dim=1)
    predicted = torch.max(probabilities, 1)[1]
    return predicted.item(), probabilities.squeeze().tolist()

# Ścieżki do katalogów z nagraniami
directories = ['D:\\Praca magisterska\\kod\\V1\\pythonProject\\.venv\\videos\\test\\ogien',
               'D:\\Praca magisterska\\kod\\V1\\pythonProject\\.venv\\videos\\test\\dym',
               'D:\\Praca magisterska\\kod\\V1\\pythonProject\\.venv\\videos\\test\\nic']
# directories = ['D:\\Praca magisterska\\kod\\V1\\pythonProject\\.venv\\videos\\test_generalizacja\\ogien',
#                'D:\\Praca magisterska\\kod\\V1\\pythonProject\\.venv\\videos\\test_generalizacja\\dym',
#                'D:\\Praca magisterska\\kod\\V1\\pythonProject\\.venv\\videos\\test_generalizacja\\nic']

correctly_classified = 0
incorrectly_classified = 0

# Tworzenie folderów dla błędnie sklasyfikowanych klatek

#output_folders = ['bledne_odpowiedzi_test_generalizacja\\ogien', 'bledne_odpowiedzi_test_generalizacja\\dym', 'bledne_odpowiedzi_test_generalizacja\\nic']
output_folders = ['bledne_odpowiedzi\\ogien', 'bledne_odpowiedzi\\dym', 'bledne_odpowiedzi\\nic']


for folder in output_folders:
    os.makedirs(folder, exist_ok=True)

# Inicjalizacja macierzy pomyłek
confusion = np.zeros((num_classes, num_classes))

for directory in directories:
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        cap = cv2.VideoCapture(filepath)

        if not cap.isOpened():
            print(f"Error: Could not open video {filepath}.")
            continue

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read the next frame from {filepath}.")
                i = i+1
                print(i)
                break

            # Przewidywanie na podstawie pojedynczej klatki
            u = u + 1
            if u % 1000 == 0:
                print(u)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            prediction, probabilities = predict_single(frame)
            labels = ["ogien", "dym", "nic"]
            detected = labels[prediction]

            # Sprawdzenie poprawności klasyfikacji
            ground_truth_label = os.path.basename(directory)
            if detected == ground_truth_label:
                correctly_classified += 1
            else:
                incorrectly_classified += 1
                # Zapisanie klatki do odpowiedniego foldera
                #output_folder = f'bledne_odpowiedzi_test_generalizacja\\{detected}'
                output_folder = f'bledne_odpowiedzi\\{detected}'
                output_path = os.path.join(output_folder, f"{filename}_frame{incorrectly_classified}.jpg")
                cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Aktualizacja macierzy pomyłek
            confusion[labels.index(ground_truth_label), prediction] += 1

        # Zwolnienie zasobów
        cap.release()

# Obliczanie czułości i swoistości
sensitivity = np.diag(confusion) / np.sum(confusion, axis=1)
specificity = [(confusion[1][1]+confusion[2][2])/(confusion[1][0]+confusion[2][0]+confusion[1][1]+confusion[2][2]),
               (confusion[0][0]+confusion[2][2])/(confusion[0][1]+confusion[2][1]+confusion[0][0]+confusion[2][2]),
               (confusion[0][0]+confusion[1][1])/(confusion[0][2]+confusion[1][2]+confusion[0][0]+confusion[1][1])]

confusion = confusion.astype(int)
sensitivity = [round(x, 4) for x in sensitivity]
specificity = [round(x, 4) for x in specificity]

# Tworzenie DataFrame z wynikami
results_df = pd.DataFrame({
    "Klasa": labels,
    "Czułość": sensitivity,
    "Swoistość": specificity,
})
# Zapis do pliku Excel
#results_df.to_excel("bledne_odpowiedzi_test_generalizacja\\Wyniki.xlsx", index=False)
results_df.to_excel("bledne_odpowiedzi\\Wyniki.xlsx", index=False)


# Wyświetlanie macierzy pomyłek jako grafiki
fig, ax = plt.subplots(2, 1, figsize=(10, 12))
im = ax[0].imshow(confusion, interpolation="nearest", cmap=plt.cm.Greys, vmin=0, vmax=0)  # Ustawienie mapy kolorów na szarości z minimalną i maksymalną wartością
#ax[0].set_title("Macierz pomyłek")
ax[0].set_xticks(np.arange(len(["ogień", "dym", "nie wykryto"])))
ax[0].set_yticks(np.arange(len(["ogień", "dym", "nie wykryto"])))
ax[0].set_xticklabels(["ogień", "dym", "nie wykryto"])
ax[0].set_yticklabels(["ogień", "dym", "nie wykryto"])

for i in range(len(labels)):
    for j in range(len(labels)):
        ax[0].add_patch(
            plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='black'))  # Ustawienie granic komórek
        text = ax[0].text(j, i, confusion[i, j], ha="center", va="center", fontsize=12)

ax[0].text(-1, 1.5, "Prawdziwe etykiety", fontsize=14, rotation=90)
ax[0].text(0.3, 2.9, "Przewidywane etykiety", fontsize=14)
ax[0].text(0.3, -0.7, "Macierz pomyłek", fontsize=18)

# Dodawanie tabelki do grafiki
ax[1].axis("off")
table = ax[1].table(
    cellText=results_df.values,
    colLabels=results_df.columns,
    cellLoc="center",
    bbox=[0.18, 0.75, 0.65, 0.25],
    colColours=["#FFD700"] * len(results_df.columns),
)
table.auto_set_font_size(False)
table.set_fontsize(12)

# # Dodawanie słupka kolorów po lewej stronie
# cax = fig.add_axes([0.75, 0.5, 0.02, 0.4])
# cbar = plt.colorbar(im, cax=cax)
# cbar.set_label("Wartości macierzy pomyłek", fontsize=14)

# Zapis grafiki
#fig.savefig("bledne_odpowiedzi_test_generalizacja\\Macierz_z_tabelka.jpg")
fig.savefig("bledne_odpowiedzi\\Macierz_z_tabelka.jpg")

duration = datetime.datetime.now() - start
print(duration)
#with open(os.path.join('bledne_odpowiedzi_test_generalizacja', "duration.txt"), "w") as file:
with open(os.path.join('bledne_odpowiedzi', "duration.txt"), "w") as file:
    file.write(str(duration))

#with open('bledne_odpowiedzi_test_generalizacja\\dane.txt', 'a') as f:
with open('bledne_odpowiedzi\\dane.txt', 'a') as f:
    f.write(f"Poprawnie sklasyfikowane klatki: {correctly_classified}\n")
    f.write(f"Błędnie sklasyfikowane klatki: {incorrectly_classified}\n")
    f.write(f"Czułość: {sensitivity}\n")
    f.write(f"Swoistość: {specificity}\n")











