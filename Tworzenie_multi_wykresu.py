import matplotlib.pyplot as plt
import re
import glob
import os
import numpy as np


resolution = 300

# Wzorzec do wyciągania nazw serii z nazw plików
series_name_pattern = re.compile(r'training_log_(.*)\.txt')

# Wyrażenie regularne do wyciągania wartości z linii
pattern = re.compile(r'Epoch \[(\d+)/\d+\], Step \[(\d+)/\d+\], Loss: ([\d.]+), Accuracy: ([\d.]+)')


# Funkcja do obliczania średnich co 'resolution' punktów
def average_data(data, resolution):
    avg_data = [sum(data[i:i + resolution]) / resolution for i in range(0, len(data) - resolution + 1, resolution)]
    return avg_data

def median_filter(data, resolution):
    median_data = [np.median(data[i:i + resolution]) for i in range(0, len(data) - resolution + 1, resolution)]
    return median_data

# Pobierz wszystkie pliki pasujące do wzorca w bieżącym katalogu
file_list = glob.glob("training_log_*.txt")

# Przygotowanie wykresu
plt.figure(figsize=(10, 5))

# Listy do zbierania wspólnych danych
all_epoch_steps = []

# Definiowanie różnych kolorów dla wykresów
colors = ['blue', 'orange', 'green', 'purple', 'red']
color_index = 0

# Tworzenie wykresów dla każdego pliku
for file in file_list:
    # Wczytaj dane z pliku
    with open(file) as f:
        lines = f.readlines()

    # Wyciągnij nazwę serii z nazwy pliku
    series_name = series_name_pattern.search(os.path.basename(file)).group(1)

    # Podziel dane na osie Global Step, Loss i Accuracy
    global_steps = []
    losses = []
    accuracies = []
    epoch_steps = []

    global_step = 0
    for line in lines:
        match = pattern.search(line)
        if match:
            epoch = int(match.group(1))
            step = int(match.group(2))
            loss = float(match.group(3))
            accuracy = float(match.group(4))

            global_steps.append(global_step)
            global_step += 1

            losses.append(loss)
            accuracies.append(accuracy)
            if step == 1 and global_step > 1:  # Zaczyna się nowa epoka
                epoch_steps.append(global_step - 1)

    # Dodaj do wspólnej listy
    all_epoch_steps.extend(epoch_steps)

    # Oblicz średnie co 'resolution' punktów
    avg_steps = average_data(global_steps, resolution)
    avg_losses = average_data(losses, resolution)
    avg_accuracies = average_data(accuracies, resolution)

    # Użyj tego samego koloru dla danej serii w obu wykresach
    current_color = colors[color_index]

    # Wykres Loss vs Step
    plt.subplot(1, 2, 1)
    plt.plot(avg_steps, avg_losses, label=f"Loss ({series_name})", color=current_color, linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss vs Step")
    plt.legend()

    # Wykres Accuracy vs Step
    plt.subplot(1, 2, 2)
    plt.plot(avg_steps, avg_accuracies, label=f"Accuracy ({series_name})", color=current_color, linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Step")
    plt.legend()

    # Zmieniamy kolor na następny w kolejności
    color_index = (color_index + 1) % len(colors)

# Dodaj pionowe linie przy rozpoczęciu nowej epoki dla każdego z plików
plt.subplot(1, 2, 1)
for step in all_epoch_steps:
    plt.axvline(x=step, color='r', linestyle='--')

plt.subplot(1, 2, 2)
for step in all_epoch_steps:
    plt.axvline(x=step, color='r', linestyle='--')

# Zapisz wykres
plt.tight_layout()
output_path = r'F:\zzzfolder_roboczy\wykresy\nowe\wykres_zbiorczy.png'
plt.savefig(output_path)
plt.show()
