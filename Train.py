
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from torch.utils.data import DataLoader, ConcatDataset, default_collate
from model import LSTM, ConvNet
from data_preprocessing import FrameDataset
import datetime
import shutil  # Import dodatkowy do kopiowania plików

# Ścieżka do folderu results
results_path = 'results'
os.makedirs(results_path, exist_ok=True)  # Utwórz folder results, jeśli nie istnieje

device = torch.device('cuda')
print(f"Używam urządzenia: {device}")
input_size = 480
num_classes = 3
num_epochs = 1
batch_size = 1
clip_value = 1
learning_rate = 0.0000000001

transform = FrameDataset.prepare_transform(input_size)

##################linux################
# root_dirs = ['/home/szyada7565/praca_magisterska/version1.0/kod_projektu/videos/Train_cut/ogien', '/home/szyada7565/praca_magisterska/version1.0/kod_projektu/videos/Train_cut/dym', '/home/szyada7565/praca_magisterska/version1.0/kod_projektu/videos/Train_cut/nic']
# labels = [
#     torch.full((len(os.listdir('/home/szyada7565/praca_magisterska/version1.0/kod_projektu/videos/Train_cut/ogien')),), 0, dtype=torch.long),  # Etykieta 0 dla ogienia
#     torch.full((len(os.listdir('/home/szyada7565/praca_magisterska/version1.0/kod_projektu/videos/Train_cut/dym')),), 1, dtype=torch.long),    # Etykieta 1 dla dymu
#     torch.full((len(os.listdir('/home/szyada7565/praca_magisterska/version1.0/kod_projektu/videos/Train_cut/nic')),), 2, dtype=torch.long)     # Etykieta 2 dla nic
# ]
############### Windows
root_dirs = ['./videos/Train_cut/ogien', './videos/Train_cut/dym', './videos/Train_cut/nic']
labels = [torch.full((len(os.listdir(dir)),), i, dtype=torch.long) for i, dir in enumerate(root_dirs)]

datasets = FrameDataset.prepare_datasets(root_dirs, labels, input_size, device='cpu')
combined_dataset = ConcatDataset(datasets)

if __name__ == '__main__':
    train_loader = DataLoader(dataset=combined_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, multiprocessing_context="spawn")
    model = ConvNet(num_classes).to(device)
    #Wczytaj model
    # model_path = 'model.pth'
    # if os.path.isfile(model_path):
    #     model.load_state_dict(torch.load(model_path))
    #     print("Model wczytany")


    start = datetime.datetime.now()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train_model(train_loader, num_epochs, criterion, optimizer, clip_value)

    duration = datetime.datetime.now() - start
    print(duration)
    model_path = os.path.join(results_path, 'model.pth')
    torch.save(model.state_dict(), model_path)

    with open(os.path.join(results_path, "duration.txt"), "w") as file:
        file.write(str(duration))

    # Kopiowanie plików źródłowych do folderu results
    shutil.copy('/home/szyada7565/praca_magisterska/version1.0/kod_projektu/Train.py', results_path)
    shutil.copy('/home/szyada7565/praca_magisterska/version1.0/kod_projektu/data_preprocessing.py', results_path)
    shutil.copy('/home/szyada7565/praca_magisterska/version1.0/kod_projektu/model.py', results_path)
    shutil.copy('training_log.txt', results_path)
    # Kompresowanie folderu results do pliku .zip
    shutil.make_archive('results', 'zip', results_path)

    os.system("echo skończyłem")
