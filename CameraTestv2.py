import cv2
import torch
import torch.nn.functional as F
from model import ConvNet
from torchvision import transforms

# Konfiguracja urządzenia
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parametry modelu (muszą być takie same jak podczas treningu)
input_size = 480
hidden_size = 1024
num_layers = 1
num_classes = 3
proj_size = hidden_size

# Wczytanie modelu
#model = ConvLSTM(input_size, hidden_size, num_layers, num_classes, proj_size)
model = ConvNet(num_classes)
model.load_state_dict(torch.load('model.pth'))
model = model.to(device)
model.eval()

# Transformacja klatek wideo
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor()
])

# Funkcja do przewidywania na podstawie pojedynczej klatki
def predict_single(frame):
    frame_transformed = transform(frame).to(device)
    frame_transformed = frame_transformed.unsqueeze(0)  # Dodanie wymiaru batch_size

    with torch.no_grad():
        outputs = model(frame_transformed)
        probabilities = F.softmax(outputs, dim=1)
        predicted = torch.max(probabilities, 1)[1]
        return predicted.item(), probabilities.squeeze().tolist()

# Otwarcie kamery internetowej
cap = cv2.VideoCapture(0)  # '0' to zazwyczaj domyślny numer kamery internetowej

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the next frame.")
        break

    # Przewidywanie na podstawie pojedynczej klatki
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    prediction, probabilities = predict_single(frame)
    labels = ["ogien", "dym", "nic"]
    detected = labels[prediction]

    # Formatowanie tekstu do wyświetlenia
    text = f"wykryto ogien: {probabilities[0] * 100:.2f}%\n" \
           f"wykryto dym: {probabilities[1] * 100:.2f}%\n" \
           f"wykryto nic: {probabilities[2] * 100:.2f}%\n" \
           f"na obrazie jest: {detected}"

    # Dodanie tekstu do klatki
    y0, dy = 50, 30
    for i, line in enumerate(text.split('\n')):
        y = y0 + i * dy
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Wyświetlenie klatki
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Frame', frame)

    # Wyjście z pętli po naciśnięciu klawisza 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Zwolnienie zasobów
cap.release()
cv2.destroyAllWindows()
