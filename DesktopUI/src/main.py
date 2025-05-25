import sys
import torch
import torch.nn.functional as F
import timm
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap
from torchvision import transforms
from PIL import Image

classes = ['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat', 'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra']  # Örneğin, ImageFolder ile eğitilen klasör adları

model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=len(classes))
model.load_state_dict(torch.load("src/best_model_10.pt", map_location=torch.device("cpu")))
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class AnimalDetector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hayvan Sınıflandırıcı")

        self.layout = QVBoxLayout()

        self.label = QLabel("Bir fotoğraf seçin")
        self.image_label = QLabel()
        self.image_label.setScaledContents(True)
        self.image_label.setFixedSize(400, 400)

        self.button = QPushButton("Fotoğraf Seç")
        self.button.clicked.connect(self.select_image)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.button)
        self.setLayout(self.layout)

    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Fotoğraf Seç", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            self.image_label.setPixmap(QPixmap(file_name))

            image = Image.open(file_name).convert("RGB")
            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)
                predicted = torch.argmax(probs, dim=1).item()
                confidence = probs[0, predicted].item() * 100

                prediction = classes[predicted]

            self.label.setText(f"Tahmin: {prediction} — Güven: %{confidence:.2f}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnimalDetector()
    window.show()
    sys.exit(app.exec_())
