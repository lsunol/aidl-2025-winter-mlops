import os

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import skimage.io
from sklearn.preprocessing import LabelEncoder
import numpy as np

class MyDataset(Dataset):

    def __init__(self, images_path, labels_path, transform=None):

        self.data = pd.read_csv(labels_path)  # 'data' contendrá el DataFrame con nombres y etiquetas
        self.images_path = images_path  # Ruta a la carpeta de imágenes
        self.transform = transform  # Transformaciones opcionales
        self.data.info()
        self.labels = self.data['value'].to_numpy()
        self.data['encoded_value'] = LabelEncoder().fit_transform(self.labels)
        print('LABELS:')
        self.encoded_labels = np.unique(self.data['encoded_value'])
        print(self.encoded_labels)

    
    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        
        img_name = self.create_image_file_name(self.data.iloc[idx])
        label = self.data.iloc[idx, 5] 
        image = self.read_image(img_name)
        # Aplicar las transformaciones si están definidas
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_labels(self):
        return self.encoded_labels
        # return sorted(self.data['value'].unique())
        
    def create_image_file_name(self, x):
        file_name = f"input_{x[0]}_{x[1]}_{x[2]}.jpg"
        return file_name

    def read_image(self, image_file_name):
        image = Image.open(self.images_path + '/' + image_file_name)

        return image

def showRandomPixels(image: Image.Image, num_pixels: int = 5):
    """
    Muestra valores de píxeles aleatorios de una imagen.

    Parámetros:
    - image: PIL.Image.Image - La imagen de la cual obtener los píxeles.
    - num_pixels: int - Número de píxeles aleatorios a mostrar (predeterminado: 5).
    """
    # Convertir la imagen a un array de numpy
    image_np = np.array(image)

    # Tamaño de la imagen (altura y anchura)
    height, width = image_np.shape[:2]

    print(f"Mostrando {num_pixels} píxeles aleatorios:")
    for _ in range(num_pixels):
        # Generar coordenadas aleatorias
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        # Obtener el valor del píxel en (y, x)
        pixel_value = image_np[y, x]
        print(f"Píxel en ({y}, {x}): {pixel_value}")

