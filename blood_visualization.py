import random
import os
import matplotlib.pyplot as plt
from PIL import Image

class BloodVisualization:
    def visualize(self, images, labels, num_samples=5):
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        for i in range(num_samples):
            index = random.randint(0, len(images) - 1)
            axes[i].imshow(images[index])
            axes[i].set_title(f"Кол-во клеток: {labels[index]}")
            axes[i].axis('off')
        plt.show()

    def visualize_divided(self, blood_bgr_dir='blood_background', blood_cell_dir = 'blood_cell'):
        background = Image.open(os.path.join(blood_bgr_dir, random.choice(os.listdir(blood_bgr_dir))))
        cell = Image.open(os.path.join(blood_cell_dir, random.choice(os.listdir(blood_cell_dir))))
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(background)
        plt.title("Фон")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(cell)
        plt.title("Клетка")
        plt.axis("off")

        plt.show()