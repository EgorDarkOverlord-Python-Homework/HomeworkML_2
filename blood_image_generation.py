import numpy as np
import cv2
from PIL import Image, ImageDraw
import random
import os
from image_processing import * # Библиотека для аугментации изображения из прошлого семестра


class BloodImageGenerator:

    def __init__(self,
                 image_size=256,
                 background_color=(100, 20, 20),
                 cell_count_range=(5, 20),
                 cell_size_range=(10, 30),
                 cell_color_range=((200, 0, 0), (255, 150, 150)),
                 noise_percent = 1
                 ):
        self.image_size = image_size
        self.background_color = background_color
        self.cell_count_range = cell_count_range
        self.cell_size_range = cell_size_range
        self.cell_color_range = cell_color_range
        self.noise_percent = noise_percent


    def generate_blood_cell_image(self):
        # Инициализация изображения
        image = Image.new(
            "RGB", (self.image_size, self.image_size), self.background_color)
        draw = ImageDraw.Draw(image)
        # Вычисление количества клеток
        cell_count = random.randint(
            self.cell_count_range[0], self.cell_count_range[1])
        # Координаты и размеры клеток
        cells = []

        for n in range(cell_count):
            # Вычисление размера клетки
            cell_size = random.randint(
                self.cell_size_range[0], self.cell_size_range[1])
            # Вычисление позиции клетки
            x = random.randint(cell_size, self.image_size - cell_size)
            y = random.randint(cell_size, self.image_size - cell_size)
            # Добавление клетки
            cells.append((x, y, cell_size))
            # Вычисление цвета клетки
            cell_color = tuple(random.randint(
                self.cell_color_range[0][i], self.cell_color_range[1][i]) for i in range(3))
            # рисование клетки
            draw.ellipse((x - cell_size, y - cell_size, x +
                         cell_size, y + cell_size), fill=cell_color)

        # Преобразование изображения из Pillow в OpenCV
        open_cv_image = np.array(image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        # Зашумление изображения
        noise_image(open_cv_image, self.noise_percent)
        # Преобразование изображения из OpenCV в Pillow
        image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)

        return np.array(image), cell_count, cells


class BloodDatasetGenerator:

    def __init__(self, blood_image_generator, num_images=10, blood_img_dir='blood_dataset', blood_bgr_dir='blood_background', blood_cell_dir = 'blood_cell'):
        self.blood_image_generator = blood_image_generator
        self.num_images = num_images
        self.blood_img_dir = blood_img_dir
        self.blood_bgr_dir = blood_bgr_dir
        self.blood_cell_dir = blood_cell_dir


    def generate_dataset(self):
        if not os.path.exists(self.blood_img_dir):
            os.makedirs(self.blood_img_dir)

        images = []
        labels = []

        for i in range(self.num_images):
            image, cell_count, cells = self.blood_image_generator.generate_blood_cell_image()
            images.append(image)
            labels.append(cell_count)

            # Сохранение изображений
            image_path = os.path.join(self.blood_img_dir, f"image_{i}.png")
            # Convert to BGR for OpenCV
            cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        return images, labels
    

    def generate_dataset_divide_images(self):
        if not os.path.exists(self.blood_bgr_dir):
            os.makedirs(self.blood_bgr_dir)
        if not os.path.exists(self.blood_cell_dir):
            os.makedirs(self.blood_cell_dir)
        
        images = []
        labels = []
        cells_arr = []

        for i in range(self.num_images):
            image, cell_count, cells = self.blood_image_generator.generate_blood_cell_image()
            images.append(image)
            labels.append(cell_count)
            cells_arr.append(cells)

            # Сохранение изображений фона
            background_image = Image.new("RGB", (self.blood_image_generator.image_size, self.blood_image_generator.image_size), self.blood_image_generator.background_color)
            background_image_path = os.path.join(self.blood_bgr_dir, f"background_{i}.png")
            background_image.save(background_image_path)

            # Сохранение изображений клеток (вырезаем клетки и сохраняем)
            for j in range(cell_count):
                x, y, cell_size = cells[j]
                # Вырезаем клетку + небольшой отступ для контекста
                crop_size = int(cell_size * 2.5) # Размер вырезаемого квадрата, увеличил чтобы не обрезало
                x1 = max(0, x - cell_size)
                y1 = max(0, y - cell_size)
                x2 = min(self.blood_image_generator.image_size, x + cell_size)
                y2 = min(self.blood_image_generator.image_size, y + cell_size)
                cell_image = Image.fromarray(image[y1:y2, x1:x2])
                cell_image_path = os.path.join(self.blood_cell_dir, f"image_{i}_cell_{j}.png")
                cell_image.save(cell_image_path)
            
            return images, labels, cells_arr