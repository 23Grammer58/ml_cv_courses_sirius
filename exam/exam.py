import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    autocheck_model, 
    set_seed, 
    visualize_prediction, 
    bbox_iou,
    load_config,
    check_dataset_exists,
    load_dataset_config,
    load_dataset_to_df,
    ObjectDetectionDataset
)

# ВАЖНО: Замените на ваш ID студента (например, '01', '02', ..., '10')
STUDENT_ID = "01"  # Замените на ваш номер!

# ======================================================
# Модель для детекции объектов (Обязательно реализуйте вашу модель здесь)
# ======================================================

class BBoxRegressor(nn.Module):
    def __init__(self, architecture_name):
        """
        Реализуйте модель для задачи детекции объектов (регрессии bbox)
        
        Args:
            architecture_name (str): Имя архитектуры из конфигурации
        """
        super(BBoxRegressor, self).__init__()
        
        # TODO: Реализуйте вашу модель здесь
        # Например, используйте предобученную сеть в качестве backbone
        # и добавьте слои для предсказания bbox
        
        # Пример архитектуры (вам нужно полностью реализовать её):
        if architecture_name == 'resnet18':
            # Например: использовать resnet18 из torchvision.models
            pass
        elif architecture_name == 'mobilenet_v2':
            # Например: использовать mobilenet_v2 из torchvision.models
            pass
        elif architecture_name == 'efficientnet_b0':
            # Например: использовать efficientnet_b0 из torchvision.models
            pass
        else:
            raise ValueError(f"Неизвестная архитектура: {architecture_name}")
        
        # Инициализация backbone сети
        self.backbone = nn.Sequential(
            # Реализуйте backbone сеть здесь
        )
        
        # Слои для предсказания bbox (x, y, w, h)
        self.bbox_head = nn.Sequential(
            # TODO: Реализуйте слои для предсказания координат bbox
            # Последний слой должен выдавать 4 значения: x_center, y_center, width, height
        )
        
    def forward(self, x):
        """
        Реализуйте forward-проход модели
        
        Args:
            x: Входные изображения (батч)
            
        Returns:
            Предсказания координат bbox (x_center, y_center, width, height)
        """
        # TODO: Реализуйте forward-проход
        features = self.backbone(x)
        # Тут может потребоваться изменение формы тензора или другие операции
        # в зависимости от выбранной архитектуры
        bbox_pred = self.bbox_head(features)
        
        return bbox_pred

# ======================================================
# Обучение модели (Обязательно реализуйте вашу функцию обучения здесь)
# ======================================================

def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device):
    """
    Реализуйте функцию для обучения модели
    
    Args:
        model: Модель для обучения
        train_loader: DataLoader для обучающего набора
        valid_loader: DataLoader для валидационного набора
        criterion: Функция потерь
        optimizer: Оптимизатор
        num_epochs: Количество эпох
        device: Устройство для обучения (cpu/cuda)
        
    Returns:
        Обученная модель
    """
    model.to(device)
    
    # История обучения для построения графиков
    history = {
        'train_loss': [],
        'valid_loss': [],
        'mean_iou': []
    }
    
    # TODO: Реализуйте цикл обучения
    for epoch in range(num_epochs):
        # Обучение
        model.train()
        train_loss = 0.0
        
        # TODO: Реализуйте цикл обучения для одной эпохи
        # Ваш код здесь...
        
        # Валидация
        model.eval()
        valid_loss = 0.0
        all_ious = []
        
        # TODO: Реализуйте цикл валидации для одной эпохи
        # Ваш код здесь...
        
        # TODO: Вычислите метрики и добавьте их в историю
        # Например:
        # train_loss = ...
        # valid_loss = ...
        # mean_iou = ...
        
        # Вывод информации о процессе обучения
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Valid Loss: {valid_loss:.4f}, "
              f"Mean IoU: {mean_iou:.4f}")
    
    # Построение графиков обучения
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['valid_loss'], label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['mean_iou'], label='Mean IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return model

# ======================================================
# Главная функция
# ======================================================

def main():
    # Проверяем, указан ли ID студента
    if STUDENT_ID == "XX":
        print("ОШИБКА: Пожалуйста, замените 'XX' на ваш ID студента в начале файла!")
        return
    
    # Загружаем конфигурацию и проверяем наличие датасета
    config = load_config(STUDENT_ID)
    if config is None:
        return
    
    if not check_dataset_exists(STUDENT_ID):
        return
    
    data_config = load_dataset_config(STUDENT_ID)
    if data_config is None:
        return
    
    # Устанавливаем seed для воспроизводимости
    set_seed(config.get('seed', 42))
    
    # Загружаем данные
    train_df = load_dataset_to_df(STUDENT_ID, 'train')
    valid_df = load_dataset_to_df(STUDENT_ID, 'valid')
    
    if train_df is None or valid_df is None:
        return
    
    # Создаем датасеты и загрузчики данных
    # TODO: Настройте трансформации для обучения и валидации
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        # TODO: Добавьте нужные преобразования для обучения
        # Например: изменение размера, аугментации и т.д.
    ])
    
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        # TODO: Добавьте нужные преобразования для валидации
    ])
    
    # Создаем датасеты
    train_dataset = ObjectDetectionDataset(train_df, transform=train_transform)
    valid_dataset = ObjectDetectionDataset(valid_df, transform=valid_transform)
    
    # Создаем загрузчики данных
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
    
    # Создаем модель
    architecture_name = config.get('arch', 'mobilenet_v2')
    model = BBoxRegressor(architecture_name)
    
    # Определяем функцию потерь и оптимизатор
    # TODO: Подберите подходящую функцию потерь и оптимизатор
    criterion = nn.MSELoss()  # Подумайте, подходит ли MSELoss для задачи или нужно что-то другое
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Подберите подходящий lr и другие параметры
    
    # Определяем устройство для обучения
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")
    
    # Обучаем модель
    # TODO: Используйте свою функцию train_model вместо train_bbox_model
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=20,  # TODO: Подберите оптимальное количество эпох
        device=device
    )
    
    # Сохраняем модель
    torch.save(trained_model.state_dict(), f"model_student_{STUDENT_ID}.pt")
    print(f"Модель сохранена в файл model_student_{STUDENT_ID}.pt")
    
    # Проверяем качество модели
    print("\nПроверка качества модели на валидационном наборе:")
    mean_iou, success_rate = autocheck_model(
        model=trained_model, 
        student_id=STUDENT_ID,
        device=device
    )


if __name__ == "__main__":
    main() 