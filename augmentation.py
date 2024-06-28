
import os
import random
import torchvision.transforms.functional as TF
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

def custom_augmentation(image_path, mask_path, output_image_dir, output_mask_dir, new_size):
    # Görüntü ve maskeyi yükleme
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')

    # Yeni boyut
    new_size = (new_size, new_size)

    # Transformlar (boyutlandırma ve augmentasyon)
    resize_transform = transforms.Resize(new_size)
    image = resize_transform(image)
    mask = resize_transform(mask)

    # Rastgele yatay ve dikey olarak yansıtma
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)
    
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)

    # Renk jitter (rastgele renk değişiklikleri)
    brightness_factor = random.uniform(0.8, 1.2)
    contrast_factor = random.uniform(0.8, 1.2)
    saturation_factor = random.uniform(0.8, 1.2)
    hue_factor = random.uniform(-0.1, 0.1)

    image = TF.adjust_brightness(image, brightness_factor)
    image = TF.adjust_contrast(image, contrast_factor)
    image = TF.adjust_saturation(image, saturation_factor)
    image = TF.adjust_hue(image, hue_factor)

    # Görüntüyü ve maskeleri tensor formatına dönüştürme
    image_tensor = TF.to_tensor(image)
    mask_tensor = TF.to_tensor(mask)

    # Kaydedilecek dosya yollarını oluşturma
    image_name = os.path.basename(image_path)
    mask_name = os.path.basename(mask_path)
    output_image_path = os.path.join(output_image_dir, f'augmented_image_{image_name}')
    output_mask_path = os.path.join(output_mask_dir, f'augmented_mask_{mask_name}')

    # Çıkış dizinlerini oluşturma
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    # Görüntüyü ve maskeleri kaydetme
    transforms.ToPILImage()(image_tensor).save(output_image_path)
    transforms.ToPILImage()(mask_tensor).save(output_mask_path)

new_size = int(input("What do you want the new size to be? "))

base_dir = input("Enter the base directory path of the dataset: ")
test_dir_name = input("Enter the name of the test directory: ")
ground_truth_dir_name = input("Enter the name of the ground truth directory: ")

while True:
    file_name = input("Enter the directory name within test/ground_truth (or 'q' to quit): ")
    
    if file_name == "q":
        break
        
    else:
        input_dir = os.path.join(base_dir, test_dir_name, file_name)
        mask_dir = os.path.join(base_dir, ground_truth_dir_name, file_name)
        output_image_dir = os.path.join(base_dir, f'augmented_images_{new_size}', file_name)
        output_mask_dir = os.path.join(base_dir, f'augmented_masks_{new_size}', file_name)
    
        # Giriş dizinindeki tüm görüntü ve maskeler için işlem
        for filename in tqdm(os.listdir(input_dir), desc="Processing: "):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(input_dir, filename)
                mask_filename = filename.replace('.jpg', '.png').replace('.png', '_mask.png')
                mask_path = os.path.join(mask_dir, mask_filename)
        
                custom_augmentation(image_path, mask_path, output_image_dir, output_mask_dir, new_size=new_size)
