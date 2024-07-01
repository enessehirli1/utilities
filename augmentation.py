import os
import random
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

def apply_augmentations(image, augmentations, mask=None):
    applied_augmentations = []
    
    # Yatay çevirme
    if 'hflip' in augmentations:
        image = TF.hflip(image)
        if mask is not None:
            mask = TF.hflip(mask)
        applied_augmentations.append('hflip')
    
    # Dikey çevirme
    if 'vflip' in augmentations:
        image = TF.vflip(image)
        if mask is not None:
            mask = TF.vflip(mask)
        applied_augmentations.append('vflip')

    # Renk jitter
    if 'color_jitter' in augmentations:
        brightness_factor = random.uniform(0.8, 1.2)
        contrast_factor = random.uniform(0.8, 1.2)
        saturation_factor = random.uniform(0.8, 1.2)
        hue_factor = random.uniform(-0.1, 0.1)
        image = TF.adjust_brightness(image, brightness_factor)
        image = TF.adjust_contrast(image, contrast_factor)
        image = TF.adjust_saturation(image, saturation_factor)
        image = TF.adjust_hue(image, hue_factor)
        applied_augmentations.append('color_jitter')

    # Döndürme
    if 'rotate' in augmentations:
        angle = random.uniform(-30, 30)
        image = TF.rotate(image, angle)
        if mask is not None:
            mask = TF.rotate(mask, angle)
        applied_augmentations.append('rotate')

    # TrivialAugmentWide
    if 'trivial_augment_wide' in augmentations:
        trivial_augment = transforms.TrivialAugmentWide()
        image = trivial_augment(image)
        applied_augmentations.append('trivial_augment_wide')

    # Noise
    if 'noise' in augmentations:
        noise = np.random.normal(0, 0.1, size=(image.size[1], image.size[0], 3))
        image = np.array(image) / 255.0 + noise
        image = np.clip(image, 0, 1)
        image = Image.fromarray((image * 255).astype(np.uint8))
        applied_augmentations.append('noise')
        
    if mask is not None:
        return image, mask, applied_augmentations
    else:
        return image, applied_augmentations

def custom_augmentation(image_path, mask_path=None, output_image_dir=None, output_mask_dir=None, new_size=None, augmentations=[]):
    # Görüntü ve maskeyi yükleme
    image = Image.open(image_path).convert('RGB')
    mask = None
    if mask_path is not None:
        mask = Image.open(mask_path).convert('L')

    # Boyutlandırma
    new_size = (new_size, new_size)
    resize_transform = transforms.Resize(new_size)
    image = resize_transform(image)
    if mask is not None:
        mask = resize_transform(mask)

    # Augmentasyonları uygulama
    if mask is not None:
        image, mask, applied_augmentations = apply_augmentations(image, augmentations, mask)
    else:
        image, applied_augmentations = apply_augmentations(image, augmentations)

    # Görüntüyü ve maskeleri tensor formatına dönüştürme
    image_tensor = TF.to_tensor(image)
    if mask is not None:
        mask_tensor = TF.to_tensor(mask)

    # Augmentasyon isimlerini dosya adına ekleme
    augmentations_suffix = '_'.join(applied_augmentations)

    # Kaydedilecek dosya yollarını oluşturma
    image_name = os.path.basename(image_path).split('.')[0]
    output_image_path = os.path.join(output_image_dir, f'augmented_image_{image_name}.png')
    if mask_path is not None:
        mask_name = os.path.basename(mask_path).split('.')[0]
        output_mask_path = os.path.join(output_mask_dir, f'augmented_mask_{mask_name}.png')

    # Çıkış dizinlerini oluşturma
    os.makedirs(output_image_dir, exist_ok=True)
    if mask_path is not None:
        os.makedirs(output_mask_dir, exist_ok=True)

    # Görüntüyü ve maskeleri kaydetme
    transforms.ToPILImage()(image_tensor).save(output_image_path)
    if mask is not None:
        transforms.ToPILImage()(mask_tensor).save(output_mask_path)

def get_valid_input(prompt, expected_type):
    while True:
        user_input = input(prompt)
        if expected_type == int:
            try:
                return int(user_input)
            except ValueError:
                print(f"Lütfen geçerli bir {expected_type.__name__} değeri girin.")
        elif expected_type == str:
            if user_input.isdigit():
                print(f"Lütfen geçerli bir {expected_type.__name__} değeri girin.")
            else:
                return user_input

def get_augmentations():
    augmentations = []
    if input("Horizontal Flip (yatay çevirme) kullanılsın mı? (y/n): ").lower() == 'y':
        augmentations.append('hFlip')
    if input("Vertical Flip (dikey çevirme) kullanılsın mı? (y/n): ").lower() == 'y':
        augmentations.append('vFlip')
    if input("Color Jitter (renk jitter) kullanılsın mı? (y/n): ").lower() == 'y':
        augmentations.append('colorJitter')
    if input("Rotate (döndürme) kullanılsın mı? (y/n): ").lower() == 'y':
        augmentations.append('rotate')
    if input("Trivial Augment Wide kullanılsın mı? (y/n): ").lower() == 'y':
        augmentations.append('triAugWide')
    if input("Noise (gürültü) kullanılsın mı? (y/n): ").lower() == 'y':
        augmentations.append('noise')
    return augmentations

def start_augmentate():
    new_size = get_valid_input("What do you want the new size to be? ", int)
    base_dir = get_valid_input("Enter the base directory path of the dataset: ", str)
    test_dir_name = get_valid_input("Enter the name of the test directory: ", str)
    gt = int(input("Is there a mask file [1/0]: "))
    ground_truth_dir_name = None
    if gt == 1:
        ground_truth_dir_name = get_valid_input("Enter the name of the ground truth directory: ", str)

    augmentations = get_augmentations()

    augmentations_suffix = '_'.join(augmentations) if augmentations else 'no_augmentations'

    while True:
        file_name = get_valid_input("Enter the directory name within test/ground_truth (or 'q' to quit): ", str)
        if file_name == "q":
            break
        else:
            input_dir = os.path.join(base_dir, test_dir_name, file_name)
            output_image_dir = os.path.join(base_dir, f'augmented_images_{new_size}_{augmentations_suffix}', file_name)
            output_mask_dir = None
            if gt == 1:
                mask_dir = os.path.join(base_dir, ground_truth_dir_name, file_name)
                output_mask_dir = os.path.join(base_dir, f'augmented_masks_{new_size}_{augmentations_suffix}', file_name)

            # Giriş dizinindeki tüm görüntü ve maskeler için işlem
            for filename in tqdm(os.listdir(input_dir), desc="Processing: "):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    image_path = os.path.join(input_dir, filename)
                    if gt == 1:
                        mask_filename = filename.replace('.jpg', '.png').replace('.png', '_mask.png')
                        mask_path = os.path.join(mask_dir, mask_filename)
                        custom_augmentation(image_path, mask_path, output_image_dir, output_mask_dir, new_size=new_size, augmentations=augmentations)
                    else:
                        custom_augmentation(image_path, None, output_image_dir, None, new_size=new_size, augmentations=augmentations)
