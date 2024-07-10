import glob
import os
from PIL import Image


def tile_images_with_overlap(input_image_path, output_folder, tile_size, overlap_size):
    image_files = glob.glob(os.path.join(input_image_path, "*"))

    for img_index, img_file in enumerate(image_files):
        input_image = Image.open(img_file)

        width, height = input_image.size
        tile_width, tile_height = tile_size
        overlap_width, overlap_height = overlap_size

        # Tile'lar için sıra ve sütun sayısını hesapla
        num_rows = (height - overlap_height) // (tile_height - overlap_height)
        num_cols = (width - overlap_width) // (tile_width - overlap_width)

        # Tile'ları oluştur ve kaydet
        for row in range(num_rows + 1):
            for col in range(num_cols + 1):
                # Tile'ın koordinatlarını belirle
                left = col * (tile_width - overlap_width)
                upper = row * (tile_height - overlap_height)
                right = left + tile_width
                lower = upper + tile_height

                # Eğer tile, görüntü boyutlarını aşıyorsa kırpma işlemini geç
                if right > width:
                    right = width
                    left = width - tile_width
                if lower > height:
                    lower = height
                    upper = height - tile_height

                # Tile'ı kırp
                tile = input_image.crop((left, upper, right, lower))

                # Tile'ı kaydet
                tile_name = f"img{img_index + 1}_tile_{row}_{col}.png"
                tile.save(os.path.join(output_folder, tile_name))
