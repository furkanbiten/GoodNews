import argparse
from PIL import Image
import sys
import os

def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.ANTIALIAS)

def resize_images(image_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    for i, image in enumerate(images):
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                img = img.convert('RGB')
                img.save(os.path.join(output_dir, image), img.format)
        sys.stdout.write('\r Processing images: %d/%d images processed...' % (i, num_images))
   	sys.stdout.flush()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--img_size', type=int, default=256, 
                        help='which size an image will be resized to')
	parser.add_argument('--root', type=str, default="../images/",
                        help='which size an image will be resized to')
	args = parser.parse_args()
	image_size = [args.img_size, args.img_size]
	resize_images(args.root+'images/', args.root+"resized/", size = image_size)	
