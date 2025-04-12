# read the image of cobble stone and make 4 copy, output a 32 * 32 image

from PIL import Image

image = Image.open('img/cobblestone.png')
# Get the dimensions of the original image
width, height = image.size

# Create a new 32x32 image
new_image = Image.new('RGB', (32, 32))

# Resize the original to 16x16 and paste four times
small_img = image.resize((16, 16))
new_image.paste(small_img, (0, 0))    # Top-left
new_image.paste(small_img, (16, 0))   # Top-right
new_image.paste(small_img, (0, 16))   # Bottom-left
new_image.paste(small_img, (16, 16))  # Bottom-right

# scale it by 8 times and blur it
new_image = new_image.resize((32 * 2, 32 * 2), Image.BICUBIC)

new_image.save('img/cobblestone_2x2.png')
new_image.show()
