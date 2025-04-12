from PIL import Image

image = Image.open('img/oak_plank.png')
# Get the dimensions of the original image
width, height = image.size

# Create a new 32x32 image
new_image = Image.new('RGB', (width * 2, height * 2))

# Resize the original to 16x16 and paste four times
new_image.paste(image, (0, 0))    # Top-left
new_image.paste(image, (width, 0))   # Top-right
new_image.paste(image, (0, height))   # Bottom-left
new_image.paste(image, (width, height))  # Bottom-right

# scale it by 8 times and blur it

new_image.save('img/cobblestone_2xttttt2.png')
# new_image.show()
