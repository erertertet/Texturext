
from PIL import Image

image = Image.open('img/result.png')
# Get the dimensions of the original image
width, height = image.size

# divide the image into 4 equal parts and rotate their position
# top left -> bottom left
# top right -> top left
# bottom left -> bottom right
# bottom right -> top right
# Get the dimensions of each tile
tile_width = width // 2
tile_height = height // 2
# Create a new image with the same size as the original
new_image = Image.new('RGBA', (width, height))
# Paste the tiles into their new positions
new_image.paste(image.crop((0, 0, tile_width, tile_height)), (0, tile_height))  # top left -> bottom left
new_image.paste(image.crop((tile_width, 0, width, tile_height)), (0, 0))  # top right -> top left
new_image.paste(image.crop((0, tile_height, tile_width, height)), (tile_width, height // 2))  # bottom left -> bottom right
new_image.paste(image.crop((tile_width, tile_height, width, height)), (tile_width, 0))  # bottom right -> top right
# Save the new image
new_image.save('img/gcobble.png')