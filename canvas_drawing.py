import skia

from node import Layer, Rectangle
import numpy as np

def get_canvas_size(layers):
    width = 0
    height = 0
    x = None
    y = None
    for layer in layers:
        if x is None and y is None:
            x = layer.x
            y = layer.y
        width = width if width > layer.x + layer.width else layer.x + layer.width
        height = height if height > layer.y + layer.height else layer.y + layer.height
        x = layer.x if layer.x < x else x
        y = layer.y if layer.y < y else y
    return width-x, height-y, x, y


def draw_according_to_rect(layers, rect):
    width, height, x, y = get_canvas_size(layers)
    surface = skia.Surface(width, height)
    with surface as canvas:
        for layer in layers:
            image = skia.Image.fromarray(layer.img_array.astype(np.uint8))
            canvas.drawImage(image, layer.x - x, layer.y - y)
    drawn_image = surface.makeImageSnapshot().toarray()
    new_rect = Rectangle(rect.x1 - x, rect.y1 - y, rect.x2 - x, rect.y2 - y)
    cropped_drawn_image = new_rect.crop_image(drawn_image)
    return Layer.get_layer_with_rect(rect, cropped_drawn_image)
