import cv2
import MTM
from rectangle import Rectangle
import numpy as np


class Layer:
    def __init__(self, x, y, node_width, node_height, img_array=None):
        self.x = x
        self.height = node_height
        self.width = node_width
        self.y = y
        self.img_array = img_array

    def get_rectangle(self):
        return Rectangle(self.x, self.y, self.x + self.width, self.y + self.height)

    @staticmethod
    def get_layer_with_rect(rect, img_array):
        return Layer(rect.x1,
                     rect.y1,
                     rect.get_width(),
                     rect.get_height(),
                     img_array=img_array)

    def expand_region(self, another_layer):
        self_rect = self.get_rectangle()
        another_layer_rect = another_layer.get_rectangle()
        intersect = self_rect & another_layer_rect
        if intersect is None or intersect is not None and intersect.area() < another_layer_rect.area():
            parent_node_layer_png_img = self.img_array
            image = []
            for i in range(parent_node_layer_png_img.shape[-1]):
                sub_color = np.max(parent_node_layer_png_img[:, :, i])
                image.append(np.full((another_layer.height, another_layer.width), sub_color))
            image = np.reshape(np.array(image, dtype=np.uint8),
                               (another_layer.height, another_layer.width, parent_node_layer_png_img.shape[-1]))
            return Layer(another_layer.x,
                         another_layer.y,
                         another_layer.width,
                         another_layer.height,
                         img_array=image)
        else:
            return self

    def expand_own_region(self, color_):
        new_image = [color_ for _ in range(self.width*3 * self.height*3)]
        new_image = np.array(new_image, dtype=np.uint8)
        new_image = np.reshape(new_image,
                                        (self.height*3, self.width*3, 4))
        new_image[self.height:self.height+self.height, self.width:self.width+self.width] = self.img_array
        return Layer(self.x,
                     self.y,
                     self.width*3,
                     self.height*3,
                     new_image), self.width, self.height # dont use x and y of this layer!!!!!!!


    @staticmethod
    def compute_layer_inside_rect(inner_layer, outer_rect):
        inner_rect = inner_layer.get_rectangle()
        intersect_rect = inner_rect & outer_rect
        new_x = intersect_rect.x1 - inner_rect.x1
        new_y = intersect_rect.y1 - inner_rect.y1
        intersect_image = inner_layer.img_array[new_y:new_y + intersect_rect.get_height(),
                          new_x:new_x + intersect_rect.get_width()]
        return Layer.get_layer_with_rect(intersect_rect, intersect_image)
