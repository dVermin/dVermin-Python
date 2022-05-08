import itertools
import cv2
import os

import numpy as np
from visible_rect import RegionRect
from layer import Layer
from rectangle import Rectangle
class Node:
    """
    A node stands for some basic and fundamental elements it should have
    during following processing, such as width, height, x of the top-left point,
    and so on...
    Note! all of the node should satisfy that x or y of the top-left point is larger
    or equal to zero, that is because we need to move node and adjust all of the minus
    x or y value to positive, just for convenience.
    """

    def __init__(self, xml_node, director_path, offset_x=0, offset_y=0):
        self.node = xml_node
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.id = xml_node.attrib["id"]
        if "attributes_helperID" in xml_node.attrib:
            self.helper_id = xml_node.attrib["attributes_helperID"]
        else:
            self.helper_id = None
        self.alpha_value = float(xml_node.attrib["drawing_getAlpha"])
        self.translation_alpha = float(xml_node.attrib["drawing_getTransitionAlpha"])
        self.png = f"{xml_node.attrib['class']}@{xml_node.attrib['id']}.png"
        self.png_path = os.path.join(director_path, self.png)
        self.layer_png = f"{xml_node.attrib['class']}@{xml_node.attrib['id']}#layer.png"
        self.is_drawer = True if "drawer" in xml_node.attrib['class'].lower() else False
        self.layer_png_path = os.path.join(director_path, self.layer_png)
        self.height = int(xml_node.attrib['layout_getHeight'])
        self.width = int(xml_node.attrib['layout_getWidth'])
        self.screen_x = int(xml_node.attrib['layout_getLocationOnScreen_x'])
        self.screen_y = int(xml_node.attrib['layout_getLocationOnScreen_y'])
        self.z = float(xml_node.attrib['drawing_getZ'])
        self.class_name = xml_node.attrib["class"]
        self.padding_top = int(xml_node.attrib["padding_mPaddingTop"])
        self.padding_bottom = int(xml_node.attrib["padding_mPaddingBottom"])
        self.padding_left = int(xml_node.attrib["padding_mPaddingLeft"])
        self.padding_right = int(xml_node.attrib["padding_mPaddingRight"])
        self.ellipsize = self.get_ellipsize()
        self.x = self.screen_x + offset_x
        self.y = self.screen_y + offset_y
        self.has_layer = True if os.path.exists(self.layer_png_path) else False
        self.background_transparency = True
        self.has_png = True if os.path.exists(self.png_path) else False
        if self.has_png:
            self.transparency = not self.check_content(self.png_path)
        if self.has_layer:
            self.background_transparency=self.check_transparency(self.layer_png_path)
        self.md5_value = None
        self.can_scroll_vertically = False
        self.can_scroll_horizontally = False
        self.check_scrollable()
        self.content_area = None
        self.seq = 0
        self.non_repetitive_parent = None

    def check_transparency(self, path):
        """
        check if this image is transparent, which means if this layer contains
        some part of 0
        :param path:
        :return: return true if it is transparent
        """
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        mask = np.where(img[:,:,-1] > 0, 0, 1)
        if np.sum(mask) <= 0:
            return False
        else:
            return True

    def check_content(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        mask = np.where(img[:,:,-1] > 0, 1, 0)
        if np.sum(mask) > 0:
            return True
        else:
            return False

    def get_ellipsize(self):
        if "getEllipsize" in self.node.attrib:
            value = self.node.attrib["getEllipsize"].lower()
            if value == "null":
                return False
            else:
                return True
        return False

    def check_scrollable(self):
        class_name = self.class_name.lower()
        if "horizontalscrollview" in class_name or "viewpager" in class_name:
            self.can_scroll_horizontally = True
            return
        if "scrollview" in class_name or \
                "recyclerview" in class_name or \
                "listview" in class_name:
            self.can_scroll_vertically = True

    @staticmethod
    def check_node_scrollable(node):
        class_name = node.attrib["class"].lower()
        if "horizontalscrollview" in class_name or "viewpager" in class_name:
            return True
        if "scrollview" in class_name or \
                "recyclerview" in class_name or \
                "listview" in class_name:
            return True
        return False

    def get_ids_of_children(self):
        children = []
        for child in self.node.getchildren():
            children.append(child.attrib["id"])
        return children

    def compute_content_area(self):
        """
        get a mask containing True for indicating the information area in a component
        the size of a mask is the same with the component
        :return:
        """
        if self.content_area is None:
            content_image = cv2.imread(self.png_path, cv2.IMREAD_UNCHANGED)
            alpha_channel = content_image[:,:,-1]
            content_area = np.where(alpha_channel > 0, True, False)
            self.content_area = RegionRect(self.x, self.y, content_area)
        return self.content_area

    def compute_full_area(self):
        """
        get a mask containing True for indicating the information area in a component
        the size of a mask is the same with the component
        :return:
        """
        content_image = cv2.imread(self.png_path, cv2.IMREAD_UNCHANGED)
        alpha_channel = content_image[:,:,-1]
        content_area = np.where(alpha_channel > 0, True, True)
        return RegionRect(self.x, self.y, content_area)

    def compute_visible_region_rect(self):
        visible_area = np.full((self.height, self.width), True)
        return RegionRect(self.x, self.y, visible_area)

    def update_offset(self, offset_x, offset_y):
        """
        if the coordination of this view changed, the corresponding
        coordinator of top-left point should also be updated
        :param offset_x: the new x offset value
        :param offset_y: the new y offset value
        """
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.x += offset_x
        self.y += offset_y

    def get_layer(self, img_path=None, img_array=None):
        if img_array is None:
            img_array = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        return Layer(self.x, self.y, self.width, self.height, img_array=img_array)

    def get_rect(self):
        # print(self.x, self.y, self.x + self.width, self.y + self.height, sep=";")
        return Rectangle(self.x, self.y, self.x + self.width, self.y + self.height)

    @staticmethod
    def compute_offsets(root):
        """
        according to the positions of all nodes, adjust them to make sure
        the coordinators are non-negative
        :param root:
        :return:
        """
        offset_x = 0
        offset_y = 0
        for node in root.iter():
            node_x = int(node.attrib['layout_getLocationOnScreen_x'])
            if node_x < 0 and node_x < offset_x:
                offset_x = node_x
            node_y = int(node.attrib['layout_getLocationOnScreen_y'])
            if node_y < 0 and node_y < offset_y:
                offset_y = node_y
        offset_x = 0 - offset_x
        offset_y = 0 - offset_y
        return offset_x, offset_y

    def compute_layer_inside_rect(self, container_rectangle):
        rect = self.get_rect()
        overlapping_rectangle = container_rectangle & rect
        if overlapping_rectangle is None:
            return None, None
        crop_region = [
            overlapping_rectangle.x1 - rect.x1,
            overlapping_rectangle.y1 - rect.y1,
            overlapping_rectangle.x1 - rect.x1 + overlapping_rectangle.get_width(),
            overlapping_rectangle.y1 - rect.y1 + overlapping_rectangle.get_height()
        ]
        image = cv2.imread(self.png_path, cv2.IMREAD_UNCHANGED)
        image = image[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]]
        return Layer(overlapping_rectangle.x1,
                     overlapping_rectangle.y1,
                     overlapping_rectangle.get_width(),
                     overlapping_rectangle.get_height(),
                     img_array=image), overlapping_rectangle

    @staticmethod
    def compute_Layer_inside_rect(rect, container_rectangle, img_path):
        overlapping_rectangle = container_rectangle & rect
        if overlapping_rectangle is None:
            return None, None
        crop_region = [
            overlapping_rectangle.x - rect.x,
            overlapping_rectangle.y - rect.y,
            overlapping_rectangle.x - rect.x + overlapping_rectangle.width,
            overlapping_rectangle.y - rect.y + overlapping_rectangle.height
        ]
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image = image[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]]
        return Layer(overlapping_rectangle.x,
                     overlapping_rectangle.y,
                     overlapping_rectangle.width,
                     overlapping_rectangle.height,
                     img_array=image), overlapping_rectangle