import io
import json
import math
import os
import pathlib
import re
import collections
import time
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from skimage import img_as_float, img_as_ubyte
from skimage.color import rgb2gray
import glob
import MTM
import skimage
from skimage import measure
from lxml import etree
import numpy as np
import numpy.ma as ma

from detect_peak import get_binary_image_of_text
from li_to_xml import write_xml
from rectangle import Rectangle
from node import Node
from layer import Layer
from canvas_drawing import draw_according_to_rect
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from visible_rect import RegionRect


def check_transparency(png_path):
    """
    check if this image is transparent
    :param png_path:
    :return:
    """
    is_transparent = True
    node_layer_img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    for i in range(node_layer_img.shape[-1]):
        if np.max(node_layer_img[:, :, i]) > 1:
            is_transparent = False
            break
    return is_transparent


class Item:
    def __init__(self, helper_id, xpath, view_id, is_leaf=False):
        self.helper_id = helper_id
        self.view_id = view_id
        self.xpath = xpath
        self.feature = ""
        self.is_leaf = is_leaf
        self.children = []


class MappingSize:
    def __init__(self, root, size, intra_analyzer):
        self.root = root
        self.size = size
        self.intra = intra_analyzer


class MappingTool:
    def __init__(self, mapping_sizes):
        self.mapping_sizes = mapping_sizes

    def update_tree(self, mapping_size_list):
        """
        update the tree, inject id into nodes without mapping id, which has children containing mapping ids
        :param mapping_size_list:
        :return:
        """
        def traverse_tree(root):
            children_same = False
            count = {}
            children = root.getchildren()
            for child in children:
                child_name = traverse_tree(child)
                if child_name not in count:
                    count[child_name] = 0
                count[child_name] += 1
                if count[child_name] > 1:
                    children_same = True
                    break
            if children_same or "" in count:
                return ""
            elif len(children) == 0 or len(count) == 0:
                return root.attrib["attributes_helperID"] if "attributes_helperID" in root.attrib else ""
            else:
                if "attributes_helperID" not in root.attrib:
                    root.attrib["attributes_helperID"] = "/".join(list(count.keys())) + "/parent"
                return root.attrib["attributes_helperID"]

        for mapping_size in mapping_size_list:
            traverse_tree(mapping_size.root)

    def find_repetition(self, mapping_size_list):
        """

        :param mapping_size_list:
        :return:
        """
        repetitive_id_set = set({})

        def traverse_tree(root, helper_id_set):
            children = root.getchildren()
            layer_count = {}
            for child in children:
                if "attributes_helperID" in child.attrib:
                    if child.attrib["attributes_helperID"] not in layer_count:
                        layer_count[child.attrib["attributes_helperID"]] = 0
                    layer_count[child.attrib["attributes_helperID"]] += 1
                    if layer_count[child.attrib["attributes_helperID"]] > 1:
                        helper_id_set.add(child.attrib["attributes_helperID"])
                traverse_tree(child, helper_id_set)

        for mapping_size in mapping_size_list:
            traverse_tree(mapping_size.root, repetitive_id_set)

        return repetitive_id_set

    def generate_feature_items(self, mapping_size_list, repetitive_id_set):
        """

        :param mapping_size_list:
        :param repetitive_id_set:
        :return:
        """
        feature_dict = {}

        def get_items(node_, items, item=None, in_repetitive_component=False):
            """
            get repetitive items according to repetitive_id,

            :param node_:
            :param item:
            """
            parent_is_repetitive_container = False
            if item is not None:
                for child in node_.getchildren():
                    if "attributes_helperID" in child.attrib:
                        child_helper_id = child.attrib["attributes_helperID"]
                    else:
                        child_helper_id = None
                    if child_helper_id in repetitive_id_set:
                        parent_is_repetitive_container = True
            for child in node_.getchildren():
                rep_item = item
                if item is not None and item.helper_id == "e28ce3":
                    item.helper_id = "e28ce3"
                if "attributes_helperID" in child.attrib:
                    child_helper_id = child.attrib["attributes_helperID"]
                else:
                    child_helper_id = None
                if child_helper_id in repetitive_id_set or in_repetitive_component or parent_is_repetitive_container:
                    in_repetitive_component = True
                    if child_helper_id in repetitive_id_set or parent_is_repetitive_container:
                        rep_item = Item(child_helper_id, child.getroottree().getpath(child), view_id=child.attrib["id"],
                                        is_leaf=True if len(child.getchildren()) == 0 else False)
                        rep_item.feature = "Rep"
                        items.append(rep_item)
                    if rep_item is not None:
                        rep_item.feature = f"{rep_item.feature}/{child_helper_id}"
                        if "text_mText" in child.attrib:
                            rep_item.feature = rep_item.feature + "/" + child.attrib["text_mText"]
                        if len(child.getchildren()) == 0:
                            rep_item.children.append(
                                Item(child_helper_id, child.getroottree().getpath(child), view_id=child.attrib["id"],
                                     is_leaf=True if len(child.getchildren()) == 0 else False))
                else:
                    if child_helper_id is not None:
                        rep_item = Item(child_helper_id, child.getroottree().getpath(child), view_id=child.attrib["id"],
                                        is_leaf=True if len(child.getchildren()) == 0 else False)
                        if len(child.getchildren()) == 0:
                            rep_item.feature = f"Dangling/{child_helper_id}"
                        else:
                            rep_item.feature = f"NonRep/{child_helper_id}"
                        items.append(rep_item)
                get_items(child, items, rep_item, in_repetitive_component)

        for mapping_size in mapping_size_list:
            items_list = []
            new_list = []
            item_dict = {}
            get_items(mapping_size.root, items_list)
            for item in items_list:
                if item.feature.startswith("Dangling") or item.feature.startswith("NonRep"):
                    new_list.append(item)
                else:
                    children = item.children
                    for child in children:
                        child_item = Item(child.helper_id,
                                          child.xpath,
                                          view_id=child.view_id,
                                          is_leaf=child.is_leaf)
                        child_item.feature = f"Sub{item.feature}/{child.helper_id}"
                        child_item.in_rep = True
                        child_item.parent_view_id = item.view_id
                        child_item.parent_helper_id = item.helper_id
                        child_item.parent_xpath = item.xpath
                        child_item.parent_feature = item.feature
                        child_item.parent_is_leaf = item.is_leaf
                        new_list.append(child_item)
            for item in new_list:
                if item.feature not in item_dict:
                    item_dict[item.feature] = []
                item_dict[item.feature].append(item)
            feature_dict[mapping_size.size] = item_dict
        return feature_dict

    def new_find_mapping(self, size_small, size_big):
        self.update_tree(self.mapping_sizes)
        repetitive_id_set = self.find_repetition(self.mapping_sizes)
        feature_dict = self.generate_feature_items(self.mapping_sizes, repetitive_id_set)
        result_feature = {}
        for normal_leaf in feature_dict[size_small]:
            if normal_leaf not in result_feature:
                result_feature[normal_leaf] = {
                    "normal": None,
                    "bigger": None
                }
            result_feature[normal_leaf]["normal"] = feature_dict[size_small][normal_leaf]

        for bigger_leaf in feature_dict[size_big]:
            if bigger_leaf not in result_feature:
                result_feature[bigger_leaf] = {
                    "normal": None,
                    "bigger": None
                }
            result_feature[bigger_leaf]["bigger"] = feature_dict[size_big][bigger_leaf]
        removed_keys = []
        for item in result_feature:
            remove = False
            for size in result_feature[item]:
                if result_feature[item][size] is None or \
                        result_feature[item][size] is not None and \
                        len(result_feature[item][size]) < 1:
                    remove = True
                    break
                if len(result_feature[item][size]) >= 1:
                    result_feature[item][size] = result_feature[item][size][0]
            if remove and "SubRep" in item:
                removed_keys.append(item)
        for item in removed_keys:
            del result_feature[item]
        return result_feature


class InterAnalyzer:
    """
    used for detect invalid view inside view tree by comparing
    also used for generating completeness score for those visible view
    """
    def __init__(self, page_director, sub_director, view_hierarchy_name="ViewHierarchy.xml"):
        """
        init denoiser with some indispensable arguments
        :param page_director: the director containing all image and li file
        :param view_hierarchy_name:
        """
        self.page_director = page_director
        self.sub_director = sub_director
        self.view_hierarchy = os.path.join(page_director, view_hierarchy_name)
        self.denoise_result = None
        self.canvas = None
        self.offset_x = 0
        self.offset_y = 0
        self.result = {}
        parser = etree.XMLParser(encoding="utf-8")
        tree = etree.parse(self.view_hierarchy, parser=parser)
        self.dataset = {}
        self.root = tree.getroot()
        # self.id_identifier(self.root, [], [])
        root = self.root
        self.offset_x, self.offset_y = Node.compute_offsets(root)
        self.init_dataset(root)
        root_node_info = Node(root, self.page_director, self.offset_x, self.offset_y)
        root_node_id = root_node_info.id
        root_node_info.seq = 0
        self.root_node_info = root_node_info
        self.dataset[root_node_id] = root_node_info

    def init_dataset(self, node):
        children = node.getchildren()
        for inx, child in enumerate(children):
            node_info = Node(child, self.page_director, self.offset_x, self.offset_y)
            node_id = node_info.id
            node_info.seq = inx
            self.dataset[node_id] = node_info
            self.init_dataset(child)

    def set_property_to_node_inter_info(self, node_id, property_name, value):
        """
        according to the given key, set information for an node inside result memo
        :param node_id:
        :param property_name:
        :param value:
        """
        if node_id not in self.result:
            self.result[node_id] = {}
        self.result[node_id][property_name] = value

    def get_property_to_node_inter_info(self, node_id, property_name):
        """
        get information of an node from result memo
        :param node_id:
        :param property_name:
        :return:
        """
        if node_id not in self.result:
            return None
        if property_name not in self.result[node_id]:
            return None
        return self.result[node_id][property_name]

    def id_identifier(self, root, parent_helper_ids, parent_helper_id_indices):
        """
        identify helper id of each node inside xml, if there is one.
        if helper id exists, compute its order relative to its siblings with the same helper id, then store
        the order inside result memo
        :param root:
        :param parent_helper_ids:
        :param parent_helper_id_indices:
        """
        children = root.getchildren()
        id_dict = {}
        id_count = {}
        for child in children:
            if "attributes_helperID" in child.attrib:
                child_helper_id = child.attrib["attributes_helperID"]
                if child_helper_id not in id_dict:
                    id_dict[child_helper_id] = 1
                else:
                    id_dict[child_helper_id] += 1
        for helper_id in id_dict:
            id_count[helper_id] = 1
        for child in children:
            if "attributes_helperID" in child.attrib:
                child_helper_id = child.attrib["attributes_helperID"]
                node_info = Node(xml_node=child,
                                 director_path=self.page_director)
                child_id = node_info.id
                index = id_count[child_helper_id]
                child_helper_id_seq = parent_helper_ids + [child_id]
                child_helper_id_index_seq = parent_helper_id_indices + [index]
                if "text_mText" in child.attrib:
                    self.set_property_to_node_inter_info(child_id, "text", child.attrib["text_mText"])
                else:
                    self.set_property_to_node_inter_info(child_id, "text", None)
                self.set_property_to_node_inter_info(child_id, "helper_id_seq", child_helper_id_seq)
                self.set_property_to_node_inter_info(child_id, "xpath", child.getroottree().getpath(child))
                self.set_property_to_node_inter_info(child_id, "helper_id_seq", child_helper_id_index_seq)
                self.id_identifier(child, child_helper_id_seq, child_helper_id_index_seq)
                id_count[child_helper_id] += 1
            else:
                self.id_identifier(child, parent_helper_ids, parent_helper_id_indices)

    def get_dominate_color(self, rendered_layer, pre_rendered_layer, background_color_set):
        """
        compute the background color of this child,
        :param rendered_layer: the rendered child layer
        :param pre_rendered_layer: the child layer before rendering
        :param background_color_set: the background color of this child
        :return:
        """
        # build mask for indicate the content inside child
        mask = np.where(pre_rendered_layer.img_array[:, :, -1] > 0, True, False)
        # build mask for indicate the content inside rendered-child
        rendered_mask = np.where(rendered_layer.img_array[:, :, -1] > 0, True, False)
        # build mask for indicate the blank inside rendered-child
        # the blank part is False
        blank_mask = np.bitwise_xor(mask, rendered_mask)
        blank_mask = np.bitwise_or(blank_mask, mask)
        cropped = rendered_layer.img_array
        color_set = []
        mask_count = np.sum(mask)
        # if the content inside child is itself, return none
        if mask_count == mask.shape[0] * mask.shape[1]:
            return None
        for i in range(3):
            layer_cropped = cropped[:, :, i]
            # replace the blank part with background color
            layer_cropped = np.where(blank_mask == False, background_color_set[i], layer_cropped)
            layer = np.ma.array(layer_cropped, mask=mask)
            # compute the mean color
            color = int(layer.mean())
            color_set.append(color)
        if np.array_equal(np.array(color_set), np.array([0, 0, 0])):
            return None
        color_set.append(255)
        return np.array(color_set)

    def check_existance(self, background_id, parent_image_id, child_ids):
        background_node_info = self.dataset[background_id]
        parent_node_info = self.dataset[parent_image_id]
        component_images_paths = [
            self.dataset[component_id].png_path for component_id in child_ids
        ]

        if parent_node_info.width <= 1 or parent_node_info.height <= 1:
            return []
        background_layer = Layer(background_node_info.x,
                                 background_node_info.y,
                                 background_node_info.width,
                                 background_node_info.height,
                                 img_array=cv2.imread(background_node_info.layer_png_path, cv2.IMREAD_UNCHANGED)
                                 )

        parent_layer, parent_rect = parent_node_info.compute_layer_inside_rect(parent_node_info.get_rect())
        if parent_rect is None:
            return []

        background_color_set = []
        background_color_set_mask = np.where(background_layer.img_array[:, :, -1] > 0, 0, 1)
        for i in range(3):
            layer = np.ma.array(background_layer.img_array[:, :, i], mask=background_color_set_mask)
            if layer.mean() is ma.masked:
                color = 0
            else:
                color = int(layer.mean())
            background_color_set.append(color)
        background_color_set.append(255)
        new_background_img = [background_color_set for _ in range(parent_layer.width * parent_layer.height)]
        new_background_img = np.array(new_background_img, dtype=np.uint8)
        new_background_img = np.reshape(new_background_img,
                                        (parent_layer.height, parent_layer.width, 4))
        background_layer = Layer(parent_layer.x,
                                 parent_layer.y,
                                 parent_layer.width,
                                 parent_layer.height,
                                 img_array=new_background_img
                                 )
        target_layers = [background_layer, parent_layer]
        drawn_target_layer = draw_according_to_rect(target_layers, parent_rect)
        target = drawn_target_layer.img_array
        target_rectangle = drawn_target_layer.get_rectangle()
        existed_children_ids_list = []
        for index in range(len(component_images_paths)):
            corresponding_node = self.dataset[child_ids[index]]
            if corresponding_node.width * corresponding_node.height <= 1:
                print(f"this component {child_ids[index]} is not distinguishable")
                self.set_property_to_node_inter_info(child_ids[index], "invisible", True)
                continue
            component_layer, component_rect = corresponding_node.compute_layer_inside_rect(parent_node_info.get_rect())
            if component_rect is None:
                continue
            component_rendered_layer, component_rendered_rect = parent_node_info.compute_layer_inside_rect(
                component_rect)
            color_ = self.get_dominate_color(component_rendered_layer, component_layer, background_color_set)
            if color_ is None:
                color_ = background_color_set
            new_background_img = [color_ for _ in range(background_node_info.width * background_node_info.height)]
            new_background_img = np.array(new_background_img, dtype=np.uint8)
            new_background_img = np.reshape(new_background_img,
                                            (background_node_info.height, background_node_info.width, 4))
            new_background_layer = Layer(component_layer.x,
                                         component_layer.y,
                                         component_layer.width,
                                         component_layer.height,
                                         img_array=new_background_img
                                         )
            layers = [new_background_layer, component_layer]

            template_layer = draw_according_to_rect(layers, component_rect)

            tpl = template_layer.img_array
            same = True
            for i in range(4):
                layer = tpl[:, :, i]
                if layer.max() != layer.min():
                    same = False
                    break
            if same:
                print(f"this component {child_ids[index]} is the same with its background")
                self.set_property_to_node_inter_info(child_ids[index], "invisible", True)
                continue

            template_rect = template_layer.get_rectangle()
            tpl_in_target_layer = Layer.compute_layer_inside_rect(drawn_target_layer, template_rect)
            small_target = tpl_in_target_layer.img_array
            relative_rect = template_rect.get_relative_rectangle(target_rectangle)

            methods = [
                cv2.TM_CCOEFF_NORMED
            ]

            length_of_method = len(methods)
            fontsize = 6
            # plt.figure()
            ax1 = plt.subplot(1, 3 + length_of_method, 1)
            ax1.imshow(tpl)
            ax1.set_title("template image", fontsize=fontsize)
            ax2 = plt.subplot(1, 3 + length_of_method, 2)
            ax2.imshow(target)
            ax2.set_title("target image", fontsize=fontsize)
            ax3 = plt.subplot(1, 3 + length_of_method, 3)
            ax3.imshow(small_target)
            ax3.set_title("cropped image", fontsize=fontsize)

            titles = {
                cv2.TM_SQDIFF: "TM_SQDIFF",
                cv2.TM_SQDIFF_NORMED: "TM_SQDIFF_NORMED",
                cv2.TM_CCORR: "TM_CCORR",
                cv2.TM_CCORR_NORMED: "TM_CCORR_NORMED",
                cv2.TM_CCOEFF: "TM_CCOEFF",
                cv2.TM_CCOEFF_NORMED: "TM_CCOEFF_NORMED",
            }
            is_match = True
            for inx, method in enumerate(methods):
                list_template = [("temp", tpl)]

                hits = MTM.matchTemplates(list_template,
                                          small_target,
                                          N_object=1,
                                          method=method,
                                          maxOverlap=0
                                          )

                for hit_index, row in hits.iterrows():
                    hits.at[hit_index, 'BBox'] = relative_rect.get_size_tuple()
                score_list = hits["Score"].tolist()
                if len(score_list) < 1:
                    print(f"score is nan!!")
                    continue
                score = hits["Score"].tolist()[0]
                match_iou = 1
                scores = hits["Score"].tolist()
                out_image = target

                if score <= 0.3:
                    is_match = False
                    full_target_layer = drawn_target_layer
                    expand_width = False
                    expand_height = False
                    expand_offset_y = 0
                    expand_offset_x = 0
                    if target.shape[0] <= template_layer.img_array.shape[0] * 2:
                        expand_height = True
                    if target.shape[1] <= template_layer.img_array.shape[1] * 2:
                        expand_width = True
                    if expand_height and expand_width:
                        full_target_layer, expand_offset_x, expand_offset_y = full_target_layer.expand_own_region(
                            color_)
                    full_target = full_target_layer.img_array
                    expanded_relative_rect = Rectangle(relative_rect.x1 + expand_offset_x,
                                                       relative_rect.y1 + expand_offset_y,
                                                       relative_rect.x2 + expand_offset_x,
                                                       relative_rect.y2 + expand_offset_y)

                    another_hits = MTM.matchTemplates(list_template,
                                                      full_target,
                                                      N_object=1,
                                                      method=cv2.TM_CCOEFF_NORMED,
                                                      maxOverlap=0
                                                      )

                    out_image = full_target
                    hits = another_hits
                    match_iou = 0
                    bounding_boxes = another_hits["BBox"].tolist()

                    for bounding_box in bounding_boxes:
                        bounding_rectangle = Rectangle(bounding_box[0], bounding_box[1],
                                                       bounding_box[0] + bounding_box[2],
                                                       bounding_box[1] + bounding_box[3])
                        intersect_region_rect = bounding_rectangle & expanded_relative_rect
                        if intersect_region_rect is None:
                            continue
                        intersect_area = intersect_region_rect.area()
                        child_area = expanded_relative_rect.area()
                        result_area = bounding_rectangle.area()
                        iou = intersect_area / (child_area + result_area - intersect_area)
                        if iou > 0.98888888:
                            is_match = True
                            match_iou = iou
                        if is_match:
                            break
                        match_iou = iou

                ax = plt.subplot(1, 3 + len(methods), 3 + inx + 1)
                ax.imshow(out_image)
                for _, row in hits.iterrows():
                    x, y, w, h = row['BBox']
                    rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)

                ax.set_axis_off()
                title = titles[method] + "\n"

                for score in scores:
                    title += str(score)[:4]
                title += "\nIs a match: " + str(is_match)
                title += "\nIoU is: " + str(match_iou)[:5]
                ax.set_title(title, fontsize=fontsize)

            plt.subplots_adjust()
            saved_dir_path = os.path.join(self.sub_director, "img_result")
            found_path = os.path.join(saved_dir_path, "found")
            not_found_path = os.path.join(saved_dir_path, "notfound")
            if not os.path.exists(saved_dir_path):
                os.mkdir(saved_dir_path)
            if not os.path.exists(found_path):
                os.mkdir(found_path)
            if not os.path.exists(not_found_path):
                os.mkdir(not_found_path)

            if is_match:
                # self.set_property_to_node_intra_info(child_ids[index], "invisible", False)
                existed_children_ids_list.append(child_ids[index])
                saved_img_path = os.path.join(found_path, str(child_ids[index]) + "result.png")
                plt.savefig(saved_img_path, dpi=600)
            else:
                self.set_property_to_node_inter_info(child_ids[index], "invisible", True)
                print(f"{child_ids[index]} is not visible")
                saved_img_path = os.path.join(not_found_path, str(child_ids[index]) + "result.png")
                plt.savefig(saved_img_path, dpi=600)
            # plt.show()
            plt.clf()
            # plt.close('all')
        return existed_children_ids_list

    def check_clipped_area(self, child_content_area_region_rect, parent_visible_region_rect):
        """
        check if child is clipped by its parent
        :param child_content_area_region_rect:
        :param parent_visible_region_rect:
        :return:
            child_is_invisible: indicate if the child is blocked completely
            child_is_partly_visible: blocked partly
            child_visible_tuple:(child_visible_area, child_is_clipped_horizontally, child_is_clipped_vertically)
        """
        child_content_rect = child_content_area_region_rect.rect
        if parent_visible_region_rect is None:
            return True, False, None
        parent_visible_rect = parent_visible_region_rect.rect
        intersect_rect = child_content_rect & parent_visible_rect
        # check if the child is inside parent
        if intersect_rect is None:
            return True, False, None
        # compute the information area of child
        # whole_area_size = np.sum(np.where(child_content_area_region_rect.visible_region == True, 1, 0))
        # compute the visible region of this child inside its its parent
        visible_region = child_content_area_region_rect.get_visible_region(parent_visible_region_rect)
        # compute the information area of visible region of this child inside its its parent
        visible_area_size = np.sum(np.where(visible_region == True, 1, 0))
        # if the area of child is larger than the actual display size,
        # then some parts of this children could not be seen definitely
        whole_height, whole_width = RegionRect.compute_height_and_width(
            np.where(child_content_area_region_rect.visible_region == True, 1, 0))
        visible_height, visible_width = RegionRect.compute_height_and_width(visible_region)

        if visible_width == 0 or visible_height == 0:
            return True, False, None

        visible_tuple = [visible_area_size, False, False]
        if whole_width > visible_width:
            visible_tuple[1] = True
        if whole_height > visible_height:
            visible_tuple[2] = True
        if visible_tuple[1] == True or visible_tuple[2] == True:
            return False, True, visible_tuple
        return False, False, (visible_area_size, False, False)

    def order_child_by_z_index_and_seq_index(self, child_list):
        result_list = []
        ordered_dict = {}
        def order_function(key):
            item = self.dataset[key]
            return item.seq
        for child_id in child_list:
            z = self.dataset[child_id].z
            if z in ordered_dict:
                ordered_dict[z].append(child_id)
            else:
                ordered_dict[z] = [child_id]
        z_order_list = list(ordered_dict.keys())
        z_order_list = sorted(z_order_list, reverse=True)
        for z_key in z_order_list:
            result_list.extend(sorted(ordered_dict[z_key], key=order_function, reverse=True))
        return result_list

    def inter_view_analysis(self):
        rm_tree(self.sub_director)
        pathlib.Path(self.sub_director).mkdir(parents=True, exist_ok=True)
        root_id = self.root_node_info.id
        start_time = time.time()
        self.inter_analysis1(root_id, None, root_id)
        end_time = time.time()
        result = json.dumps(self.result, indent="\t")
        result_path = os.path.join(self.sub_director, "intra_analysis_output.json")
        with io.open(result_path, "w", encoding="UTF-8") as f:
            f.write(result)
        time_path = os.path.join(self.sub_director, "intra_analysis_output_time.json")
        with io.open(time_path, "w", encoding="UTF-8") as f:
            f.write(json.dumps({"time": end_time - start_time}, indent="\t"))
        return self.result

    def crop_scrollable_out(self, node):
        content = node.compute_content_area()
        def crop_scrollable_node_out(node, contents):
            for child in node.getchildren():
                child_id = child.attrib["id"]
                if Node.check_node_scrollable(child):
                    existed_child_node_info = self.dataset[child_id]
                    child_content_area_region_rect = existed_child_node_info.compute_full_area()
                    contents = RegionRect.compute_visible_subtract(contents, child_content_area_region_rect)
                else:
                    contents = crop_scrollable_node_out(child, contents)
            return contents
        content = crop_scrollable_node_out(node.node, content)
        return content

    def inter_analysis1(self, tree_node_id,
                        parent_visible_region_rect,
                        background_node_id,
                        can_scroll_horizontally=False,
                        can_scroll_vertically=False):
        tree_node_info = self.dataset[tree_node_id]
        visible_region_rect = tree_node_info.compute_content_area()
        if can_scroll_horizontally or can_scroll_vertically:
            self.set_property_to_node_inter_info(tree_node_id, "scrollable", True)
            parent_visible_region_rect = None
        # check if this canvas is bigger than parent, if so,
        # then check if it is scrollable, if not, then clip this node
        if parent_visible_region_rect is not None and \
                not (can_scroll_horizontally or can_scroll_vertically):
            visible_region_rect = RegionRect.compute_visible_clipping(visible_region_rect, parent_visible_region_rect)
        # get the children of this layer and compute their visibility
        children_ids_list = tree_node_info.get_ids_of_children()
        if len(children_ids_list) > 0:
            exist_child_ids_list = self.check_existance(background_node_id, tree_node_id, children_ids_list)
        else:
            exist_child_ids_list = []
        this_content_area_region_rect = tree_node_info.compute_content_area()

        for existed_child_id in exist_child_ids_list:
            existed_child_node_info = self.dataset[existed_child_id]
            child_content_area_region_rect = existed_child_node_info.compute_content_area()
            this_content_area_region_rect = RegionRect.compute_visible_subtract(this_content_area_region_rect,
                                                                                child_content_area_region_rect)

        if np.sum(this_content_area_region_rect.visible_region) > 0:
            child_is_invisible, child_is_partly_visible, child_visible_tuple = \
                self.check_clipped_area(tree_node_info.compute_content_area(), visible_region_rect)
            if child_is_invisible:
                print(f"{tree_node_id} is invisible for its parent.")
                ax1 = plt.subplot(1, 2, 1)
                ax1.imshow(
                    np.where(tree_node_info.compute_content_area().visible_region == True, 255, 0).astype(np.uint8),
                    cmap='gray', vmin=0, vmax=255)
                ax1.set_title(f"invisible child\n{tree_node_id}", fontsize=6)
                if visible_region_rect is not None:
                    ax1 = plt.subplot(1, 2, 2)
                    ax1.imshow(np.where(visible_region_rect.visible_region == True, 255, 0).astype(np.uint8),
                               cmap='gray', vmin=0, vmax=255)
                    ax1.set_title(f"its parent", fontsize=6)
                saved_dir = os.path.join(self.sub_director, "invisible")
                if not os.path.exists(saved_dir):
                    pathlib.Path(saved_dir).mkdir(parents=True, exist_ok=True)
                saved_img_path = os.path.join(saved_dir, f"{tree_node_id}_result.png")
                plt.savefig(saved_img_path, dpi=600)
                plt.clf()
                self.set_property_to_node_inter_info(tree_node_id, "invisible", True)
            elif child_is_partly_visible:
                ax1 = plt.subplot(1, 2, 1)
                ax1.imshow(
                    np.where(tree_node_info.compute_content_area().visible_region == True, 255, 0).astype(np.uint8),
                    cmap='gray', vmin=0, vmax=255)
                ax1.set_title(f"partly visible child\n{tree_node_id}", fontsize=6)
                ax1 = plt.subplot(1, 2, 2)
                ax1.imshow(np.where(visible_region_rect.visible_region == True, 255, 0).astype(np.uint8), cmap='gray',
                           vmin=0, vmax=255)
                ax1.set_title(f"its parent", fontsize=6)
                saved_dir = os.path.join(self.sub_director, "clipped")
                if not os.path.exists(saved_dir):
                    pathlib.Path(saved_dir).mkdir(parents=True, exist_ok=True)
                saved_img_path = os.path.join(saved_dir, f"{tree_node_id}_result.png")
                plt.savefig(saved_img_path, dpi=600)
                plt.clf()
                self.set_property_to_node_inter_info(tree_node_id, "clipped", True)
                print(f"{tree_node_id} is partly visible for its parent.")
            else:
                self.set_property_to_node_inter_info(tree_node_id, "fully_visible", True)
        ordered_child_list = self.order_child_by_z_index_and_seq_index(exist_child_ids_list)
        sum_shadow_area = visible_region_rect
        shadow_siblings = []
        for child_id in ordered_child_list:
            overlapped_siblings = []
            overlapped_siblings_id = []
            self.set_property_to_node_inter_info(child_id, "overlapped_siblings", overlapped_siblings)
            child_node_info = self.dataset[child_id]

            child_content_area_region_rect = self.crop_scrollable_out(child_node_info)
            if not child_node_info.is_drawer:
                shadow_siblings.append(child_node_info)
            for sibling in shadow_siblings[:-1]:
                sibling_region_rect = self.crop_scrollable_out(sibling)
                is_overlapped = child_content_area_region_rect.check_if_overlapped_with(sibling_region_rect)
                if is_overlapped:

                    overlapped_siblings.append(sibling.helper_id)
                    overlapped_siblings_id.append(sibling.id)
                    print(f"{child_id} is overlapped with its sibling: {sibling.id}")
            new_background_node_id = background_node_id
            if child_node_info.has_layer and (not child_node_info.background_transparency):
                new_background_node_id = child_id
            can_scroll_vertically = tree_node_info.can_scroll_vertically
            can_scroll_horizontally = tree_node_info.can_scroll_horizontally
            self.inter_analysis1(child_id, sum_shadow_area, new_background_node_id, can_scroll_horizontally,
                                 can_scroll_vertically)
            self.draw_overlapping_nodes(child_id, overlapped_siblings_id)
            self.set_property_to_node_inter_info(child_id, "overlapped_siblings_id", overlapped_siblings_id)
            self.set_property_to_node_inter_info(child_id, "overlapped_siblings", overlapped_siblings)
            if not child_node_info.is_drawer:
                sum_shadow_area = RegionRect.compute_visible_subintersect(sum_shadow_area,
                                                                          child_content_area_region_rect)

    def draw_overlapping_nodes(self, child_id, overlapping_ids):
        saved_dir_path = os.path.join(self.sub_director, "overlapping")
        if not os.path.exists(saved_dir_path):
            os.mkdir(saved_dir_path)
        length_of_overlapping_ids = len(overlapping_ids)
        if length_of_overlapping_ids < 1:
            return
        max_column = 3
        row = math.ceil(length_of_overlapping_ids / 3)
        ax1 = plt.subplot(row, max_column, 1)
        child_node_info = self.dataset[child_id]
        ax1.imshow(cv2.imread(child_node_info.png_path, cv2.IMREAD_UNCHANGED))
        ax1.set_title(f"blocked_image\n{child_node_info.id}", fontsize=6)
        for inx, overlapping_id in enumerate(overlapping_ids):
            overlapping_sibling = self.dataset[overlapping_id]
            ax2 = plt.subplot(row, max_column, 1 + inx + 1)
            ax2.imshow(cv2.imread(overlapping_sibling.png_path, cv2.IMREAD_UNCHANGED))
            ax2.set_title(f"blocking_sibling\n{overlapping_sibling.id}", fontsize=6)
        saved_img_path = os.path.join(saved_dir_path, f"{child_id}_result.png")
        plt.savefig(saved_img_path, dpi=600)
        plt.clf()

    def draw_clipped_nodes(self, child_id, parent_id):
        saved_dir_path = os.path.join(self.sub_director, "clipping")
        if not os.path.exists(saved_dir_path):
            os.mkdir(saved_dir_path)
        ax1 = plt.subplot(1, 2, 1)
        child_node_info = self.dataset[child_id]
        ax1.imshow(cv2.imread(child_node_info.png_path, cv2.IMREAD_UNCHANGED))
        ax1.set_title(f"blocked_image\n{child_node_info.id}", fontsize=6)
        parent_node_info = self.dataset[parent_id]
        ax1 = plt.subplot(1, 2, 2)
        ax1.imshow(cv2.imread(parent_node_info.png_path, cv2.IMREAD_UNCHANGED))
        ax1.set_title(f"blocking_sibling\n{parent_node_info.id}", fontsize=6)
        saved_img_path = os.path.join(saved_dir_path, f"{child_id}_result.png")
        plt.savefig(saved_img_path, dpi=600)
        plt.clf()


def dumper(obj):
    try:
        return obj.toJSON()
    except:
        return obj.__dict__


def compute_intra_feature(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    alpha_layer = image[:, :, -1:]
    rgb_layer = image[:, :, :-1]
    gray_layer = skimage.rgb2gray(rgb_layer)
    if np.max(gray_layer) < 1:
        analysis_layer = skimage.img_as_float(alpha_layer)
    else:
        analysis_layer = gray_layer
    nums = []
    for threshold in reversed(np.linspace(0, 1, 100)):
        binarized = analysis_layer < threshold
        all_labels, num = measure.label(binarized, return_num=True)
        nums.append(num)
        print(f"Threshold is: {threshold}, num is:{num}")
    c = collections.Counter(nums)
    print(c)
    return c.keys()[0]


def compute_intra_feature_of_text(image_node_info):
    image = cv2.imread(image_node_info.png_path, cv2.IMREAD_UNCHANGED)

    node = image_node_info.node
    if "text_getTextSize" in node.attrib:
        textSize = float(node.attrib["text_getTextSize"])
    else:
        textSize = -1

    alpha_layer = image[:, :, -1:]
    alpha_layer = np.reshape(alpha_layer, (alpha_layer.shape[0], alpha_layer.shape[1]))
    keep_matrix = np.where(alpha_layer == 0, False, True)
    alpha_layer_width = np.sum(np.where(np.sum(alpha_layer, axis=0) > 0, 1, 0))
    alpha_layer_height = np.sum(np.where(np.sum(alpha_layer, axis=1) > 0, 1, 0))
    area_alpha_layer = alpha_layer_width * alpha_layer_height
    bg = False
    img = alpha_layer

    if area_alpha_layer * 0.9 < np.sum(keep_matrix):
        print("this contains bg! do my best to eliminate the background and get its text inside!")
        img = cv2.imread(image_node_info.png_path, 0)
        img = 255 - img
        sub_color = 0
        img = np.where(alpha_layer == 0, 365, img)
        c = collections.Counter(img.reshape((-1)))
        most_common = c.most_common(10)
        for pixel_color, count in most_common:
            if pixel_color == 365:
                continue
            sub_color = pixel_color
            break
        img = np.where(img == 365, sub_color, img).astype(np.uint8)

    _, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    th2_inverse = 255 - th2

    th2, th2_inverse, bg = get_binary_image_of_text(image_node_info.png_path)

    def remove_noise(image_th):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_th, connectivity=8)
        th_areas_width = stats[1:, 2].reshape((-1))
        th_areas_height = stats[1:, 3].reshape((-1))
        th_areas = stats[1:, -1].reshape((-1))

        # th_areas = th_areas_width*th_areas_height

        th_area_sum = np.sum(th_areas)
        if len(stats) > 1:
            filter_th_width = th2.shape[1]
            filter_th_height = np.min(th_areas_height) / 2
            ratio_filter = filter_th_width / filter_th_height
            th_ratio = th_areas_width / th_areas_height
            th_areas = stats[1:, -1].reshape((-1))
            th_areas = np.where((th_ratio > 50), -1, th_areas)
            th_areas = th_areas[th_areas != -1]
            th_area_sum = np.sum(th_areas)
        return th_areas, th_area_sum

    th2_areas, th2_area_sum = remove_noise(th2)
    th2_inverse_areas, th2_inverse_area_sum = remove_noise(th2_inverse)

    return {
        "bg": bg,
        "area": th2_area_sum,
        "inverse_area": th2_inverse_area_sum,
        "rgb_layer": img_as_ubyte(img),
        "areas": th2_areas,
        "inverse_areas": th2_inverse_areas,
        "ellipsize": image_node_info.ellipsize,
        "text_size": textSize
    }


def compute_intra_feature_of_image(image_node_info):
    image = cv2.imread(image_node_info.png_path, cv2.IMREAD_UNCHANGED)

    node = image_node_info.node
    if "text_getTextSize" in node.attrib:
        textSize = float(node.attrib["text_getTextSize"])
    else:
        textSize = -1

    alpha_layer = image[:, :, -1:]
    alpha_layer = np.reshape(alpha_layer, (alpha_layer.shape[0], alpha_layer.shape[1]))
    keep_matrix = np.where(alpha_layer == 0, False, True)
    alpha_layer_width = np.sum(np.where(np.sum(alpha_layer, axis=0) > 0, 1, 0))
    alpha_layer_height = np.sum(np.where(np.sum(alpha_layer, axis=1) > 0, 1, 0))
    area_alpha_layer = alpha_layer_width * alpha_layer_height
    bg = False
    img = alpha_layer

    if area_alpha_layer * 0.9 < np.sum(keep_matrix):
        print("this contains bg! do my best to eliminate the background and get its text inside!")
        img = cv2.imread(image_node_info.png_path, 0)
        img = 255 - img
        sub_color = 0
        img = np.where(alpha_layer == 0, 365, img)
        c = collections.Counter(img.reshape((-1)))
        most_common = c.most_common(10)
        for pixel_color, count in most_common:
            if pixel_color == 365:
                continue
            sub_color = pixel_color
            break
        img = np.where(img == 365, sub_color, img).astype(np.uint8)

    _, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    th2_inverse = 255 - th2

    def remove_noise(image_th):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_th, connectivity=8)
        th_areas_width = stats[1:, 2].reshape((-1))
        th_areas_height = stats[1:, 3].reshape((-1))
        th_areas = stats[1:, -1].reshape((-1))

        # th_areas = th_areas_width*th_areas_height

        th_area_sum = np.sum(th_areas)
        if len(stats) > 1:
            filter_th_width = th2.shape[1]
            filter_th_height = np.min(th_areas_height) / 2
            ratio_filter = filter_th_width / filter_th_height
            th_ratio = th_areas_width / th_areas_height
            th_areas = stats[1:, -1].reshape((-1))
            th_areas = np.where((th_ratio > 50), -1, th_areas)
            th_areas = th_areas[th_areas != -1]
            th_area_sum = np.sum(th_areas)
        return th_areas, th_area_sum

    th2_areas, th2_area_sum = remove_noise(th2)
    th2_inverse_areas, th2_inverse_area_sum = remove_noise(th2_inverse)

    return {
        "bg": bg,
        "area": th2_area_sum,
        "inverse_area": th2_inverse_area_sum,
        "rgb_layer": img_as_ubyte(img),
        "areas": th2_areas,
        "inverse_areas": th2_inverse_areas,
        "ellipsize": image_node_info.ellipsize,
        "text_size": textSize
    }


def get_image(path_of_image):
    image = cv2.imread(path_of_image, cv2.IMREAD_UNCHANGED)
    rgb_layer = image[:, :, :-1]
    alpha_layer = image[:, :, -1:]
    gray_layer = rgb2gray(rgb_layer)
    if np.max(gray_layer) == 0:
        alpha_layer = np.reshape(alpha_layer, (alpha_layer.shape[0], alpha_layer.shape[1]))
        analysis_layer = img_as_float(alpha_layer)
    else:
        analysis_layer = gray_layer
    return analysis_layer


def rm_tree(pth):
    pth = pathlib.Path(pth)
    if pth.exists():
        for child in pth.glob('*'):
            if child.is_file():
                child.unlink()
            else:
                rm_tree(child)
        pth.rmdir()


def compute_ratio_of_two_areas(bigger, normal):
    normal_sorted = sorted(normal)
    bigger_sorted = sorted(bigger)

    if len(normal_sorted) == len(bigger_sorted):
        normal_var = []
        bigger_var = []

        for inx, value in enumerate(normal_sorted):
            if 0.95 < bigger_sorted[inx] / normal_sorted[inx] < 1.05:
                pass
            else:
                normal_var.append(normal_sorted[inx])
                bigger_var.append(bigger_sorted[inx])
        a_bigger_sum = np.sum(np.array(bigger_var))
        a_middle_sum = np.sum(np.array(normal_var))
    else:
        a_middle_counter = collections.Counter(normal)
        a_bigger_counter = collections.Counter(bigger)

        print(a_bigger_counter & a_middle_counter)
        common = a_bigger_counter & a_middle_counter
        a_middle_counter -= common
        a_bigger_counter -= common
        a_middle_sum = 0
        for item in a_middle_counter:
            a_middle_sum += item * a_middle_counter[item]

        a_bigger_sum = 0
        for item in a_bigger_counter:
            a_bigger_sum += item * a_bigger_counter[item]

    if a_bigger_sum == 0 and a_middle_sum == 0:
        ratio = 1
    elif a_bigger_sum != 0 and a_middle_sum == 0 or a_bigger_sum == 0 and a_middle_sum != 0:
        ratio = 100
    else:
        ratio = a_bigger_sum / a_middle_sum
    return ratio


def smallestbox(a):
    r = a.any(1)
    if r.any():
        m, n = a.shape
        c = a.any(0)
        return r.argmax(), m - r[::-1].argmax(), c.argmax(), n - c[::-1].argmax()
    else:
        return 0, 0, 0, 0


def compute_ssim(normal_image_path, bigger_image_path):
    image_normal = cv2.imread(normal_image_path, cv2.IMREAD_UNCHANGED)
    image_bigger = cv2.imread(bigger_image_path, cv2.IMREAD_UNCHANGED)

    if np.max(image_normal[:, :, 3].reshape((-1))) == 0:
        image_normal = image_normal[:, :, 3:]
        image_bigger = image_bigger[:, :, 3:]
        channel = 1
    else:
        image_normal = image_normal[:, :, :3]
        image_bigger = image_bigger[:, :, :3]
        channel = 3

    if image_normal.shape[0] <= 1 or image_bigger.shape[1] <= 1:
        return None
    if image_normal.shape[0] <= 7 or image_bigger.shape[1] <= 7:
        return None

    image_bigger_another = resize(image_bigger, (image_normal.shape[0], image_normal.shape[1], channel),
                                  anti_aliasing=True)

    image_normal = img_as_ubyte(image_normal)
    image_bigger_another = img_as_ubyte(image_bigger_another)
    ssim_another = ssim(image_normal, image_bigger_another, channel_axis=2)

    if ssim_another > 0.9:
        return ssim_another

    if channel == 3:
        gray_channel = rgb2gray(image_normal)
        y1, h, x1, w = smallestbox(gray_channel)
        image_normal_crop = image_normal[y1:h, x1:w, :]
        gray_channel = rgb2gray(image_bigger)
        y1, h, x1, w = smallestbox(gray_channel)
        image_bigger_crop = image_bigger[y1:h, x1:w, :]
    else:
        y1, h, x1, w = smallestbox(image_normal)
        image_normal_crop = image_normal[y1:h, x1:w, :]
        y1, h, x1, w = smallestbox(image_bigger)
        image_bigger_crop = image_bigger[y1:h, x1:w, :]

    if image_bigger_crop.shape[0] <= 1 or image_bigger_crop.shape[1] <= 1:
        return None
    if image_normal_crop.shape[0] <= 7 or image_normal_crop.shape[1] <= 7:
        return None

    image_resized_crop = resize(image_bigger_crop, (image_normal_crop.shape[0], image_normal_crop.shape[1], channel),
                                anti_aliasing=True)
    image_resized_crop = img_as_ubyte(image_resized_crop)
    image_normal = img_as_ubyte(image_normal_crop)
    ssim_noise = ssim(image_resized_crop, image_normal, channel_axis=2)
    print(ssim_noise)
    return ssim_noise


def intra_view_anlysis(normal_tree, bigger_tree, pairs, result_dir, normal_activity_inter_result, standard_ratio,
                       buggy_view_ids):
    """

    :param normal_tree: the IntraAnalyzer of normal page
    :param bigger_tree: the IntraAnalyzer of bigger page
    :param pairs:
    :param result_dir:
    :param normal_activity_inter_result:
    :param standard_ratio: the scaling ratio of normal texts between bigger texts
    :return:
    """
    result = {
        "consistency": [],
        "inconsistency": []
    }
    pairs = json.loads(json.dumps(pairs, default=dumper))
    text_dir = os.path.join(result_dir, "text")
    ssim_dir = os.path.join(result_dir, "ssim")
    rm_tree(text_dir)
    rm_tree(ssim_dir)
    for key in pairs:
        if pairs[key]["normal"] is None:
            continue
        normal_view_id = pairs[key]["normal"]["view_id"]
        if normal_view_id not in normal_activity_inter_result:
            continue

        if pairs[key]["bigger"] is None:
            result["inconsistency"].append(
                {
                    "normal_view_id": pairs[key]["normal"]["view_id"],
                    "type": "not found inconsistency"
                }
            )
            continue

        if pairs[key]["normal"]["is_leaf"] == False or pairs[key]["bigger"]["is_leaf"] == False:
            continue

        normal_node_result = normal_activity_inter_result[normal_view_id]
        normal_node_invisible = normal_node_result["invisible"] if "invisible" in normal_node_result else False
        if normal_node_invisible is True:
            continue
        normal_node_info = normal_tree.dataset[normal_view_id]
        normal_image_path = normal_node_info.png_path
        bigger_view_id = pairs[key]["bigger"]["view_id"]

        if normal_view_id in buggy_view_ids:
            result["inconsistency"].append(
                {
                    "normal_view_id": normal_view_id,
                    "bigger_view_id": bigger_view_id,
                    "type": "oracle inconsistency"
                }
            )

        bigger_node_info = bigger_tree.dataset[bigger_view_id]
        bigger_image_path = bigger_node_info.png_path
        if "text_mText" in bigger_node_info.node.attrib:
            if normal_view_id == "31be7ff" and bigger_view_id == "4619a3e":
                normal_view_id = "31be7ff"
            normal_image = compute_intra_feature_of_text(normal_node_info)
            bigger_image = compute_intra_feature_of_text(bigger_node_info)
            ratio = compute_ratio_of_two_areas(bigger_image["areas"], normal_image["areas"])
            ratio_inv = compute_ratio_of_two_areas(bigger_image["inverse_areas"], normal_image["inverse_areas"])

            ratio_cross = compute_ratio_of_two_areas(bigger_image["areas"], normal_image["inverse_areas"])
            ratio_cross_inv = compute_ratio_of_two_areas(bigger_image["inverse_areas"], normal_image["areas"])

            text_standard_ratio = standard_ratio
            if bigger_image["text_size"] != -1 and normal_image["text_size"] != -1:
                text_standard_ratio = (bigger_image["text_size"] / normal_image["text_size"]) * \
                                      (bigger_image["text_size"] / normal_image["text_size"])

            if text_standard_ratio - 0.15 <= ratio <= text_standard_ratio + 0.3 or 1 == ratio or \
                    ratio_inv == 1 or text_standard_ratio - 0.15 <= ratio_inv <= text_standard_ratio + 0.3 or \
                    ratio_cross == 1 or text_standard_ratio - 0.15 <= ratio_cross <= text_standard_ratio + 0.3 or \
                    ratio_cross_inv == 1 or text_standard_ratio - 0.15 <= ratio_cross_inv <= text_standard_ratio + 0.3:
                print("text area consistency detected!")

                result["consistency"].append(
                    {
                        "normal_view_id": normal_view_id,
                        "bigger_view_id": bigger_view_id,
                        "type": "text area consistency",
                        "standard_ratio": standard_ratio,
                        "ratio": ratio,
                        "bigger areas": str(bigger_image['areas']),
                        "normal areas": str(normal_image['areas']),
                    }
                )
                continue
            if bigger_image["ellipsize"] == True or normal_image["ellipsize"] == True:
                result["consistency"].append(
                    {
                        "normal_view_id": normal_view_id,
                        "bigger_view_id": bigger_view_id,
                        "type": "text ellipsized consistency",
                        "ratio": ratio,
                        "bigger areas": str(bigger_image['areas']),
                        "normal areas": str(normal_image['areas']),
                    }
                )
                continue
            pathlib.Path(text_dir).mkdir(parents=True, exist_ok=True)
            ax1 = plt.subplot(1, 2, 1)
            ax1.imshow(cv2.imread(normal_image_path, cv2.IMREAD_UNCHANGED))
            ax1.set_title(f"normal image\n{normal_view_id}", fontsize=6)
            ax2 = plt.subplot(1, 2, 2)
            ax2.imshow(cv2.imread(bigger_image_path, cv2.IMREAD_UNCHANGED))
            ax2.set_title(f"bigger sibling\n{bigger_view_id}", fontsize=6)
            saved_img_path = os.path.join(text_dir, f"text_{normal_view_id}_{bigger_view_id}_{ratio}_result.png")
            plt.savefig(saved_img_path, dpi=600, transparent=True)
            plt.clf()

            result["inconsistency"].append(
                {
                    "normal_view_id": normal_view_id,
                    "bigger_view_id": bigger_view_id,
                    "type": "text inconsistency",
                    "ratio": ratio,
                    "bigger areas": str(bigger_image['areas']),
                    "normal areas": str(normal_image['areas']),
                }
            )

            continue
        else:
            normal_image = compute_intra_feature_of_image(normal_node_info)
            bigger_image = compute_intra_feature_of_image(bigger_node_info)

            connected_part_normal = len(normal_image["areas"])
            connected_part_bigger = len(bigger_image["areas"])

            inv_connected_part_normal = len(normal_image["inverse_areas"])
            inv_connected_part_bigger = len(bigger_image["inverse_areas"])

            if connected_part_bigger == connected_part_normal:
                result["consistency"].append(
                    f"image consistency detected between {normal_view_id} and {bigger_view_id}, "
                    f"connected part is {connected_part_normal}, {connected_part_bigger},"
                    f"inv connected part is {inv_connected_part_normal}, {inv_connected_part_bigger}")
                print("image consistency detected!")
                continue

            image_normal = cv2.imread(normal_image_path, cv2.IMREAD_UNCHANGED)
            image_bigger = cv2.imread(bigger_image_path, cv2.IMREAD_UNCHANGED)

            if np.max(image_normal[:, :, 3].reshape((-1))) == 0:
                image_normal = image_normal[:, :, 3:]
                image_bigger = image_bigger[:, :, 3:]
                channel = 1
            else:
                image_normal = image_normal[:, :, :3]
                image_bigger = image_bigger[:, :, :3]
                channel = 3

            if image_normal.shape[0] <= 1 or image_bigger.shape[1] <= 1:
                continue
            if image_normal.shape[0] <= 7 or image_bigger.shape[1] <= 7:
                continue

            image_bigger_another = resize(image_bigger, (image_normal.shape[0], image_normal.shape[1], channel),
                                          anti_aliasing=True)

            image_normal = img_as_ubyte(image_normal)
            image_bigger_another = img_as_ubyte(image_bigger_another)
            ssim_another = ssim(image_normal, image_bigger_another, channel_axis=2)

            if ssim_another > 0.9:
                continue

            if channel == 3:
                gray_channel = rgb2gray(image_normal)
                y1, h, x1, w = smallestbox(gray_channel)
                image_normal_crop = image_normal[y1:h, x1:w, :]
                gray_channel = rgb2gray(image_bigger)
                y1, h, x1, w = smallestbox(gray_channel)
                image_bigger_crop = image_bigger[y1:h, x1:w, :]
            else:
                y1, h, x1, w = smallestbox(image_normal)
                image_normal_crop = image_normal[y1:h, x1:w, :]
                y1, h, x1, w = smallestbox(image_bigger)
                image_bigger_crop = image_bigger[y1:h, x1:w, :]

            if image_bigger_crop.shape[0] <= 1 or image_bigger_crop.shape[1] <= 1:
                continue
            if image_normal_crop.shape[0] <= 7 or image_normal_crop.shape[1] <= 7:
                continue

            image_resized_crop = resize(image_bigger_crop,
                                        (image_normal_crop.shape[0], image_normal_crop.shape[1], channel),
                                        anti_aliasing=True)
            image_resized_crop = img_as_ubyte(image_resized_crop)
            image_normal = img_as_ubyte(image_normal_crop)
            ssim_noise = ssim(image_resized_crop, image_normal, channel_axis=2)
            print(ssim_noise)

            if ssim_noise < 0.85:
                pathlib.Path(ssim_dir).mkdir(parents=True, exist_ok=True)
                ax1 = plt.subplot(1, 2, 1)
                ax1.imshow(image_normal)
                ax1.set_title(f"normal image\n{normal_view_id}", fontsize=6)
                ax2 = plt.subplot(1, 2, 2)
                ax2.imshow(image_bigger)
                ax2.set_title(f"bigger sibling\n{bigger_view_id}", fontsize=6)
                saved_img_path = os.path.join(ssim_dir,
                                              f"ssim_{normal_view_id}_{bigger_view_id}_result_{ssim_noise}.png")
                plt.savefig(saved_img_path, dpi=600, transparent=True)
                plt.clf()

                result["inconsistency"].append(
                    {
                        "normal_view_id": normal_view_id,
                        "bigger_view_id": bigger_view_id,
                        "type": "ssim inconsistency",
                        "ssim": ssim_noise,
                        'normal_con_parts': connected_part_normal,
                        'bigger_con_parts': connected_part_bigger,
                        'inv_normal_con_parts': inv_connected_part_normal,
                        'inv_bigger_con_parts': inv_connected_part_bigger,
                    }
                )
                print("ssim inconsistency detected!")
            else:
                result["consistency"].append(
                    {
                        "normal_view_id": normal_view_id,
                        "bigger_view_id": bigger_view_id
                    }
                )
    return result


def inter_view_analysis_result(normal_tree, bigger_tree, pairs, result_dir, normal_denoise, bigger_denoise,
                               buggy_view_ids):
    """

    :param normal_tree: inter analysis result of the normal tree
    :param bigger_tree: inter analysis result of the bigger tree
    :param pairs: the mapping result
    :param result_dir: the dir for saving all of the middle and analysis result
    :param normal_denoise: the IntraAnalyzer of normal page
    :param bigger_denoise: the IntraAnalyzer of bigger page
    :return:
    """
    result = {
        "consistency": [],
        "inconsistency": []
    }
    pairs = json.loads(json.dumps(pairs, default=dumper))
    normal_tree = json.loads(json.dumps(normal_tree, default=dumper))
    bigger_tree = json.loads(json.dumps(bigger_tree, default=dumper))
    intra_dir = os.path.join(result_dir)
    rm_tree(intra_dir)
    for key in pairs:
        if pairs[key]["normal"] is None:
            continue
        normal_view_id = pairs[key]["normal"]["view_id"]

        if normal_view_id not in normal_tree:
            continue

        if pairs[key]["bigger"] is None:
            result["inconsistency"].append(
                {
                    "normal_view_id": pairs[key]["normal"]["view_id"],
                    "type": "not found inconsistency"
                }
            )
            continue

        normal_node_info = normal_denoise.dataset[normal_view_id]
        normal_img_path = normal_node_info.png_path
        normal_node_result = normal_tree[normal_view_id]
        bigger_view_id = pairs[key]["bigger"]["view_id"]
        if key.startswith("SubRep"):
            parent_view_id = pairs[key]["bigger"]["parent_view_id"]
            if parent_view_id not in bigger_tree:
                continue
            parent_invisible = bigger_tree[parent_view_id]["invisible"] if "invisible" in bigger_tree[
                parent_view_id] else False
            if parent_invisible:
                continue

        if bigger_view_id not in bigger_tree:
            result["inconsistency"].append(
                {
                    "normal_view_id": normal_view_id,
                    "bigger_view_id": bigger_view_id,
                    "type": "structure inconsistency"
                }
            )
            continue

        if normal_view_id in buggy_view_ids:
            result["inconsistency"].append(
                {
                    "normal_view_id": normal_view_id,
                    "bigger_view_id": bigger_view_id,
                    "type": "ocacle inconsistency"
                }
            )
            continue

        bigger_node_info = bigger_denoise.dataset[bigger_view_id]
        bigger_img_path = bigger_node_info.png_path
        bigger_node_result = bigger_tree[bigger_view_id]

        normal_node_fully_visible = normal_node_result[
            "fully_visible"] if "fully_visible" in normal_node_result else True

        normal_node_invisible = normal_node_result["invisible"] if "invisible" in normal_node_result else False
        normal_node_clipped = normal_node_result["clipped"] if "clipped" in normal_node_result else False
        normal_node_overlapped_siblings = set(
            normal_node_result["overlapped_siblings"]) if "overlapped_siblings" in normal_node_result else {}
        normal_node_overlapped_siblings_id = set(
            normal_node_result["overlapped_siblings_id"]) if "overlapped_siblings_id" in normal_node_result else {}

        bigger_node_fully_visible = bigger_node_result[
            "fully_visible"] if "fully_visible" in bigger_node_result else True
        bigger_node_invisible = bigger_node_result["invisible"] if "invisible" in bigger_node_result else False
        bigger_node_clipped = bigger_node_result["clipped"] if "clipped" in bigger_node_result else False
        bigger_node_overlapped_siblings = set(
            bigger_node_result["overlapped_siblings"]) if "overlapped_siblings" in bigger_node_result else {}
        bigger_node_overlapped_siblings_id = set(
            bigger_node_result["overlapped_siblings_id"]) if "overlapped_siblings_id" in bigger_node_result else {}

        consistent = True
        title = ""
        if normal_node_fully_visible != bigger_node_fully_visible:
            consistent = False

            result["inconsistency"].append(
                {
                    "normal_view_id": normal_view_id,
                    "bigger_view_id": bigger_view_id,
                    "type": "fully_visible inconsistency"
                }
            )

            title = f"{title}/fully_visible"
            print("inconsistency detected!")
        if normal_node_clipped != bigger_node_clipped:
            result["inconsistency"].append(
                {
                    "normal_view_id": normal_view_id,
                    "bigger_view_id": bigger_view_id,
                    "type": "clipping inconsistency"
                }
            )

            consistent = False

            title = f"{title}/clipped"
            print("inconsistency detected!")
        if normal_node_invisible != bigger_node_invisible:
            result["inconsistency"].append(
                {
                    "normal_view_id": normal_view_id,
                    "bigger_view_id": bigger_view_id,
                    "type": "invisible inconsistency"
                }
            )

            consistent = False
            title = f"{title}/invisible"
            print("inconsistency detected!")
        if normal_node_overlapped_siblings != bigger_node_overlapped_siblings:
            pass
            result["inconsistency"].append(
                {
                    "normal_view_id": normal_view_id,
                    "bigger_view_id": bigger_view_id,
                    "type": "overlapped sibling inconsistency",
                    "normal_view_siblings": list(normal_node_overlapped_siblings),
                    "bigger_view_siblings": list(bigger_node_overlapped_siblings),
                    "normal_view_siblings_id": list(normal_node_overlapped_siblings_id),
                    "bigger_view_siblings_id": list(bigger_node_overlapped_siblings_id),
                }
            )
            consistent = False
            title = f"{title}/overlapped_siblings"
            print("inconsistency detected!")
        if not consistent:
            pathlib.Path(intra_dir).mkdir(parents=True, exist_ok=True)
            ax1 = plt.subplot(1, 2, 1)
            if normal_img_path is not None:
                ax1.imshow(cv2.imread(normal_img_path, cv2.IMREAD_UNCHANGED))
            ax1.set_title(f"{normal_view_id}", fontsize=6)
            ax2 = plt.subplot(1, 2, 2)
            if bigger_img_path is not None:
                ax2.imshow(cv2.imread(bigger_img_path, cv2.IMREAD_UNCHANGED))
            ax2.set_title(f"{bigger_view_id}", fontsize=6)
            saved_img_path = os.path.join(intra_dir,
                                          f"intra_{normal_view_id}_{bigger_view_id}_result.png")
            plt.title(title, fontsize=6)
            plt.savefig(saved_img_path, dpi=600, transparent=True)
            plt.clf()
            print("intra inconsistency detected!")
        else:
            result["consistency"].append(
                {
                    "normal_view_id": normal_view_id,
                    "bigger_view_id": bigger_view_id
                }
            )
    return result


def get_inter_and_intra_result(normal_activity_inter_result,
                               bigger_activity_inter_result,
                               result_normal_middle,
                               page_result_dir,
                               normal_activity,
                               bigger_activity,
                               postfix,
                               ratio, buggy_view_ids):
    inter_time = time.time()
    inter_result = inter_view_analysis_result(normal_activity_inter_result,
                                              bigger_activity_inter_result,
                                              result_normal_middle,
                                              os.path.join(page_result_dir, f"intra_{postfix}"),
                                              normal_activity,
                                              bigger_activity,
                                              buggy_view_ids)
    inter_time_end = time.time()
    inter_times.append(inter_time_end - inter_time)
    intra_result_string = json.dumps(inter_result, default=dumper, indent=2)
    intra_result_path = os.path.join(page_result_dir, f"inter_{postfix}_result.json")
    with open(intra_result_path, 'w') as f:
        f.write(intra_result_string)
    intra_time = time.time()
    intra_result = intra_view_anlysis(normal_activity,
                                      bigger_activity,
                                      result_normal_middle,
                                      os.path.join(page_result_dir, f"inter_{postfix}"),
                                      normal_activity_inter_result,
                                      ratio,
                                      buggy_view_ids)
    intra_time_end = time.time()
    intra_times.append(intra_time_end - intra_time)
    inter_result_string = json.dumps(intra_result, default=dumper, indent=2)

    inter_result_path = os.path.join(page_result_dir, f"intra_{postfix}_result.json")
    with open(inter_result_path, 'w') as f:
        f.write(inter_result_string)

    buggy_view_ids = []

    if len(inter_result["inconsistency"]) > 0 or len(inter_result["inconsistency"]) > 0:
        for inconsistency_item in inter_result["inconsistency"]:
            if "bigger_view_id" in inconsistency_item:
                buggy_view_ids.append(inconsistency_item["bigger_view_id"])
            else:
                buggy_view_ids.append(inconsistency_item["normal_view_id"])
        for inconsistency_item in inter_result["inconsistency"]:
            if "bigger_view_id" in inconsistency_item:
                buggy_view_ids.append(inconsistency_item["bigger_view_id"])
            else:
                buggy_view_ids.append(inconsistency_item["normal_view_id"])

        return True, buggy_view_ids
    return False, buggy_view_ids


inter_times = []
intra_times = []

if __name__ == '__main__':
    # convert li to xml files
    li_to_text_jar_path = "layout-txt-generator.jar"
    li_path_pattern = "path/to/*/VH.li"
    li_path_list = glob.glob(li_path_pattern)
    for li_path in li_path_list:
        write_xml(li_path, li_to_text_jar_path)


    def size_sorter(item):
        """Get an item from the list (one-by-one) and return a score for that item."""
        base_name = os.path.basename(os.path.normpath(item))
        p = re.compile('([0-9]+)_fs_([0-9]+)_wm_([0-9]+)_.*?')
        m = p.match(base_name)
        return int(m.group(3)) * 10 + int(m.group(2))


    def fragment_sorter(item):
        base_name = os.path.basename(os.path.normpath(item))
        p = re.compile('([0-9]+)')
        m = p.match(base_name)
        return int(m.group(1))


    TN = 0
    TP = 0
    FN = 0
    FP = 0

    mapping_times = []

    raw_dataset_path = r"path/to/dataset_of_UI_pages"

    app_dirs = glob.glob(os.path.join(raw_dataset_path, "*/"))

    for app_dir in app_dirs:
        app_id = os.path.basename(os.path.normpath(app_dir))
        page_dirs = glob.glob(os.path.join(app_dir, "*/"))
        for page_dir in page_dirs:
            page_id = os.path.basename(os.path.normpath(page_dir))
            size_dirs = glob.glob(os.path.join(page_dir, "*/"))
            size_dirs = sorted(size_dirs, key=size_sorter)
            normal_activity = None
            bigger_activity = None
            bigger_mapping_size = None
            normal_mapping_size = None
            bigger_activity_inter_result = None
            normal_activity_inter_result = None
            middle_activity = None
            middle_mapping_size = None
            middle_activity_inter_result = None
            page_result_dir = os.path.join("path/to/save/results", f"{app_id}_{page_id}")

            for size_dir in size_dirs:
                p = re.compile('([0-9]+)_fs_([0-9]+)_wm_([0-9]+)_.*?')
                m = p.match(os.path.basename(os.path.normpath(size_dir)))
                size_int = int(m.group(1))
                fs_int = int(m.group(2))
                wm_int = int(m.group(3))

                if not (size_int == 7 or size_int == 20 or size_int == 10):
                    continue

                fragment_dirs = glob.glob(os.path.join(size_dir, "*/"))
                fragment_dirs = sorted(fragment_dirs, key=fragment_sorter)
                page_director = fragment_dirs[-1]

                if size_int == 7:
                    normal_activity = InterAnalyzer(
                        page_director=page_director,
                        sub_director=os.path.join(page_result_dir, f"{size_int}_{fs_int}_{wm_int}"))
                    normal_mapping_size = MappingSize(normal_activity.root, 7, normal_activity)
                    if not os.path.exists(os.path.join(page_result_dir, f"{size_int}_{fs_int}_{wm_int}",
                                                       "inter_analysis_output.json")):
                        normal_activity_inter_result = normal_activity.inter_view_analysis()
                    else:
                        with open(os.path.join(page_result_dir, f"{size_int}_{fs_int}_{wm_int}",
                                               "inter_analysis_output.json"), 'r') as f:
                            normal_activity_inter_result = json.load(f)
                    continue

                if size_int == 10:
                    middle_activity = InterAnalyzer(
                        page_director=page_director,
                        sub_director=os.path.join(page_result_dir, f"{size_int}_{fs_int}_{wm_int}"))
                    middle_mapping_size = MappingSize(middle_activity.root, 10, middle_activity)
                    if not os.path.exists(os.path.join(page_result_dir, f"{size_int}_{fs_int}_{wm_int}",
                                                       "intra_analysis_output.json")):
                        middle_activity_inter_result = middle_activity.inter_view_analysis()
                    else:
                        with open(os.path.join(page_result_dir, f"{size_int}_{fs_int}_{wm_int}",
                                               "intra_analysis_output.json"), 'r') as f:
                            middle_activity_inter_result = json.load(f)
                    continue

                if size_int == 20:
                    bigger_activity = InterAnalyzer(
                        page_director=page_director,
                        sub_director=os.path.join(page_result_dir, f"{size_int}_{fs_int}_{wm_int}"))
                    bigger_mapping_size = MappingSize(bigger_activity.root, 20, bigger_activity)
                    if not os.path.exists(os.path.join(page_result_dir, f"{size_int}_{fs_int}_{wm_int}",
                                                       "inter_analysis_output.json")):
                        bigger_activity_inter_result = bigger_activity.inter_view_analysis()
                    else:
                        with open(os.path.join(page_result_dir, f"{size_int}_{fs_int}_{wm_int}",
                                               "inter_analysis_output.json"), 'r') as f:
                            bigger_activity_inter_result = json.load(f)

            mapping_tool = MappingTool([normal_mapping_size, middle_mapping_size, bigger_mapping_size])
            mapping_start_time = time.time()
            result_normal_middle = mapping_tool.new_find_mapping(7, 10)
            mapping_end_time = time.time()
            mapping_times.append(mapping_end_time - mapping_start_time)
            mapping_start_time = time.time()
            result_middle_bigger = mapping_tool.new_find_mapping(10, 20)
            mapping_end_time = time.time()
            mapping_times.append(mapping_end_time - mapping_start_time)
            mapping_normal_middle_result_path = os.path.join(page_result_dir, "mapping_normal_middle_result.json")
            with open(mapping_normal_middle_result_path, 'w') as f:
                f.write(json.dumps(result_normal_middle, default=dumper, indent=2))

            mapping_middle_bigger_result_path = os.path.join(page_result_dir, "mapping_middle_bigger_result.json")
            with open(mapping_middle_bigger_result_path, 'w') as f:
                f.write(json.dumps(result_middle_bigger, default=dumper, indent=2))

            has_bug_1, buggy_view_ids = get_inter_and_intra_result(normal_activity_inter_result,
                                                                   middle_activity_inter_result,
                                                                   result_normal_middle,
                                                                   page_result_dir,
                                                                   normal_activity,
                                                                   middle_activity,
                                                                   "normal_middle",
                                                                   (540 / 420) * (540 / 420), [])

            has_bug_2, _ = get_inter_and_intra_result(middle_activity_inter_result,
                                                      bigger_activity_inter_result,
                                                      result_middle_bigger,
                                                      page_result_dir,
                                                      middle_activity,
                                                      bigger_activity,
                                                      "middle_bigger",
                                                      1.3 * 1.3, buggy_view_ids)