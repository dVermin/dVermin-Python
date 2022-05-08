import numpy as np

from rectangle import Rectangle
import copy

class RegionRect:
    def __init__(self, x, y, visible_region):
        self.x = x
        self.y = y
        self.visible_region = visible_region
        self.width = self.visible_region.shape[1]
        self.height = self.visible_region.shape[0]
        self.rect = Rectangle(x, y, x + self.width, y + self.height)

    def __deepcopy__(self, memo_dict):
        x = self.x
        y = self.y
        visible_region = np.copy(self.visible_region)
        return RegionRect(x, y, visible_region)

    @staticmethod
    def compute_visible_clipping(inner_visible_region_rect, outer_visible_region_rect):
        intersect_rect = inner_visible_region_rect.rect & outer_visible_region_rect.rect
        if intersect_rect is None:
            return None
        intersect_visible_region = inner_visible_region_rect.visible_region[
                                   intersect_rect.y1 - inner_visible_region_rect.rect.y1:intersect_rect.y1 - inner_visible_region_rect.rect.y1 + intersect_rect.get_height(),
                                   intersect_rect.x1 - inner_visible_region_rect.rect.x1:intersect_rect.x1 - inner_visible_region_rect.rect.x1 + intersect_rect.get_width()
                                   ]
        intersect_visible_region = np.copy(intersect_visible_region)
        outer_intersect_visible_region = outer_visible_region_rect.visible_region[
                                   intersect_rect.y1 - outer_visible_region_rect.rect.y1:intersect_rect.y1 - outer_visible_region_rect.rect.y1 + intersect_rect.get_height(),
                                   intersect_rect.x1 - outer_visible_region_rect.rect.x1:intersect_rect.x1 - outer_visible_region_rect.rect.x1 + intersect_rect.get_width()
                                   ]
        intersect_visible_region = np.bitwise_and(intersect_visible_region, outer_intersect_visible_region)
        return RegionRect(intersect_rect.x1, intersect_rect.y1, intersect_visible_region)


    def check_if_overlapped_with(self, outer_visible_region_rect):
        intersect_rect = self.rect & outer_visible_region_rect.rect
        if intersect_rect is None:
            return False
        intersect_self_region = self.visible_region[
                                   intersect_rect.y1 - self.rect.y1:intersect_rect.y1 - self.rect.y1 + intersect_rect.get_height(),
                                   intersect_rect.x1 - self.rect.x1:intersect_rect.x1 - self.rect.x1 + intersect_rect.get_width()
                                   ]
        outer_intersect_visible_region = outer_visible_region_rect.visible_region[
                                   intersect_rect.y1 - outer_visible_region_rect.rect.y1:intersect_rect.y1 - outer_visible_region_rect.rect.y1 + intersect_rect.get_height(),
                                   intersect_rect.x1 - outer_visible_region_rect.rect.x1:intersect_rect.x1 - outer_visible_region_rect.rect.x1 + intersect_rect.get_width()
                                   ]
        result = np.bitwise_and(intersect_self_region, outer_intersect_visible_region)
        result = np.bitwise_and(result, intersect_self_region)
        if np.sum(result) > 0:
            return True
        return False

    @staticmethod
    def compute_visible_subintersect(inner_visible_region_rect, outer_visible_region_rect):
        result_region_rect = copy.deepcopy(inner_visible_region_rect)
        intersect_rect = result_region_rect.rect & outer_visible_region_rect.rect
        if intersect_rect is None:
            return result_region_rect
        intersect_visible_region = result_region_rect.visible_region[
                                   intersect_rect.y1 - result_region_rect.rect.y1:intersect_rect.y1 - result_region_rect.rect.y1 + intersect_rect.get_height(),
                                   intersect_rect.x1 - result_region_rect.rect.x1:intersect_rect.x1 - result_region_rect.rect.x1 + intersect_rect.get_width()
                                   ]
        outer_intersect_visible_region = outer_visible_region_rect.visible_region[
                                   intersect_rect.y1 - outer_visible_region_rect.rect.y1:intersect_rect.y1 - outer_visible_region_rect.rect.y1 + intersect_rect.get_height(),
                                   intersect_rect.x1 - outer_visible_region_rect.rect.x1:intersect_rect.x1 - outer_visible_region_rect.rect.x1 + intersect_rect.get_width()
                                   ]
        outer_intersect_visible_region = np.bitwise_not(outer_intersect_visible_region)
        union = np.bitwise_and(intersect_visible_region, outer_intersect_visible_region)
        result_region_rect.visible_region[
            intersect_rect.y1 - result_region_rect.rect.y1:intersect_rect.y1 - result_region_rect.rect.y1 + intersect_rect.get_height(),
            intersect_rect.x1 - result_region_rect.rect.x1:intersect_rect.x1 - result_region_rect.rect.x1 + intersect_rect.get_width()
        ] = union
        return result_region_rect

    @staticmethod
    def compute_visible_subtract(inner_visible_region_rect, outer_visible_region_rect):
        result_region_rect = copy.deepcopy(inner_visible_region_rect)
        intersect_rect = result_region_rect.rect & outer_visible_region_rect.rect
        if intersect_rect is None:
            return result_region_rect
        intersect_visible_region = result_region_rect.visible_region[
                                   intersect_rect.y1 - result_region_rect.rect.y1:intersect_rect.y1 - result_region_rect.rect.y1 + intersect_rect.get_height(),
                                   intersect_rect.x1 - result_region_rect.rect.x1:intersect_rect.x1 - result_region_rect.rect.x1 + intersect_rect.get_width()
                                   ]
        outer_intersect_visible_region = outer_visible_region_rect.visible_region[
                                         intersect_rect.y1 - outer_visible_region_rect.rect.y1:intersect_rect.y1 - outer_visible_region_rect.rect.y1 + intersect_rect.get_height(),
                                         intersect_rect.x1 - outer_visible_region_rect.rect.x1:intersect_rect.x1 - outer_visible_region_rect.rect.x1 + intersect_rect.get_width()
                                         ]
        union = np.where(outer_intersect_visible_region>0, False, intersect_visible_region)
        # outer_intersect_visible_region = np.bitwise_not(outer_intersect_visible_region)
        # union = np.bitwise_and(intersect_visible_region, outer_intersect_visible_region)
        result_region_rect.visible_region[
        intersect_rect.y1 - result_region_rect.rect.y1:intersect_rect.y1 - result_region_rect.rect.y1 + intersect_rect.get_height(),
        intersect_rect.x1 - result_region_rect.rect.x1:intersect_rect.x1 - result_region_rect.rect.x1 + intersect_rect.get_width()
        ] = union
        return result_region_rect

    def get_visible_region(self, visible_region_rect):
        if visible_region_rect is None:
            return self.visible_region
        intersect_rect = self.rect & visible_region_rect.rect
        if intersect_rect is None:
            return None
        y1 = intersect_rect.y1 - self.rect.y1
        x1 = intersect_rect.x1 - self.rect.x1
        intersect_region = self.visible_region[y1:y1+intersect_rect.get_height(),
                                            x1:x1+intersect_rect.get_width()]
        y1 = intersect_rect.y1 - visible_region_rect.rect.y1
        x1 = intersect_rect.x1 - visible_region_rect.rect.x1
        intersect_visible_region = visible_region_rect.visible_region[y1:y1+intersect_rect.get_height(),
                                            x1:x1+intersect_rect.get_width()]
        return np.bitwise_and(intersect_region, intersect_visible_region)

    @staticmethod
    def compute_height_and_width(visible_region):
        """
        return the minimum region containing the content region
        :param visible_region: a region containing information
        :return: the height and width of this minimum region
        """
        width = np.sum(np.where(np.sum(visible_region, axis=0) > 0, 1, 0))
        height = np.sum(np.where(np.sum(visible_region, axis=1) > 0, 1, 0))
        return height, width

    @staticmethod
    def compute_offset(visible_region):
        """
        return the minimum region containing the content region
        :param visible_region: a region containing information
        :return: the height and width of this minimum region
        """
        width = np.where(np.sum(visible_region, axis=0) > 0, 1, 0)
        width_index = np.argwhere(width==1).reshape((-1))
        if len(width_index)>1:
            width_offset = sorted(width_index)[0]
        else:
            width_offset = -1

        height = np.where(np.sum(visible_region, axis=1) > 0, 1, 0)
        height_index = np.argwhere(height == 1).reshape((-1))
        if len(height_index) > 1:
            height_offset = sorted(height_index)[0]
        else:
            height_offset = -1
        if height_offset!=-1 and height_offset!=-1:
            visible_region = visible_region[height_offset:height_offset+np.sum(height),
                                 width_offset:width_offset+np.sum(width)]


        return height_offset, width_offset, visible_region
