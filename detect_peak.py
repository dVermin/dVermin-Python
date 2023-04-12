import math
import os

import cv2
import numpy as np
import pylab
import matplotlib.pyplot as plt
from skimage.filters.thresholding import threshold_otsu

from rectangle import Rectangle
from skimage import filters


def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0] * len(y)
    stdFilter = [0] * len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i - 1]) > threshold * stdFilter[i - 1]:
            if y[i] > avgFilter[i - 1]:
                signals[i] = 1
            else:
                signals[i] = -1
            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i - 1]
            avgFilter[i] = np.mean(filteredY[(i - lag + 1):i + 1])
            stdFilter[i] = np.std(filteredY[(i - lag + 1):i + 1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i - lag + 1):i + 1])
            stdFilter[i] = np.std(filteredY[(i - lag + 1):i + 1])

    return dict(signals=np.asarray(signals),
                avgFilter=np.asarray(avgFilter),
                stdFilter=np.asarray(stdFilter))


# 计算灰度直方图
def calcGrayHist(grayimage):
    # 灰度图像矩阵的高，宽
    rows, cols = grayimage.shape
    print(grayimage.shape)
    # 存储灰度直方图
    grayHist = np.zeros([256], np.uint64)
    for r in range(rows):
        for c in range(cols):
            grayHist[grayimage[r][c]] += 1
    return grayHist


# OTSU自动阈值分割
def OTSU(image):
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape
    # 1.计算灰度直方图
    grayHist = calcGrayHist(gray)
    # 2.灰度直方图归一化
    uniformGrayHist = grayHist / float(rows * cols)
    # 3.计算零阶累计矩何一阶累计矩
    zeroCumuMoment = np.zeros([256], np.float32)
    oneCumuMoment = np.zeros([256], np.float32)
    for k in range(256):
        if k == 0:
            zeroCumuMoment[k] = uniformGrayHist[0]
            oneCumuMoment[k] = (k) * uniformGrayHist[0]
        else:
            zeroCumuMoment[k] = zeroCumuMoment[k - 1] + uniformGrayHist[k]
            oneCumuMoment[k] = oneCumuMoment[k - 1] + k * uniformGrayHist[k]
    # 计算类间方差
    variance = np.zeros([256], np.float32)
    for k in range(255):
        if zeroCumuMoment[k] == 0 or zeroCumuMoment[k] == 1:
            variance[k] = 0
        else:
            variance[k] = math.pow(oneCumuMoment[255] * zeroCumuMoment[k] -
                                   oneCumuMoment[k], 2) / (zeroCumuMoment[k] * (1.0 - zeroCumuMoment[k]))
    # 找到阈值、
    threshLoc = np.where(variance[0:255] == np.max(variance[0:255]))
    thresh = threshLoc[0][0]
    # 阈值处理
    threshold = np.copy(gray)
    threshold[threshold > thresh] = 255
    threshold[threshold <= thresh] = 0
    return threshold, thresh


def threshTwoPeaks(image, image_alpha):
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算灰度直方图
    # histogram = calcGrayHist(gray)
    histogram = cv2.calcHist([gray], [0], image_alpha, [256], [0, 255])
    histogram = histogram.reshape((-1))
    # 寻找灰度直方图的最大峰值对应的灰度值
    maxLoc = np.where(histogram == np.max(histogram))
    firstPeak = maxLoc[0][0]
    # 寻找灰度直方图的第二个峰值对应的灰度值
    measureDists = np.zeros([256], np.float32)
    for k in range(256):
        measureDists[k] = pow(k - firstPeak, 2) * histogram[k]
    maxLoc2 = np.where(measureDists == np.max(measureDists))
    secondPeak = maxLoc2[0][0]

    # 找到两个峰值之间的最小值对应的灰度值，作为阈值
    thresh = 0
    if firstPeak > secondPeak:  # 第一个峰值再第二个峰值的右侧
        temp = histogram[int(secondPeak):int(firstPeak)]
        minloc = np.where(temp == np.min(temp))
        thresh = secondPeak + minloc[0][0] + 1
    else:  # 第一个峰值再第二个峰值的左侧
        temp = histogram[int(firstPeak):int(secondPeak)]
        minloc = np.where(temp == np.min(temp))
        thresh = firstPeak + minloc[0][0] + 1

    # 找到阈值之后进行阈值处理，得到二值图
    threshImage_out = gray.copy()
    # 大于阈值的都设置为255
    threshImage_out[threshImage_out > thresh] = 255
    # 小于阈值的都设置为0
    threshImage_out[threshImage_out <= thresh] = 0
    return thresh, threshImage_out


def get_binary_image_of_text(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_alpha = image[:, :, -1]
    image_alpha = np.where(image_alpha > 0, 1, 0).astype(np.uint8)

    histogram = cv2.calcHist([gray], [0], image_alpha, [256], [0, 255])
    try:
        kkk = threshold_otsu(hist=histogram.reshape((-1)))
    except:
        kkk = 0

    print(kkk)
    kkkk_1 = np.where((image_alpha > 0) & (gray < kkk), 255, 0).astype(np.uint8)
    kkkk_2 = np.where((image_alpha > 0) & (gray >= kkk), 255, 0).astype(np.uint8)
    num_labels_1, labels_1, stats_1, centroids_1 = cv2.connectedComponentsWithStats(kkkk_1, connectivity=8)
    num_labels_2, labels_2, stats_2, centroids_2 = cv2.connectedComponentsWithStats(kkkk_2, connectivity=8)

    def get_rect_with_stat(st):
        return Rectangle(st[0], st[1], st[0] + st[2], st[1] + st[3])

    stats_1 = stats_1[1:]
    stats_2 = stats_2[1:]

    has_background = False
    _, ret_image = cv2.threshold(image[:, :, -1], 0, 255, cv2.THRESH_OTSU)
    counter_part = 255 - ret_image
    if len(stats_1) > 0 and len(stats_2) > 0:
        if stats_1[0][-1] > stats_2[0][-1]:
            larger_stats = stats_1[0]
            over_threshold = False
        else:
            larger_stats = stats_2[0]
            over_threshold = True

        if larger_stats is not None:
            has_background = True
            larger_stat_rect = get_rect_with_stat(larger_stats)
            # for stat in stats_1:
            #     this_stat_rect = get_rect_with_stat(stat)
            #     if this_stat_rect & larger_stat_rect == None:
            #         has_background = False
            #         break
            # if has_background:
            #     for stat in stats_2:
            #         this_stat_rect = get_rect_with_stat(stat)
            #         if this_stat_rect & larger_stat_rect == None:
            #             has_background = False
            #             break
            for stat in stats_1:
                this_stat_rect = get_rect_with_stat(stat)
                intersect = this_stat_rect & larger_stat_rect
                if intersect == None or intersect.area() < this_stat_rect.area():
                    has_background = False
                    break
            if has_background:
                for stat in stats_2:
                    this_stat_rect = get_rect_with_stat(stat)
                    intersect = this_stat_rect & larger_stat_rect
                    if intersect == None or intersect.area() < this_stat_rect.area():
                        has_background = False
                        break
        else:
            has_background = False
        if has_background:
            if over_threshold:
                ret_image = kkkk_1
                counter_part = 255 - kkkk_1
            else:
                ret_image = kkkk_2
                counter_part = 255 - kkkk_2

    # plt.title(f"has bg: {has_background}")
    # plt.imshow(ret_image)
    # plt.show()
    return ret_image, counter_part, has_background


def connected_component_label(labels, stats):
    # # Getting the input image
    # img = cv2.imread(path, 0)
    # # Converting those pixels with values 1-127 to 0 and others to 1
    # img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    # # Applying cv2.connectedComponents()
    # num_labels, labels = cv2.connectedComponents(img)

    # Map component labels to hue val, 0-179 is the hue range in OpenCV
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Converting cvt to BGR
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0
    labeled_img[label_hue > 0] = 255

    img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB)

    for inx, stat in enumerate(stats):
        if inx==0:
            continue
        img = cv2.rectangle(img, (stat[0], stat[1]), (stat[0]+stat[2], stat[1]+stat[3]), (0,0,255), 2)
        if inx>6:
            offset = inx-10
            # offset = 1
            if inx<=9:

                    img = cv2.putText(img, f'{inx}\'', (stat[0], stat[1] + 60),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            else:
                img = cv2.putText(img, f'{inx}\'', (stat[0]+offset*(13+offset), stat[1] + 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            # For printing text
            # if offset%2 == 0:
            #     img = cv2.putText(img, f'{inx}\'', (stat[0]-15, stat[1]-5),
            #                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            # else:
            #     img = cv2.putText(img, f'{inx}\'', (stat[0]-15, stat[1] + 60),
            #                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        else:
            if inx==6:

                img = cv2.putText(img, f'{1}\'', (stat[0], stat[1] - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            else:
                img = cv2.putText(img, f'{inx+1}\'', (stat[0], stat[1] - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

    return img

    return cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB)

    # # Showing Original Image
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title("Orginal Image")
    # plt.show()
    #
    # # Showing Image after Component Labeling
    # plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.title("Image after Component Labeling")
    # plt.show()


def get_binary_image_of_text_debug(image_path):
    basename = os.path.basename(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    cv2.imwrite(f'{basename}-ori.png', image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f'{basename}-gray.png', gray)
    image_alpha = image[:, :, -1]
    cv2.imwrite(f'{basename}-alpha.png', image_alpha)
    cv2.imwrite(f'{basename}-rgb.png', image[:, :, :-1])
    image_alpha = np.where(image_alpha > 0, 1, 0).astype(np.uint8)

    histogram = cv2.calcHist([gray], [0], image_alpha, [256], [0, 255])
    histogram = histogram.reshape((-1))
    log_hist = np.log(histogram+1)
    # log_hist = np.where(log_hist > 0, log_hist, 0)

    try:
        kkk = threshold_otsu(hist=histogram.reshape((-1)))
    except:
        kkk = 0

    x_axis = [x for x in range(256)]
    plt.figure(figsize=(7, 8))
    plt.hist(x_axis, 256, weights=log_hist)
    plt.axvline(x=kkk, color='r', ls=':')
    plt.annotate(f"Threshold: {kkk}", xy=(kkk, np.max(log_hist)-2), xytext =(25, np.max(log_hist)-2), color='black',
                 arrowprops=dict(
                     arrowstyle="->",
                     facecolor='red'
                     ),
                 fontsize=20,
                 bbox={'facecolor':'white','alpha':1,'edgecolor':'none','pad':1},
                 textcoords='offset points', )
    plt.ylabel("Log of (frequency value + 1)", fontsize=20)

    begin = 300
    end = 0
    for i in range(256):
        if log_hist[i] > 0 and i < begin:
            begin = i
        if log_hist[i] > 0 and i > end:
            end = i
    plt.xlim(begin-5, end+5)
    # plt.ylim(0, max(log_hist)+1)
    # plt.xticks(np.linspace(0, 255, 7).astype(np.uint8))  # arbitrary chosen
    # plt.yticks(np.linspace(0, max(log_hist)+1, 5))  # arbitrary chosen
    plt.xlabel("Grayscale value", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{basename}-hist.png", bbox_inches='tight', dpi=1000)
    plt.show()

    print(kkk)
    kkkk_1 = np.where((image_alpha > 0) & (gray < kkk), 255, 0).astype(np.uint8)
    kkkk_2 = np.where((image_alpha > 0) & (gray >= kkk), 255, 0).astype(np.uint8)
    num_labels_1, labels_1, stats_1, centroids_1 = cv2.connectedComponentsWithStats(kkkk_1, connectivity=8)

    labels_1_color = connected_component_label(labels_1, stats_1)
    plt.imshow(labels_1_color)
    plt.show()
    cv2.imwrite(f'{basename}-more-nocolor.png', kkkk_1)

    cv2.imwrite(f'{basename}-more1.png', labels_1_color)
    num_labels_2, labels_2, stats_2, centroids_2 = cv2.connectedComponentsWithStats(kkkk_2, connectivity=8)
    labels_2_color = connected_component_label(labels_2, stats_2)
    plt.imshow(labels_2_color)
    plt.show()
    cv2.imwrite(f'{basename}-less-nocolor.png', kkkk_2)

    cv2.imwrite(f'{basename}-less1.png', labels_2_color)
    def get_rect_with_stat(st):
        return Rectangle(st[0], st[1], st[0] + st[2], st[1] + st[3])

    stats_1 = stats_1[1:]
    stats_2 = stats_2[1:]
    print(
        stats_2
    )
    print(np.sum(stats_2, axis=0))

    has_background = False
    _, ret_image = cv2.threshold(image[:, :, -1], 0, 255, cv2.THRESH_OTSU)
    counter_part = 255 - ret_image

    stats_1_areas = stats_1[:, -1].reshape((-1))
    stats_1_areas_args = np.argwhere(stats_1_areas < 5)
    filtered_stats_1 = np.delete(stats_1, stats_1_areas_args, 0)

    stats_2_areas = stats_2[:, -1].reshape((-1))
    stats_2_areas_args = np.argwhere(stats_2_areas < 5)
    filtered_stats_2 = np.delete(stats_2, stats_2_areas_args, 0)

    if len(filtered_stats_1) > 0 and len(filtered_stats_2) > 0:
        if stats_1[0][-1] > stats_2[0][-1]:
            larger_stats = stats_1[0]
            over_threshold = False
        else:
            larger_stats = stats_2[0]
            over_threshold = True

        if larger_stats is not None:
            has_background = True
            larger_stat_rect = get_rect_with_stat(larger_stats)
            for stat in stats_1:
                this_stat_rect = get_rect_with_stat(stat)
                intersect = this_stat_rect & larger_stat_rect
                if intersect == None or intersect.area()<this_stat_rect.area():
                    has_background = False
                    break
            if has_background:
                for stat in stats_2:
                    this_stat_rect = get_rect_with_stat(stat)
                    intersect = this_stat_rect & larger_stat_rect
                    if intersect == None or intersect.area() < this_stat_rect.area():
                        has_background = False
                        break
        else:
            has_background = False
        if has_background:
            if over_threshold:
                ret_image = kkkk_1
                counter_part = 255 - kkkk_1
            else:
                ret_image = kkkk_2
                counter_part = 255 - kkkk_2


    print(f"has bg: {has_background}")
    # plt.imshow(ret_image)
    # plt.show()
    return ret_image, counter_part, has_background


