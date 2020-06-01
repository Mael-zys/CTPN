
import sys
sys.path.append("../..")
import os
import codecs
import cv2
import draw_image
import lmdb
import numpy as np
import Net as Net
from torch.utils.data import Dataset
from tqdm import tqdm
import random
from torchvision.transforms import transforms
from PIL import Image,ImageEnhance,ImageOps,ImageFile

def get_rotate_img_boxes(img,bboxes,random_angle = 20):
    # img = cv2.imread(img_file)
    # fid = open(txt_file, 'r', encoding='utf-8-sig')
    # bboxes = []
    # for line in fid.readlines():
    #     line = line.strip().replace('\ufeff', '').split(',')
    #     line = line[:8]
    #     line = [int(x) for x in line]
    #     line = np.array(line)
    #     line = line.reshape(4, 2)
    #     line = cv2.minAreaRect(line)
    #     line = cv2.boxPoints(line).astype(np.int)
    #     line = order_point(line)
    #     bboxes.append(line)
    img1, M = random_rotate(img,random_angle)
    new_all_rects = []
    # bboxes = np.array(bboxes)
    for item in bboxes:
        rect = []
        # print(item)
        item = np.array(item)
        # print(item)
        item=item.reshape(4,2)
        for coord in item:
            # print(coord)
            rotate_coord = cal_affine_coord(coord,M)
            rect.append(rotate_coord)
        new_all_rects.append((np.array(rect).reshape(8)).tolist())
    return img1,new_all_rects

def cal_affine_coord(ori_coord,M):
    x = ori_coord[0]
    y = ori_coord[1]
    _x = x * M[0, 0] + y * M[0, 1] + M[0, 2]
    _y = x * M[1, 0] + y * M[1, 1] + M[1, 2]
    return [int(_x),int(_y)]

def random_rotate(img,random_angle = 20):
    angle = random.random() * 2 * random_angle - random_angle
    w, h = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
    img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
    return img_rotation,rotation_matrix


def randomGaussian(image, mean=0.2, sigma=0.3):
    """
    对图像进行高斯噪声处理
    :param image:
    :return:
    """

    def gaussianNoisy(im, mean=0.2, sigma=0.3):
        """
        对图像做高斯噪音处理
        :param im: 单通道图像
        :param mean: 偏移量
        :param sigma: 标准差
        :return:
        """
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    # 将图像转化成数组
    img = np.asarray(image)
    img = np.array(img)
    img.flags.writeable = True  # 将数组改为读写模式
    width, height = img.shape[:2]
    img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
    img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
    img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
    img[:, :, 0] = img_r.reshape([width, height])
    img[:, :, 1] = img_g.reshape([width, height])
    img[:, :, 2] = img_b.reshape([width, height])
    return Image.fromarray(np.uint8(img))

def randomColor(image):
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    random_factor = np.random.randint(0, 21) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 21) / 10.  # 随机因子
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度 

# 读取图片的标签，以list形式返回
def read_gt_file(path, have_BOM=False):
    result = []
    if have_BOM:
        fp = codecs.open(path, 'r', 'utf-8-sig')
    else:
        fp = open(path, 'r')
    for line in fp.readlines():
        pt = line.split(',')
        if have_BOM:
            box = [int(round(float(pt[i]))) for i in range(8)]
        else:
            box = [int(round(float(pt[i]))) for i in range(8)]
        result.append(box)
    fp.close()
    return result


def create_dataset_icdar2015(img_root, gt_root, output_path):
    im_list = os.listdir(img_root)
    im_path_list = []
    gt_list = []
    for im in im_list:
        name, _ = os.path.splitext(im)
        gt_name = 'gt_' + name + '.txt'
        gt_path = os.path.join(gt_root, gt_name)
        if not os.path.exists(gt_path):
            print('Ground truth file of image {0} not exists.'.format(im))
        im_path_list.append(os.path.join(img_root, im))
        gt_list.append(gt_path)
    assert len(im_path_list) == len(gt_list)
    create_dataset(output_path, im_path_list, gt_list)


# 缩放图像具有一定规则：首先要保证文本框label的最短边也要等于600。
# 我们通过scale = float(shortest_side)/float(min(height, width))来求得图像的缩放系数，对原始图像进行缩放。
# 同时我们也要对我们的label也要根据该缩放系数进行缩放。
def scale_img0(img, gt):
    height = img.shape[0]
    width = img.shape[1]
    short_side=640
    long_side=960
    if min(height, width)<long_side:
        scale = float(short_side)/float(min(height, width))
        # img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        if scale*max(height,width)>1440:
            longer=1440
        else :
            longer=int(np.floor(scale*max(height,width)))
        if img.shape[0] < img.shape[1]: #and img.shape[0] != short_side:
            img = cv2.resize(img, (longer,short_side ))
        else:# img.shape[0] > img.shape[1]: # and img.shape[1] != short_side:
            img = cv2.resize(img, (short_side,longer))
        # elif img.shape[0] != short_side:
        #     img = cv2.resize(img, (short_side, short_side))
    else :
        scale = float(long_side)/float(min(height, width))
        # img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        if scale*max(height,width)>1440:
            longer=1440
        else :
            longer=int(np.floor(scale*max(height,width)))
        if img.shape[0] < img.shape[1]:# and img.shape[0] != long_side:
            img = cv2.resize(img, (longer,long_side))
        else:# img.shape[0] > img.shape[1]: #and img.shape[1] != long_side:
            img = cv2.resize(img, (long_side,longer))
        # elif img.shape[0] != long_side:
        #     img = cv2.resize(img, (long_side, long_side))
    # img = cv2.resize(img,(shortest_side,shortest_side))

    # 求出h和w方向上的缩放比例
    h_scale = float(img.shape[0])/float(height)
    w_scale = float(img.shape[1])/float(width)
    scale_gt = []
    for box in gt:
        scale_box = []
        for i in range(len(box)):
            if i % 2 == 0:
                scale_box.append(int(int(box[i]) * w_scale))
            else:
                scale_box.append(int(int(box[i]) * h_scale))
        scale_gt.append(scale_box)
    # 返回缩放后的图片和label
    return img, scale_gt

min_size_list=[512,640,720,800,960]
def scale_img(img, gt):
    height = img.shape[0]
    width = img.shape[1]
    short_side = np.random.choice(min_size_list,1)[0]

    scale = float(short_side)/float(min(height, width))
        # img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    if scale*max(height,width)>1440:
        longer=1440
    else :
        longer=int(np.floor(scale*max(height,width)))
    if img.shape[0] < img.shape[1]: #and img.shape[0] != short_side:
        img = cv2.resize(img, (longer,short_side ))
    else:# img.shape[0] > img.shape[1]: # and img.shape[1] != short_side:
        img = cv2.resize(img, (short_side,longer))
    
    # 求出h和w方向上的缩放比例
    h_scale = float(img.shape[0])/float(height)
    w_scale = float(img.shape[1])/float(width)
    scale_gt = []
    for box in gt:
        scale_box = []
        for i in range(len(box)):
            if i % 2 == 0:
                scale_box.append(int(int(box[i]) * w_scale))
            else:
                scale_box.append(int(int(box[i]) * h_scale))
        scale_gt.append(scale_box)
    # 返回缩放后的图片和label
    return img, scale_gt

def scale_img_only(img, shortest_side=600):
    height = img.shape[0]
    width = img.shape[1]
    scale = float(shortest_side)/float(min(height, width))
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    if img.shape[0] < img.shape[1] and img.shape[0] != 600:
        img = cv2.resize(img, (600, img.shape[1]))
    elif img.shape[0] > img.shape[1] and img.shape[1] != 600:
        img = cv2.resize(img, (img.shape[0], 600))
    elif img.shape[0] != 600:
        img = cv2.resize(img, (600, 600))

    return img


def check_img(img):
    if img is None:
        return False
    height, width = img.shape[0], img.shape[1]
    if height * width == 0:
        return False
    return True


def write_cache(env, data):
    with env.begin(write=True) as e:
        for i, l in data.iteritems():
            e.put(i, l)


def box_list2str(l):
    result = []
    for box in l:
        if not len(box) % 8 == 0:
            return '', False
        result.append(','.join(box))
    return '|'.join(result), True


def create_dataset(output_path, img_list, gt_list):
    assert len(img_list) == len(gt_list)
    net = Net.VGG_16()
    num = len(img_list)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    env = lmdb.open(output_path, map_size=1099511627776)
    cache = {}
    counter = 1
    for i in range(num):
        img_path = img_list[i]
        gt = gt_list[i]
        if not os.path.exists(img_path):
            print("{0} is not exist.".format(img_path))
            continue

        if len(gt) == 0:
            print("Ground truth of {0} is not exist.".format(img_path))
            continue

        img = cv2.imread(img_path)
        if not check_img(img):
            print('Image {0} is not valid.'.format(img_path))
            continue

        img, gt = scale_img(img, gt)
        gt_str = box_list2str(gt)
        if not gt_str[1]:
            print("Ground truth of {0} is not valid.".format(img_path))
            continue

        img_key = 'image-%09d' % counter
        gt_key = 'gt-%09d' % counter
        cache[img_key] = draw_image.np_img2base64(img, img_path)
        cache[gt_key] = gt_str[0]
        counter += 1
        if counter % 100 == 0:
            write_cache(env, cache)
            cache.clear()
            print('Written {0}/{1}'.format(counter, num))
    cache['num'] = str(counter - 1)
    write_cache(env, cache)
    print('Create dataset with {0} image.'.format(counter - 1))


class LmdbDataset(Dataset):
    def __init__(self, root, transformer=None):
        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print("Cannot create lmdb from root {0}.".format(root))
        with self.env.begin(write=False) as e:
            self.data_num = int(e.get('num'))
        self.transformer = transformer

    def __len__(self):
        return self.data_num

    def __getitem__(self, index):
        assert index <= len(self), 'Index out of range.'
        index += 1
        with self.env.begin(write=False) as e:
            img_key = 'image-%09d' % index
            img_base64 = e.get(img_key)
            img = draw_image.base642np_image(img_base64)
            gt_key = 'gt-%09d' % index
            gt = str(e.get(gt_key))
        return img, gt
