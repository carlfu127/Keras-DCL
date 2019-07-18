import numpy as np
import numbers
import random
import os
from PIL import Image, ImageStat

def to_categorical(y, nb_class):
    y = np.asarray(y, dtype='int32')
    Y = np.zeros((len(y), nb_class))
    Y[np.arange(len(y)),y] = 1.
    return Y

def resize(img, size, interpolation=Image.BILINEAR):
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)

def randomRotation(img, degrees, resample=False, expand=False, center=None):
    if isinstance(degrees, numbers.Number):
        if degrees < 0:
            raise ValueError("If degrees is a single number, it must be positive.")
        degrees = (-degrees, degrees)
    else:
        if len(degrees) != 2:
            raise ValueError("If degrees is a sequence, it must be of len 2.")
        degrees = degrees
    angle = random.uniform(degrees[0], degrees[1])
    return img.rotate(angle, resample, expand, center)

def randomCrop(img, size):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        size = size
    w, h = img.size
    th, tw = size
    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)

    return img.crop((j, i, j + tw, i + th))

def centerCrop(img, size):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        size = size
    w, h = img.size
    th, tw = size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return img.crop((j, i, j + th, i + tw))

def randomHorizontalFlip(img):
    if random.random() < 0.5:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def randomSwap(img, size):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
        size = size

    widthcut, highcut = img.size
    img = img.crop((10, 10, widthcut - 10, highcut - 10))
    images = crop_image(img, size)
    pro = 5
    if pro >= 5:
        tmpx = []
        tmpy = []
        count_x = 0
        count_y = 0
        k = 1
        RAN = 2
        for i in range(size[1] * size[0]):
            tmpx.append(images[i])
            count_x += 1
            if len(tmpx) >= k:
                tmp = tmpx[count_x - RAN:count_x]
                random.shuffle(tmp)
                tmpx[count_x - RAN:count_x] = tmp
            if count_x == size[0]:
                tmpy.append(tmpx)
                count_x = 0
                count_y += 1
                tmpx = []
            if len(tmpy) >= k:
                tmp2 = tmpy[count_y - RAN:count_y]
                random.shuffle(tmp2)
                tmpy[count_y - RAN:count_y] = tmp2
        random_im = []
        for line in tmpy:
            random_im.extend(line)

        # random.shuffle(images)
        width, high = img.size
        iw = int(width / size[0])
        ih = int(high / size[1])
        toImage = Image.new('RGB', (iw * size[0], ih * size[1]))
        x = 0
        y = 0
        for i in random_im:
            i = i.resize((iw, ih), Image.ANTIALIAS)
            toImage.paste(i, (x * iw, y * ih))
            x += 1
            if x == size[0]:
                x = 0
                y += 1
    else:
        toImage = img
    toImage = toImage.resize((widthcut, highcut))
    return toImage

def normalize(img):
    # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    # std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    #
    # img_data = np.array(img, dtype=np.float32) / 255.
    # img_data = (img_data - mean) / std

    mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    img_data = np.array(img)
    img_data = img_data[..., ::-1]
    img_data = img_data - mean
    return img_data


def swap(img):
    img = randomSwap(img, (7, 7))
    return img

def unswap(img):
    img = resize(img, (512, 512))
    img = randomRotation(img, degrees=15)
    img = randomCrop(img, 448)
    img = randomHorizontalFlip(img)
    return img

def totensor(img):
    img = resize(img, (448, 448))
    img = normalize(img)
    return img

def noswap(img):
    img = resize(img, (512, 512))
    img = centerCrop(img, 448)
    return img

def crop_image(image, cropnum):
    width, high = image.size
    crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
    crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
    im_list = []
    for j in range(len(crop_y) - 1):
        for i in range(len(crop_x) - 1):
            im_list.append(image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
    return im_list

def get_random_data(annotation_line, num_class, is_train, root='/mnt/sde/clf8113/datasets/CUB_200_2011'):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    output_size = (448, 448, 3)
    crop_num = [7, 7]
    # try:
    img_path = os.path.join(root, line[0])
    img = Image.open(img_path)
    img = img.convert('RGB')
    gt = int(line[1])

    if is_train:
        img_unswap = unswap(img)
        image_unswap_list = crop_image(img_unswap, crop_num)
        swap_law1 = [(i - 24) / 49 for i in range(crop_num[0] * crop_num[1])]

        img_swap = swap(img_unswap)
        image_swap_list = crop_image(img_swap, crop_num)
        unswap_stats = [sum(ImageStat.Stat(im).mean) for im in image_unswap_list]
        swap_stats = [sum(ImageStat.Stat(im).mean) for im in image_swap_list]
        swap_law2 = []
        for swap_im in swap_stats:
            distance = [abs(swap_im - unswap_im) for unswap_im in unswap_stats]
            index = distance.index(min(distance))
            swap_law2.append((index - 24) / 49)
        img_unswap = totensor(img_unswap)
        img_swap = totensor(img_swap)
        # label = gt
        label_swap = gt + num_class
        label = to_categorical([gt], num_class)[0]
        label_s = to_categorical([1], 2)[0]#to_categorical([gt], num_class*2)[0]
        label_swap = to_categorical([0], 2)[0]#to_categorical([label_swap], num_class*2)[0]
        return [img_unswap, img_swap, label, label_s, label_swap, swap_law1, swap_law2]
    else:
        img_unswap = noswap(img)
        img_unswap = totensor(img_unswap)
        swap_law1 = [(i - 24) / 49 for i in range(crop_num[0] * crop_num[1])]
        img_swap = img_unswap
        # label = gt
        label_swap = gt
        label = to_categorical([gt], num_class)[0]
        label_s = to_categorical([1], 2)[0]#to_categorical([gt], num_class * 2)[0]
        swap_law2 = [(i - 24) / 49 for i in range(crop_num[0] * crop_num[1])]
        return [img_unswap, label, label_s, swap_law1]

def data_generator(annotation_lines, batch_size, num_class, is_train):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    b = 0
    while True:
        if b == 0:
            imgs = []
            label = []
            label_adv = []
            swap_law = []
        if i == 0:
            np.random.shuffle(annotation_lines)
        i = (i + 1) % n
        try:
            sample = get_random_data(annotation_lines[i], num_class=num_class, is_train=is_train)
            if is_train:
                imgs.append(sample[0])
                imgs.append(sample[1])
                label.append(sample[2])
                label.append(sample[2])
                label_adv.append(sample[3])
                label_adv.append(sample[4])
                swap_law.append(sample[5])
                swap_law.append(sample[6])
            else:
                imgs.append(sample[0])
                label.append(sample[1])
                label_adv.append(sample[2])
                swap_law.append(sample[3])
            b += 1
        except:
            print("Error processing image {}".format(annotation_lines[i]))
            continue
        if b >= batch_size:
            b = 0
            yield np.array(imgs), [np.array(label), np.array(label_adv), np.array(swap_law)]


def data_generator_wrapper(annotation_lines, batch_size, num_class=200, is_train=False):
    n = len(annotation_lines)
    annotation_lines = np.array(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, num_class, is_train)