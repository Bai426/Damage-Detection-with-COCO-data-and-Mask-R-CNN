import warnings

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector


def init_detector(config, checkpoint=None, device='cuda:0'):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_detector(model, img):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
        # print(result)
    return result


async def async_inference_detector(model, img):
    """Async inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        Awaitable detection results.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]

    # We don't restore `torch.is_grad_enabled()` value during concurrent
    # inference since execution can overlap
    torch.set_grad_enabled(False)
    result = await model.aforward_test(rescale=True, **data)
    return result


# TODO: merge this method with the one in BaseDetector
def show_result(img,
                result,
                class_names,
                score_thr=0.3,
                wait_time=0,
                show=True,
                out_file=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        wait_time (int): Value of waitKey param.
        show (bool, optional): Whether to show the image with opencv or not.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    ### deal with coordinates of bounding boxes  ###
    # a = bboxes[0]
    # b = bboxes[1]
    # center_bboxes1 = np.array([(a[2] + a[0]) / 2., (a[3] + a[1]) / 2.])
    # center_bboxes2 = np.array([(b[2] + b[0]) / 2., (b[3] + b[1]) / 2.])

    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    if labels[0].size == 0:      ### for this cannot predict
        # Num_unpredict = 1      ####
        return img     #####

    labels = np.concatenate(labels)
    # if aedede !=0 or aedede !=1:
    #     return img
    print(max(labels))
    ###  modification end
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        np.random.seed(42)
        color_masks = [
            np.random.randint(0, 256, (1, 3), dtype=np.uint8)     ### I change np.random.randint(0, 256, (1, 3) to np.random.randint(100, 150, (1, 3)
            # np.array([255, 0, 255])     ## blue mask
            for _ in range(max(labels) + 1)
        ]
        # color_masks = int(color_masks/2)
        print(color_masks)
        for i in inds:
            i = int(i)
            color_mask = color_masks[labels[i]]
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False
    # draw bounding boxes
    mmcv.imshow_det_bboxes(
        img,
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        show=show,
        wait_time=wait_time,
        out_file=out_file)
    if not (show or out_file):
        # Num_unpredict = 0              ####
        return img             #####


def show_result_pyplot(img,
                       result,
                       class_names,
                       score_thr=0.3,
                       fig_size=(15, 10)):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    img = show_result(
        img, result, class_names, score_thr=score_thr, show=False)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))

############## ADD NEW LINES FOR ACQUIRING coordinates of bounding boxes center for each class.
##### It should be pointed out that multiple bbox for one class is possible.
#
def show_coordinate(img_name, img,
                result,
                class_names,
                score_thr=0.3):
    """Find the coordinates for each bounding box center.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        wait_time (int): Value of waitKey param.
        show (bool, optional): Whether to show the image with opencv or not.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)

    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    if labels[0].size == 0:      ### for this cannot predict
        # Num_unpredict = 1      ####
        return img     #####

    labels = np.concatenate(labels)

    ### new lines begin
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    import csv
    data_file = open('/home/bys2058/NSF_inside-camera_5th_beam_5_1_22/5th_column_removal_beam_deflection_5_3_22New.csv', 'a+', newline='')     #######     need to change the file name   #######
    csv_writer = csv.writer(data_file)

    coord_inf = []
    idNum = 0              ## order of the bouding boxes, from 0.
    crop_path = os.path.join('/media/bys2058/Elements/NSF project data/align_image_affine/d/', 'crop_5_3_22New')
    if not os.path.exists(crop_path):
        os.makedirs(crop_path)
    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)   ### use integer to crop the original image, instead of float value.  ####
        ### Get some information about the bounding boxes: left_top = (bbox_int[0], bbox_int[1]); right_bottom = (bbox_int[2], bbox_int[3])   ####
        x_corner = int(bbox[0])              ### left_top = (bbox[0], bbox[1]), it should be a float type, but in order to be fit for the cropped image, it should be an integer. changed on 4/18/2021
        y_corner = int(bbox[1])              ### left_top = (bbox[0], bbox[1]) ,it should be an integer. Changed on 4/18/2021
        #right_bottom = (bbox_int[2], bbox_int[3])
        #### some modification for coordinates of bboxes  ####      DYNAMIC DISPLACEMENT       #####
        label_name = label  ## store the name of class
        x_center = (bbox[0] + bbox[2]) / 2.  ## get the center of bbox in Y
        ### crop image in each bounding box. For example: new_img=image[y:y+h,x:x+w]
        new_img = img[bbox_int[1]:bbox_int[3], bbox_int[0]:bbox_int[2]]
        img_path = os.path.join(crop_path, str(label_name))
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        img_path1 = os.path.join(img_path, str(img_name))
        cv2.imwrite(img_path1 + '.jpg', new_img)



        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        label_name = label_text  ## store the name of class
        x_center = (bbox[0] + bbox[2]) / 2.  ## get the center of bbox in Y axis with float digit
        y_center = (bbox[1] + bbox[3]) / 2.  ## get the center of bbox in Y axis with float digit
        csv_writer.writerow([img_name, label_name, x_corner, y_corner])  # store class name and left corner of bbox to a csv file
        coord_inf.append([img_name, label_name, x_center, y_center])
    return coord_inf


###
from PIL import Image, ImageOps
import os, sys
import glob
import matplotlib
import cv2
import mmcv
import re

def main():
    config_fname = '/home/bys2058/mmdetection/configs/hrnet/mask_rcnn_hrnetv2p_NSF_Cannon_Garage_5th_column_5_1_2022.py'
    checkpoint_file = '/home/bys2058/work_dirs/mask_rcnn_hrnetv2p_NSF_Cannon_Garage_5th_column_5_1_2022/latest.pth'

    score_thr = 0.9

    # build the model from a config file and a checkpoint file
    model = init_detector(config_fname, checkpoint_file)

    # test a single image and show the results
    # img = '/home/bys2058/mmdetection/data/sezen204/DSC_0640.JPG'
    root_path = os.path.join('/media/bys2058/Elements/NSF project data/align_image_affine/', 'd')
    path = os.path.join(root_path, 'beam_deflectionNew_5_3')
    if not os.path.exists(path):
        os.makedirs(path)

    image_files = [f for f in os.listdir(root_path) if f.endswith('.jpg')]
    # image_files = glob.glob("/home/bys2058/matching/2020_11_12_Centra_Camp_bridge/cent_cap_bri_1112_jpg/crop/0/*.jpg")
    image_files.sort(key=lambda f: int(re.sub('\D', '', f)))
    # for file_name in os.listdir(root_path):
    count = 0
    bbox_center = []
    for j in range(len(image_files)):
        model = init_detector(config_fname, checkpoint_file)
        img = os.path.join(root_path, image_files[j])
        save_prediction = os.path.join(path, image_files[j])
        print(img, j)
        result = inference_detector(model, img)     #  for not ms-rcnn
        # result = inference_detector(model, img)    ### for prediction of ms_rcnn
        # result = (result[0], result[1][0])   ###  for prediction of ms_rcnn_x101_64x4d_pafpn_1.py, modified like this
        #     print(save_prediction, j)
        show_result(img, result, model.CLASSES,
                    score_thr=score_thr, out_file='/home/bys2058/result1.jpg')
        N_img, e = os.path.splitext(image_files[j])       ###  split the file name 426.jpg as 426 and jpg, so the frame order is known
        tem = show_coordinate(N_img, img, result, model.CLASSES, score_thr=score_thr)
        bbox_center.append(tem)     ## save bbox center
        img_pred = cv2.imread('/home/bys2058/result1.jpg')
        if result[0][0].size == 0:                ## prediction is nothing, return the original image.
            count = count + 1
            img_pred = cv2.imread(img)

        cv2.imwrite(save_prediction, img_pred)
        # if j == 0:
        #     break


    print("No Prediction", count)


if __name__ == '__main__':
    main()
