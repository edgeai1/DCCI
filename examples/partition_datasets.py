import argparse
import ntpath
import os
import shutil

import cv2
import numpy as np
import torch

from models import Darknet
from utils.datasets import letterbox
from utils.utils import non_max_suppression, scale_coords, bbox_iou
from utils.utils import xywh2xyxy

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_light_model(cfg="../cfg/yolov4_light_voc.cfg",
                     img_size=416, weights='../weights/light_weight_model.pt'):
    heavy_weight_model = Darknet(cfg, img_size)
    heavy_weight_model.load_state_dict(torch.load(weights, map_location=device)["model"])
    heavy_weight_model.to(device).eval()
    return heavy_weight_model


def load_heavy_model(cfg="../cfg/yolov4.cfg", img_size=416,
                     weights='../weights/heavy_weight_model.pt'):
    heavy_weight_model = Darknet(cfg, img_size)
    heavy_weight_model.load_state_dict(torch.load(weights, map_location=device)["model"])
    heavy_weight_model.to(device).eval()
    return heavy_weight_model


def transfer_image(img, img_size=416, half=False):
    img = letterbox(img, new_shape=img_size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float16 if half else np.float32)  # uint8 to fp16/fp32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = torch.from_numpy(img).to(device)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img.to(device)


def execution_train_data(dataset_path):
    heavy_model = load_heavy_model()
    light_model = load_light_model()
    images = os.listdir(dataset_path)

    heavy_model_res = {}
    light_model_res = {}
    for idx, image in enumerate(images):
        image_path = os.path.join(dataset_path, image)
        im0 = cv2.imread(image_path)
        dw = 1.0 / im0.shape[1]
        dh = 1.0 / im0.shape[0]

        img = transfer_image(im0)
        img = img.to(device)
        with torch.no_grad():
            heavy_pred, _ = heavy_model(img)
            light_pred, _ = light_model(img)

        heavy_pred = non_max_suppression(heavy_pred, 0.5, 0.5)
        if heavy_pred[0] is not None and len(heavy_pred[0]):
            heavy_pred[0][:, :4] = scale_coords(img.shape[2:], heavy_pred[0][:, :4], im0.shape).round()

            heavy_pred[0][:, 0] = heavy_pred[0][:, 0] * dw
            heavy_pred[0][:, 2] = heavy_pred[0][:, 2] * dw
            heavy_pred[0][:, 1] = heavy_pred[0][:, 1] * dh
            heavy_pred[0][:, 3] = heavy_pred[0][:, 3] * dh

        light_pred = non_max_suppression(light_pred, 0.5, 0.5)
        if light_pred[0] is not None and len(light_pred[0]):
            light_pred[0][:, :4] = scale_coords(img.shape[2:], light_pred[0][:, :4], im0.shape).round()
            light_pred[0][:, 0] = light_pred[0][:, 0] * dw
            light_pred[0][:, 2] = light_pred[0][:, 2] * dw
            light_pred[0][:, 1] = light_pred[0][:, 1] * dh
            light_pred[0][:, 3] = light_pred[0][:, 3] * dh

        heavy_model_res[image] = heavy_pred
        light_model_res[image] = light_pred

    return heavy_model_res, light_model_res


def load_gt_label(label_path, train_path):
    # 6 0.507 0.551051051051051 0.39 0.5195195195195195   class,x,y,w,h. Normalized
    labels = []
    with open(train_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            file_name = ntpath.basename(line).split(".")[0]
            labels.append(file_name + ".txt")

    gt_labels = {}
    for label in labels:
        path = os.path.join(label_path, label)
        with open(path, "r") as f:
            lines = f.readlines()
            bbs = []
            for line in lines:
                bbox = line.replace("\n", "").split(" ")[1:]
                res = xywh2xyxy(np.array([[float(x) for x in bbox]]))
                bb = [float(x) for x in res[0]]
                bb.append(int(line.split(" ")[0]))
                bbs.append(bb)
            gt_labels[label] = bbs  #
    return gt_labels


def compare_gt_heavy_pred(gt_labels, heavy_preds, thresh_conf):
    results = {}  # 6 0.507 0.551051051051051 0.39 0.5195195195195195
    for img_name, gt_label in gt_labels.items():
        # gt_label= [[0.312, 0.2912912912912913, 0.702, 0.8108108108108107, '6']]
        object_num = len(gt_label)
        heavy_pred = heavy_preds[img_name.split(".")[0] + ".jpg"][0]  # [tensor([[0.45646, 0.20600, 1.06306, 0.53800, 0.91220, 0.99559, 6.00000]], device='cuda:0')]
        if heavy_pred is None:
            results[img_name] = 1
            continue
        heavy_pred = np.array(heavy_pred.cpu())
        heavy_pred = [list(l) for l in heavy_pred]

        temp = np.array([False] * object_num)

        for index, obj in enumerate(gt_label):
            gt_x1, gt_y1, gt_x2, gt_y2, gt_cls = obj[0], obj[1], obj[2], obj[3], int(obj[4])  # Gt value

            pred_iou = 0.0
            for idx, pred in enumerate(heavy_pred):
                pred_x1, pred_y1, pred_x2, pred_y2, pred_conf, pred_cls = pred[0], pred[1], pred[2], pred[3], pred[4], int(pred[6])  # Heavy_weight model predict value
                iou = bbox_iou(torch.tensor([gt_x1, gt_y1, gt_x2, gt_y2]), torch.tensor([pred_x1, pred_y1, pred_x2, pred_y2])).item()

                if iou >= thresh_conf and gt_cls == pred_cls:
                    pred_iou = max(pred_iou, iou)
                    pred_index = idx
                    temp[index] = True
                    heavy_pred.pop(pred_index)
        if np.sum(temp) == object_num:
            results[img_name] = 0
        else:
            results[img_name] = 1

    return results


def compare_heavy_ligt_pred(heavy_preds, light_preds, thresh_conf):
    results = {}
    img_names = heavy_preds.keys()
    for img_name in img_names:
        heavy_pred = heavy_preds[img_name][0]
        if heavy_pred is None:
            results[img_name] = 1
            continue
        heavy_pred = np.array(heavy_pred.cpu())
        heavy_pred = [list(l) for l in heavy_pred]
        object_num = len(heavy_pred)
        light_pred = light_preds[img_name][0]
        if light_pred is None:
            results[img_name] = 1
            continue
        light_pred = np.array(light_pred.cpu())
        light_pred = [list(l) for l in light_pred]

        temp = np.array([False] * len(heavy_pred))
        for index, obj in enumerate(heavy_pred):
            gt_x1, gt_y1, gt_x2, gt_y2, gt_cls = obj[0], obj[1], obj[2], obj[3], int(obj[6])

            pred_iou = 0.0
            for idx, pred in enumerate(light_pred):
                pred_x1, pred_y1, pred_x2, pred_y2, pred_cls = pred[0], pred[1], pred[2], pred[3], int(pred[6])
                iou = bbox_iou(torch.tensor([gt_x1, gt_y1, gt_x2, gt_y2]), torch.tensor([pred_x1, pred_y1, pred_x2, pred_y2])).item()

                if iou >= thresh_conf and gt_cls == pred_cls:
                    pred_iou = max(pred_iou, iou)
                    pred_index = idx
                    temp[index] = True
                    light_pred.pop(pred_index)
        if np.sum(temp) == object_num:
            results[img_name] = 0
        else:
            results[img_name] = 1
    return results


def combine_res(train_heavy_gt, train_heavy_light, dataset_type):
    img_names = train_heavy_gt.keys()
    results = {}
    for img_name in img_names:
        train_heavy_gt_per = train_heavy_gt[img_name]
        train_heavy_light_per = train_heavy_light[img_name.split(".")[0] + ".jpg"]
        if train_heavy_gt_per == train_heavy_light_per:
            results[img_name] = train_heavy_gt_per
        else:
            results[img_name] = 1
    hard_count = 0
    simple_count = 0
    for v in results.values():
        if v == 0:
            simple_count += 1
        else:
            hard_count += 1
    print("dataset_type：{}, hard_count: {}, simple_count: {}".format(dataset_type, hard_count, simple_count))
    return results


def copy_to_image(source_path, target_path, source_labels_path, target_labels_path, final_res):
    if not os.path.exists(target_path):
        os.makedirs(target_path, exist_ok=True)

    if not os.path.exists(target_labels_path):
        os.makedirs(target_labels_path, exist_ok=True)

    images = os.listdir(source_path)

    simple_file = open(target_path + '/simple.txt', 'w+')
    hard_file = open(target_path + '/hard.txt', 'w+')

    img_old_path = source_path + "/"
    img_new_path = target_path + "/images/"
    os.makedirs(img_new_path, exist_ok=True)
    for image in images:
        if os.path.exists(source_labels_path + "/" + image.split(".")[0] + ".txt"):
            shutil.copyfile(img_old_path + image, img_new_path + image)
            shutil.copyfile(source_labels_path + "/" + image.split(".")[0] + ".txt", target_labels_path + "/" + image.split(".")[0] + ".txt")
            txt_image = image.split(".")[0] + ".txt"
            if txt_image in final_res.keys() and final_res[txt_image] == 0:
                simple_file.write(img_new_path + image + "\n")
            else:
                hard_file.write(img_new_path + image + "\n")
    simple_file.close()
    hard_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_path", type=str, default="")
    parser.add_argument("--test_dataset_path", type=str, default="")
    parser.add_argument("--heavy_model_train_res_path", type=str, default="")
    parser.add_argument("--light_model_train_res_path", type=str, default="")
    parser.add_argument("--heavy_model_test_res_path", type=str, default="")
    parser.add_argument("--light_model_test_res_path", type=str, default="")
    parser.add_argument("--label_path", type=str, default="")
    parser.add_argument("--train_path", type=str, default="")
    parser.add_argument("--test_path", type=str, default="")
    parser.add_argument("--train_source_path", type=str, default="")
    parser.add_argument("--train_target_path", type=str, default="")
    parser.add_argument("--train_source_labels_path", type=str, default="")
    parser.add_argument("--train_target_labels_path", type=str, default="")
    parser.add_argument("--test_source_path", type=str, default="")
    parser.add_argument("--test_target_path", type=str, default="")
    parser.add_argument("--test_source_labels_path", type=str, default="")
    parser.add_argument("--test_target_labels_path", type=str, default="")

    opt = parser.parse_args()

    # train_dataset_path = "datasets/train_images"
    train_dataset_path = opt.train_dataset_path
    # test_dataset_path = "datasets/images"
    test_dataset_path = opt.test_dataset_path

    # Save the training data set results
    heavy_model_train_res, light_model_train_res = execution_train_data(train_dataset_path)
    torch.save(heavy_model_train_res, opt.heavy_model_train_res_path)
    torch.save(light_model_train_res, opt.light_model_train_res_path)
    # heavy_model_train_res = torch.load( opt.heavy_model_train_res_path)
    # light_model_train_res = torch.load( opt.light_model_train_res_path)

    # Save the test data set results
    heavy_model_test_res, light_model_test_res = execution_train_data(test_dataset_path)
    torch.save(heavy_model_test_res, opt.heavy_model_test_res_path)
    torch.save(light_model_test_res, opt.light_model_test_res_path)
    # heavy_model_test_res = torch.load(opt.heavy_model_test_res_path)
    # light_model_test_res = torch.load(opt.light_model_test_res_path)

    label_path = opt.label_path
    # train_path = "datasets/voc0712/train.txt"
    train_path = opt.train_path
    train_gt_labels = load_gt_label(label_path, train_path)

    # test_path = datasets/voc0712/2007_test.txt"
    test_path = opt.test_path
    test_gt_labels = load_gt_label(label_path, test_path)

    for thresh_conf in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]:
        # The heavy_weight model detection results are compared with the GT value, and the ones that are not checked are hard cases
        train_heavy_gt = compare_gt_heavy_pred(train_gt_labels, heavy_model_train_res, thresh_conf)
        test_heavy_gt = compare_gt_heavy_pred(test_gt_labels, heavy_model_test_res, thresh_conf)

        # If the detection value of the heavy_weight model is compared with that of the light_weight model, if it is inconsistent and set as a hard case.
        train_heavy_light = compare_heavy_ligt_pred(heavy_model_train_res, light_model_train_res, thresh_conf)
        test_heavy_light = compare_heavy_ligt_pred(heavy_model_test_res, light_model_test_res, thresh_conf)

        final_train_res = combine_res(train_heavy_gt, train_heavy_light, "training dataset")
        final_test_res = combine_res(test_heavy_gt, test_heavy_light, "testing dataset")

        # save labels
        # torch.save(final_train_res, "./labels/train_label_{}.pkl".format(thresh_conf))
        # torch.save(final_test_res, "./labels/test_label_{}.pkl".format(thresh_conf))

        # Copy the training set to the corresponding folder
        # train_source_path = "datasets/train_images"
        # train_target_path = "datasets/hard_simple_{}/train".format(thresh_conf)
        # train_source_labels_path = "datasets/voc0712/labels"
        # train_target_labels_path = "datasets/hard_simple_{}/train/labels/".format(thresh_conf)
        train_source_path = opt.train_source_path
        train_target_path = opt.train_target_path
        train_source_labels_path = opt.train_source_labels_path
        train_target_labels_path = (opt.train_target_labels_path + "_{}").format(thresh_conf)
        copy_to_image(train_source_path, train_target_path, train_source_labels_path, train_target_labels_path, final_train_res)

        # Copy the test set to the corresponding folder
        # test_source_path = "datasets/images"
        # test_target_path = "datasets/hard_simple_{}/test".format(thresh_conf)
        # test_source_labels_path = "datasets/voc0712/labels"
        # test_target_labels_path = "datasets/hard_simple_{}/test/labels/".format(thresh_conf)
        test_source_path = opt.test_source_path
        test_target_path = opt.test_target_path
        test_source_labels_path = opt.test_source_labels_path
        test_target_labels_path = (opt.test_target_labels_path + "_{}").format(thresh_conf)
        copy_to_image(test_source_path, test_target_path, test_source_labels_path, test_target_labels_path, final_test_res)
