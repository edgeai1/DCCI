import argparse
import json
import warnings

from torch.utils.data import DataLoader
from logsdir.discriminator1 import res
from partition_datasets import load_light_model, load_heavy_model
from models import *
from utils.datasets import *
from utils.utils import *
import ntpath

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"

total_hard_count = 0
for key, value in res.items():
    if res[key] == 1:
        total_hard_count += 1
total_conf = total_hard_count / 4952
print(total_conf)


# test e2e mAP/RmAP
def test(cfg, data, weights=None, batch_size=16, img_size=416, iou_thres=0.5, conf_thres=0.001, nms_thres=0.5, save_json=False):
    verbose = True
    # Initialize/load model and set device
    light_model = load_light_model()
    heavy_model = load_heavy_model()

    # Configure run
    data = parse_data_cfg(data)
    nc = int(data["classes"])  # number of classes
    test_path = data["valid"]  # path to test images
    names = load_classes(data["names"])  # class names

    # Dataloader
    dataset = LoadImagesAndLabels(test_path, img_size, 1)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    seen = 0
    coco91class = coco80_to_coco91_class()
    s = ("%20s" + "%10s" * 6) % ("Class", "Images", "Targets", "P", "R", "mAP", "F1")
    p, r, f1, mp, mr, map, mf1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3)
    jdict, stats, ap, ap_class = [], [], [], []
    hard_count = 0
    thresh_conf = 0.38
    need_simple_ratio = None
    simple_count = 0
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        targets = targets.to(device)
        imgs = imgs.to(device)
        _, _, height, width = imgs.shape  # batch size, channels, height, width

        # Plot images with bounding boxes

        if batch_i == 0 and not os.path.exists("test_batch0.jpg"):
            plot_images(imgs=imgs, targets=targets, paths=paths, fname="test_batch0.jpg")
        image_name = ntpath.basename(paths[0])

        if total_conf < thresh_conf:
            need_simple_ratio = thresh_conf - total_conf

        if hard_count / 4952 < thresh_conf and res[image_name] == 1:
            inf_out, train_out = heavy_model(imgs)  # inference and training outputs
            hard_count += 1
        elif need_simple_ratio and simple_count / 4952 < need_simple_ratio:
            inf_out, train_out = heavy_model(imgs)  # inference and training outputs
            hard_count += 1
            simple_count += 1
            # print(image_name)
        else:
            inf_out, train_out = light_model(imgs)  # inference and training outputs

        # Compute loss
        if hasattr(light_model, "hyp"):  # if model has loss hyperparameters
            loss += compute_loss(train_out, targets, light_model)[1][:3].cpu()  # GIoU, obj, cls
        # Run NMS
        output = non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres)

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split("_")[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(imgs[si].shape[1:], box, shapes[si])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for di, d in enumerate(pred):
                    jdict.append(
                        {
                            "image_id": image_id,
                            "category_id": coco91class[int(d[6])],
                            "bbox": [floatn(x, 3) for x in box[di]],
                            "score": floatn(d[4], 5),
                        }
                    )

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Assign all predictions as incorrect
            correct = [0] * len(pred)
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                tbox[:, [0, 2]] *= width
                tbox[:, [1, 3]] *= height

                # Search for correct predictions
                for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                    # Break if all targets already located in image
                    if len(detected) == nl:
                        break

                    # Continue if predicted class not among image classes
                    if pcls.item() not in tcls:
                        continue

                    # Best iou, index between pred and targets
                    m = (pcls == tcls_tensor).nonzero().view(-1)
                    iou, bi = bbox_iou(pbox, tbox[m]).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if (
                            iou > iou_thres and m[bi] not in detected
                    ):  # and pcls == tcls[bi]:
                        correct[i] = 1
                        detected.append(m[bi])

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))
    print("hard count number: {}, simple count number: {}".format(hard_count, 4952 - hard_count))
    # torch.save(stats, "./simple_map.pkl")
    if os.path.exists("./simple_map.pkl"):
        simple_map = torch.load("./simple_map.pkl")
        for simple in simple_map:
            stats.append(simple)
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy

    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = "%20s" + "%10.3g" * 6  # print format
    print(pf % ("all", seen, nt.sum(), mp, mr, map, mf1))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Save JSON
    if save_json and map and len(jdict):
        try:
            imgIds = [int(Path(x).stem.split("_")[-1]) for x in dataset.img_files]
            with open("results.json", "w") as file:
                json.dump(jdict, file)

            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            cocoGt = COCO(
                "../coco/annotations/instances_val2014.json"
            )  # initialize COCO ground truth api
            cocoDt = cocoGt.loadRes("results.json")  # initialize COCO pred api

            cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
            cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            map = cocoEval.stats[1]  # update mAP to pycocotools mAP
        except:
            print("Test Error!!!")

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map, mf1, *(loss / len(dataloader)).tolist()), maps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="test.py")
    parser.add_argument("--cfg", type=str, default="cfg/yolov3-spp.cfg", help="cfg file path")
    parser.add_argument("--data", type=str, default="data/coco.data", help="coco.data file path")
    parser.add_argument("--weights", type=str, default="weights/yolov3-spp.weights", help="path to weights file")
    parser.add_argument("--batch-size", type=int, default=16, help="size of each image batch")
    parser.add_argument("--img-size", type=int, default=416, help="inference size (pixels)")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms-thres", type=float, default=0.5, help="iou threshold for non-maximum suppression")
    parser.add_argument("--save-json", action="store_true", help="save a cocoapi-compatible JSON results file", )
    parser.add_argument("--device", default="", help="device id (i.e. 0 or 0,1) or cpu")
    opt = parser.parse_args()
    print(opt)
    # --cfg cfg/yolov4_light_voc.cfg --data data/voc.data  --weights weights/yolov4/voc/best.pt --batch-size 32       light_weight model
    # --cfg cfg/yolov4.cfg --data data/voc.data  --weights weights/yolov4/voc/yolo-v4-best-step1.pt --batch-size 32   heavy_weight model
    with torch.no_grad():
        test(opt.cfg, opt.data, opt.weights, opt.batch_size, opt.img_size, opt.iou_thres, opt.conf_thres, opt.nms_thres, opt.save_json)
