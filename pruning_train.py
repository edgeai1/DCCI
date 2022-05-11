import argparse

import torch.distributed as dist
import torch.optim as optim

import test  # import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.prune_utils import *
from utils.utils import *
import warnings

warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    mixed_precision = False  # not installed

wdir = "weights" + os.sep  # weights dir
last = wdir + "last.pt"
best = wdir + "best.pt"
results_file = "results.txt"

# Hyperparameters (j-series, 50.5 mAP yolov3-320) evolved by @ktian08 https://github.com/ultralytics/yolov3/issues/310
hyp = {
    "giou": 1.582,  # giou loss gain
    "cls": 27.76,  # cls loss gain  (CE=~1.0, uCE=~20)
    "cls_pw": 1.446,  # cls BCELoss positive_weight
    "obj": 21.35,  # obj loss gain (*=80 for uBCE with 80 classes)
    "obj_pw": 3.941,  # obj BCELoss positive_weight
    "iou_t": 0.2635,  # iou training threshold
    "lr0": 0.002324,  # initial learning rate (SGD=1E-3, Adam=9E-5)
    "lrf": -4.0,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
    "momentum": 0.97,  # SGD momentum
    "weight_decay": 0.0004569,  # optimizer weight decay
    "fl_gamma": 0.5,  # focal loss gamma
    "hsv_h": 0.10,  # image HSV-Hue augmentation (fraction)
    "hsv_s": 0.5703,  # image HSV-Saturation augmentation (fraction)
    "hsv_v": 0.3174,  # image HSV-Value augmentation (fraction)
    "degrees": 1.113,  # image rotation (+/- deg)
    "translate": 0.06797,  # image translation (+/- fraction)
    "scale": 0.1059,  # image scale (+/- gain)
    "shear": 0.5768,
}  # image shear (+/- deg)

# Overwrite hyp with hyp*.txt (optional)
f = glob.glob("hyp*.txt")
if f:
    for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
        hyp[k] = v


def rl_train(params):
    opt = {}
    opt["accumulate"] = 2
    opt["adam"] = False
    opt["arc"] = 'defaultpw'
    opt["batch_size"] = 16
    opt["bucket"] = ''
    opt["cache_images"] = False
    opt["cfg"] = 'cfg/yolov4.cfg'
    opt["data"] = 'data/voc.data'
    opt["device"] = '0'
    opt["epochs"] = 100,
    opt["evolve"] = False
    opt["img_size"] = 416
    opt["img_weights"] = False
    opt["multi_scale"] = False
    opt["name"] = ''
    opt["nosave"] = False
    opt["notest"] = False
    opt["prebias"] = False
    opt["prune"] = 1
    opt["rect"] = False
    opt["resume"] = False
    opt["s"] = 0.001,
    opt["sr"] = False
    opt["t_cfg"] = ''
    opt["t_weights"] = ''
    opt["transfer"] = False
    opt["var"] = None
    opt["weights"] = 'weights/yolov4.weights'

    def prebias():
        # trains output bias layers for 1 epoch and creates new backbone
        if opt["prebias"]:
            rl_train()  # transfer-learn yolo biases for 1 epoch
            create_backbone(last)  # saved results as backbone.pt
            opt["weights"] = wdir + "backbone.pt"  # assign backbone
            opt["prebias"] = False  # disable prebias

    # # --cfg cfg/yolov4.cfg --data data/voc.data --weights weights/yolov4.weights --epochs 100 --batch-size 32
    opt["cfg"] = params["cfg"]
    opt["data"] = params["data"]
    opt["weights"] = params["weights"]
    opt["batch_size"] = params["batch_size"]

    opt["weights"] = last if opt["resume"] else opt["weights"]
    print(opt)
    device = torch_utils.select_device(opt["device"], apex=mixed_precision)

    from torch.utils.tensorboard import SummaryWriter

    tb_writer = SummaryWriter()
    # except:
    #     pass

    prebias()  # optional

    cfg = opt["cfg"]
    t_cfg = opt["t_cfg"]  # teacher model cfg for knowledge distillation
    data = opt["data"]
    img_size = opt["img_size"]
    epochs = (1 if opt["prebias"] else opt["epochs"])[0]  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt["batch_size"]
    accumulate = opt["accumulate"]  # effective bs = batch_size * accumulate = 16 * 4 = 64
    weights = opt["weights"]  # initial training weights
    t_weights = opt["t_weights"]  # teacher model weights

    if "pw" not in opt["arc"]:  # remove BCELoss positive weights
        hyp["cls_pw"] = 1.0
        hyp["obj_pw"] = 1.0

    # Initialize
    init_seeds()
    multi_scale = opt["multi_scale"]

    if multi_scale:
        img_sz_min = round(img_size / 32 / 1.5) + 1
        img_sz_max = round(img_size / 32 * 1.5) - 1
        img_size = img_sz_max * 32  # initiate with maximum multi_scale size
        print("Using multi-scale %g - %g" % (img_sz_min * 32, img_size))

    # Configure run
    data_dict = parse_data_cfg(data)
    train_path = data_dict["train"]
    nc = int(data_dict["classes"])  # number of classes

    # Remove previous results
    for f in glob.glob("*_batch*.jpg") + glob.glob(results_file):
        os.remove(f)

    # Initialize model
    model = Darknet(cfg, (img_size, img_size), arc=opt["arc"]).to(device)
    if t_cfg:
        t_model = Darknet(t_cfg, (img_size, img_size), arc=opt["arc"]).to(device)

    # Optimizer
    pg0, pg1 = [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if "Conv2d.weight" in k:
            pg1 += [v]  # parameter group 1 (apply weight_decay)
        else:
            pg0 += [v]  # parameter group 0

    if opt["adam"]:
        optimizer = optim.Adam(pg0, lr=hyp["lr0"])
        # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
    else:
        optimizer = optim.SGD(pg0, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True)
    optimizer.add_param_group({"params": pg1, "weight_decay": hyp["weight_decay"]})  # add pg1 with weight_decay
    del pg0, pg1

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_fitness = 0.0
    attempt_download(weights)
    if weights.endswith(".pt"):  # pytorch format
        # possible weights are 'last-step3.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
        if opt["bucket"]:
            os.system("gsutil cp gs://%s/last-step3.pt %s" % (opt["bucket"], last))  # download from bucket
        chkpt = torch.load(weights, map_location=device)

        # load model
        # if opt.transfer:
        chkpt["model"] = {
            k: v
            for k, v in chkpt["model"].items()
            if model.state_dict()[k].numel() == v.numel()
        }
        model.load_state_dict(chkpt["model"], strict=False)
        print("loaded weights from", weights, "\n")
        # else:
        #    model.load_state_dict(chkpt['model'])

        # load optimizer
        if chkpt["optimizer"] is not None:
            optimizer.load_state_dict(chkpt["optimizer"])
            best_fitness = chkpt["best_fitness"]

        # load results
        if chkpt.get("training_results") is not None:
            with open(results_file, "w") as file:
                file.write(chkpt["training_results"])  # write results.txt

        start_epoch = chkpt["epoch"] + 1
        del chkpt

    # elif weights.endswith('.pth'):
    #     model.load_state_dict(torch.load(weights))

    elif len(weights) > 0:  # darknet format
        # possible weights are 'yolov3.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
        cutoff = load_darknet_weights(model, weights)
        print("loaded weights from", weights, "\n")

    if t_cfg:
        if t_weights.endswith(".pt"):
            t_model.load_state_dict(torch.load(t_weights, map_location=device)["model"])
        elif t_weights.endswith(".weights"):
            load_darknet_weights(t_model, t_weights)
        else:
            raise Exception("pls provide proper teacher weights for knowledge distillation")
        if not mixed_precision:
            t_model.eval()
        print("<.....................using knowledge distillation.......................>")
        print("teacher model:", t_weights, "\n")

    if opt["prune"] == 1:
        CBL_idx, _, prune_idx, shortcut_idx, _ = parse_module_defs2(model.module_defs)
        if opt["sr"]:
            print("shortcut sparse training")
    elif opt["prune"] == 0:
        CBL_idx, _, prune_idx = parse_module_defs(model.module_defs)
        if opt["sr"]:
            print("normal sparse training ")

    if opt["transfer"] or opt["prebias"]:  # transfer learning edge (yolo) layers
        nf = int(
            model.module_defs[model.yolo_layers[0] - 1]["filters"]
        )  # yolo layer size (i.e. 255)

        if opt["prebias"]:
            for p in optimizer.param_groups:
                # lower param count allows more aggressive training settings: i.e. SGD ~0.1 lr0, ~0.9 momentum
                p["lr"] *= 100  # lr gain
                if p.get("momentum") is not None:  # for SGD but not Adam
                    p["momentum"] *= 0.9

        for p in model.parameters():
            if opt["prebias"] and p.numel() == nf:  # train (yolo biases)
                p.requires_grad = True
            elif opt["transfer"] and p.shape[0] == nf:  # train (yolo biases+weights)
                p.requires_grad = True
            else:  # freeze layer
                p.requires_grad = False

    # Scheduler https://github.com/ultralytics/yolov3/issues/238
    # lf = lambda x: 1 - x / epochs  # linear ramp to zero
    # lf = lambda x: 10 ** (hyp['lrf'] * x / epochs)  # exp ramp
    # lf = lambda x: 1 - 10 ** (hyp['lrf'] * (1 - x / epochs))  # inverse exp ramp
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=range(59, 70, 1), gamma=0.8)  # gradual fall to 0.1*lr0
    # if opt.sr:
    #     scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(opt.epochs * x) for x in [0.8, 0.9]], gamma=0.1)
    # else:
    #     scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(opt.epochs * x) for x in [0.8, 0.9]], gamma=0.1)
    # scheduler.last_epoch = start_epoch - 1

    def adjust_learning_rate(optimizer, gamma, epoch, iteration, epoch_size):
        """调整学习率进行warm up和学习率衰减
        """
        step_index = 0
        if epoch < 6:
            # 对开始的6个epoch进行warm up
            lr = 1e-6 + (hyp["lr0"] - 1e-6) * iteration / (epoch_size * 2)
        else:
            if epoch > opt["epochs"] * 0.7:
                # 在进行总epochs的70%时，进行以gamma的学习率衰减
                step_index = 1
            if epoch > opt["epochs"] * 0.9:
                # 在进行总epochs的90%时，进行以gamma^2的学习率衰减
                step_index = 2

            lr = hyp["lr0"] * (gamma ** (step_index))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    # # Plot lr schedule
    # y = []
    # for _ in range(epochs):
    #     scheduler.step()
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y, label='LambdaLR')
    # plt.xlabel('epoch')
    # plt.ylabel('LR')
    # plt.tight_layout()
    # plt.savefig('LR.png', dpi=300)

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        if t_cfg:
            [model, t_model], optimizer = amp.initialize(
                [model, t_model], optimizer, opt_level="O1", verbosity=1
            )
        else:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level="O1", verbosity=1
            )
    # Initialize distributed training
    if torch.cuda.device_count() > 1:
        dist.init_process_group(
            backend="nccl",  # 'distributed backend'
            init_method="tcp://127.0.0.1:9999",  # distributed training init method
            world_size=1,  # number of nodes for distributed training
            rank=0,
        )  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model)
        model.module_list = model.module.module_list
        model.yolo_layers = (
            model.module.yolo_layers
        )  # move yolo layer indices to top level

    # Dataset
    dataset = LoadImagesAndLabels(
        train_path,
        img_size,
        batch_size,
        augment=True,
        hyp=hyp,  # augmentation hyperparameters
        rect=opt["rect"],  # rectangular training
        image_weights=opt["img_weights"],
        cache_labels=True if epochs > 10 else False,
        cache_images=False if opt["prebias"] else opt["cache_images"],
    )

    # Dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=min([os.cpu_count(), batch_size, 16]),
        # num_workers=min([8, batch_size, 16]),
        shuffle=not opt["rect"],  # Shuffle=True unless rectangular training is used
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    for idx in prune_idx:
        bn_weights = gather_bn_weights(model.module_list, [idx])
        tb_writer.add_histogram(
            "before_train_perlayer_bn_weights/hist",
            bn_weights.numpy(),
            idx,
            bins="doane",
        )

    # Start training
    model.nc = nc  # attach number of classes to model
    model.arc = opt["arc"]  # attach yolo architecture
    model.hyp = hyp  # attach hyperparameters to model
    # model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    torch_utils.model_info(model, report="summary")  # 'full' or 'summary'
    nb = len(dataloader)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0,)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    print("Starting %s for %g epochs..." % ("prebias" if opt["prebias"] else "training", epochs))
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()
        # print('learning rate:',optimizer.param_groups[0]['lr'])
        print(("\n" + "%10s" * 10) % ("Epoch", "gpu_mem", "GIoU", "obj", "cls", "total", "soft", "rratio", "targets", "img_size",))

        # Freeze backbone at epoch 0, unfreeze at epoch 1 (optional)
        freeze_backbone = False
        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                if int(name.split(".")[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if epoch == 0 else True

        # Update image weights (optional)
        if dataset.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(
                dataset.labels, nc=nc, class_weights=w
            )
            dataset.indices = random.choices(
                range(dataset.n), weights=image_weights, k=dataset.n
            )  # rand weighted idx

        mloss = torch.zeros(4).to(device)  # mean losses
        msoft_target = torch.zeros(1).to(device)
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        sr_flag = get_sr_flag(epoch, opt["sr"])
        for (i, (imgs, targets, paths, _),) in (pbar):
            ni = i + nb * epoch

            lr = adjust_learning_rate(optimizer, 0.1, epoch, ni, nb)
            if i == 0:
                print("learning rate:", lr)

            imgs = imgs.to(device)
            targets = targets.to(device)

            # Multi-Scale training
            if multi_scale:
                if ni / accumulate % 10 == 0:  #  adjust (67% - 150%) every 10 batches
                    img_size = random.randrange(img_sz_min, img_sz_max + 1) * 32
                sf = img_size / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / 32.0) * 32 for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)

            # Plot images with bounding boxes
            if ni == 0:
                fname = "train_batch%g.jpg" % i
                plot_images(imgs=imgs, targets=targets, paths=paths, fname=fname)
                if tb_writer:
                    tb_writer.add_image(fname, cv2.imread(fname)[:, :, ::-1], dataformats="HWC")

            # Run model
            pred = model(imgs)

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model)
            if not torch.isfinite(loss):
                print("WARNING: non-finite loss, ending training ", loss_items)
                return results

            soft_target = 0
            reg_ratio = 0  # 表示有多少target的回归是不如老师的，这时学生会跟gt再学习
            if t_cfg:
                if mixed_precision:
                    with torch.no_grad():
                        output_t = t_model(imgs)
                else:
                    _, output_t = t_model(imgs)
                # soft_target = distillation_loss1(pred, output_t, model.nc, imgs.size(0))
                # 这里把蒸馏策略改为了二，想换回一的可以注释掉loss2，把loss1取消注释
                soft_target, reg_ratio = distillation_loss2(model, targets, pred, output_t)
                loss += soft_target

            # Scale loss by nominal batch_size of 64
            loss *= batch_size / 64

            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            idx2mask = None
            # if opt.sr and opt.prune==1 and epoch > opt.epochs * 0.5:
            #     idx2mask = get_mask2(model, prune_idx, 0.85)

            BNOptimizer.updateBN(sr_flag, model.module_list, opt["s"], prune_idx, epoch, idx2mask, opt)

            # Accumulate gradient for x batches before optimizing
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            msoft_target = (msoft_target * i + soft_target) / (i + 1)
            mem = (torch.cuda.memory_cached() / 1e9 if torch.cuda.is_available() else 0)  # (GB)
            s = ("%10s" * 2 + "%10.3g" * 8) % (
                "%g/%g" % (epoch, epochs - 1),
                "%.3gG" % mem,
                *mloss,
                msoft_target,
                reg_ratio,
                len(targets),
                img_size,
            )
            pbar.set_description(s)


        # Update scheduler
        # scheduler.step()

        # Process epoch results
        final_epoch = epoch + 1 == epochs
        if opt["prebias"]:
            print_model_biases(model)
        else:
            # Calculate mAP (always test final epoch, skip first 10 if opt.nosave)
            if not (opt["notest"] or (opt["nosave"] and epoch < 10)) or final_epoch:
                with torch.no_grad():
                    results, maps = test.test(
                        cfg,
                        data,
                        batch_size=batch_size,
                        img_size=opt["img_size"],
                        model=model,
                        conf_thres=0.001
                        if final_epoch and epoch > 0
                        else 0.1,  # 0.1 for speed
                        save_json=final_epoch and epoch > 0 and "coco.data" in data, )

        # Write epoch results
        with open(results_file, "a") as f:
            f.write(s + "%10.3g" * 7 % results + "\n")  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

        # Write Tensorboard results
        if tb_writer:
            x = list(mloss) + list(results) + [msoft_target]
            titles = ["GIoU", "Objectness", "Classification", "Train loss", "Precision", "Recall", "mAP", "F1", "val GIoU", "val Objectness", "val Classification", "soft_loss", ]
            for xi, title in zip(x, titles):
                tb_writer.add_scalar(title, xi, epoch)
            bn_weights = gather_bn_weights(model.module_list, prune_idx)
            tb_writer.add_histogram("bn_weights/hist", bn_weights.numpy(), epoch, bins="doane")

        # Update best mAP
        fitness = results[2]  # mAP
        if fitness > best_fitness:
            best_fitness = fitness

        # Save training results
        save = (not opt["nosave"]) or (final_epoch and not opt["evolve"]) or opt["prebias"]
        if save:
            with open(results_file, "r") as f:
                # Create checkpoint
                chkpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "training_results": f.read(),
                    "model": model.module.state_dict() if type(model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                    "optimizer": None if final_epoch else optimizer.state_dict(),
                }

            # Save last checkpoint
            torch.save(chkpt, last)
            if opt["bucket"] and not opt["prebias"]:
                os.system("gsutil cp %s gs://%s" % (last, opt["bucket"]))  # upload to bucket

            if best_fitness == fitness:
                torch.save(chkpt, best)
            if epoch > 0 and epoch % 10 == 0:
                torch.save(chkpt, wdir + "backup%g.pt" % epoch)

            del chkpt

    for idx in prune_idx:
        bn_weights = gather_bn_weights(model.module_list, [idx])
        tb_writer.add_histogram("after_train_perlayer_bn_weights/hist", bn_weights.numpy(), idx, bins="doane", )

    # end training
    if len(opt["name"]):
        os.rename("results.txt", "results_%s.txt" % opt["name"])
    plot_results()  # save as results.png
    print("%g epochs completed in %.3f hours.\n" % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()
    return best_fitness
