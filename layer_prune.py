import argparse
import logging
import warnings
import os
from models import *
from pruning_train import rl_train
from rl import *
from test import test
from utils.prune_utils import *

warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

logging.basicConfig(level=logging.ERROR)


def get_controller():
    # ----HYPERPARAMETERS----
    # Initialize controller based on mode
    skipSupport = False
    num_layers = 2
    lookup = [0.25, .25, .25, .25, .8, .25, .25, 0.3, .5, .5, 1.]
    num_hidden = 30
    num_input = 7 if skipSupport else 5
    lookup = [0.25, .5, .5, .5, .5, .5, .6, .7, .8, .9, 1.]  # Used for shrinkage only
    num_output = 2
    lr = 0.003
    controller = Controller(None, num_input, num_output, num_hidden, num_layers, lr=lr, skipSupport=skipSupport, kwargs={'lookup': lookup})
    # architecture = Architecture("removal", model, datasetInputTensor, args.dataset, baseline_acc=baseline_acc, lookup=lookup)
    return controller


# --cfg cfg/yolov4.cfg --data data/voc.data --weights weights/yolov4/voc/yolo-v4-best-step1.pt --rl_epochs 80 --batch-size 32
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--data", type=str, default="")
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--img_size", type=int, default=416)
    parser.add_argument("--rl_epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)

    opt = parser.parse_args()

    img_size = opt.img_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.cfg, (img_size, img_size)).to(device)

    if opt.weights.endswith(".pt"):
        model.load_state_dict(torch.load(opt.weights, map_location=device)["model"])
    else:
        load_darknet_weights(model, opt.weights)

    eval_model = lambda model: test(model=model, cfg=opt.cfg, data=opt.data, batch_size=32, img_size=img_size)
    get_parameters = lambda model: sum([param.nelement() for param in model.parameters()])

    with torch.no_grad():
        print("\nHeavy-weight model mAP:")
        heavy_weight_model_metric = eval_model(model)
    heavy_weight_model_parameters = get_parameters(model)

    for rl_epoch in range(opt.rl_epochs):
        CBL_idx, Conv_idx, shortcut_idx = parse_module_defs4(model.module_defs)

        # --------------RL pruning-----------------------
        controller = get_controller()
        layers = get_layers(model.module_list, shortcut_idx)
        actions = controller.rolloutActions(layers)
        sorted_index_thre = np.array(shortcut_idx)[np.array(actions[0]).astype(bool)]

        prune_shortcuts = [int(x) for x in sorted_index_thre]

        index_all = list(range(len(model.module_defs)))
        index_prune = []
        for idx in prune_shortcuts:
            index_prune.extend([idx - 1, idx, idx + 1])
        index_remain = [idx for idx in index_all if idx not in index_prune]


        def prune_and_eval(model, prune_shortcuts=[]):
            model_copy = deepcopy(model)
            for idx in prune_shortcuts:
                for i in [idx, idx - 1]:
                    bn_module = model_copy.module_list[i][1]

                    mask = torch.zeros(bn_module.weight.data.shape[0]).cuda()
                    bn_module.weight.data.mul_(mask)

            with torch.no_grad():
                mAP = eval_model(model_copy)[0][2]
            print(f"After RL_pruning mAP: {mAP:.4f}")


        def obtain_filters_mask(model, CBL_idx, prune_shortcuts):

            filters_mask = []
            for idx in CBL_idx:
                bn_module = model.module_list[idx][1]
                mask = np.ones(bn_module.weight.data.shape[0], dtype="float32")
                filters_mask.append(mask.copy())
            CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}
            for idx in prune_shortcuts:
                for i in [idx, idx - 1]:
                    bn_module = model.module_list[i][1]
                    mask = np.zeros(bn_module.weight.data.shape[0], dtype="float32")
                    CBLidx2mask[i] = mask.copy()
            return CBLidx2mask


        CBLidx2mask = obtain_filters_mask(model, CBL_idx, prune_shortcuts)

        pruned_model = prune_model_keep_size2(model, CBL_idx, CBL_idx, CBLidx2mask)

        compact_module_defs = deepcopy(model.module_defs)

        for j, module_def in enumerate(compact_module_defs):
            if module_def["type"] == "route":
                from_layers = [int(s) for s in module_def["layers"].split(",")]
                if len(from_layers) == 1 and from_layers[0] > 0:
                    count = 0
                    for i in index_prune:
                        if i <= from_layers[0]:
                            count += 1
                    from_layers[0] = from_layers[0] - count
                    from_layers = str(from_layers[0])
                    module_def["layers"] = from_layers

                elif len(from_layers) == 2:
                    count = 0
                    if from_layers[1] > 0:
                        for i in index_prune:
                            if i <= from_layers[1]:
                                count += 1
                        from_layers[1] = from_layers[1] - count
                    else:
                        for i in index_prune:
                            if i > j + from_layers[1] and i < j:
                                count += 1
                        from_layers[1] = from_layers[1] + count

                    from_layers = ", ".join([str(s) for s in from_layers])
                    module_def["layers"] = from_layers

        compact_module_defs = [compact_module_defs[i] for i in index_remain]
        compact_model = Darknet([model.hyperparams.copy()] + compact_module_defs, (img_size, img_size)).to(device)
        for i, index in enumerate(index_remain):
            compact_model.module_list[i] = pruned_model.module_list[index]

        light_weight_model_parameters = get_parameters(compact_model)

        random_input = torch.rand((1, 3, img_size, img_size)).to(device)


        def get_inference_time(input, model, repeat=200):

            model.eval()
            start = time.time()
            with torch.no_grad():
                for i in range(repeat):
                    output = model(input)
            avg_infer_time = (time.time() - start) / repeat

            return avg_infer_time, output


        pruned_forward_time, pruned_output = get_inference_time(random_input, pruned_model)
        light_weight_model_inference_time, light_weight_model_output = get_inference_time(random_input, compact_model)

        with torch.no_grad():
            print(f"Light-weight mAP:")
            light_weight_model_metric = eval_model(compact_model)

        metric_table = [
            ["Metric", "Before", "After"],
            [
                "mAP",
                f"{heavy_weight_model_metric[0][2]:.6f}",
                f"{light_weight_model_metric[0][2]:.6f}",
            ],
            ["Parameters", f"{heavy_weight_model_parameters}", f"{light_weight_model_parameters}"],
            ["Inference", f"{pruned_forward_time:.4f}", f"{light_weight_model_inference_time:.4f}"],
        ]
        print(AsciiTable(metric_table).table)


        def Reward(acc, light_weight_model_parameters, heavy_weight_model_acc,
                   heavy_weight_model_parameters, size_constraint=None, acc_constraint=None, epoch=-1):
            R_a = (acc / heavy_weight_model_acc)  # if acc > 0.92 else -1
            C = (float(heavy_weight_model_parameters - light_weight_model_parameters)) / heavy_weight_model_parameters
            R_c = C * (2 - C)
            if size_constraint or acc_constraint:
                return getConstrainedReward(R_a, R_c, acc, light_weight_model_parameters, acc_constraint, size_constraint, epoch)
            return (R_a) * (R_c)


        light_weight_cfg_name = os.path.join("cfg", "light_weight_layer.cfg")
        light_weight_cfg_file = write_cfg(light_weight_cfg_name, [model.hyperparams.copy()] + compact_module_defs)
        print(f"Config file has been saved: {light_weight_cfg_file}")

        light_weight_model = os.path.join("./weights", "light_weight_layer.weights")
        save_weights(compact_model, path=light_weight_model)
        print(f"light_weight model has been saved: {light_weight_model}")

        params = {
            "cfg": light_weight_cfg_name,
            "data": opt.data,
            "weights": light_weight_model,
            "batch_size": opt.batch_size
        }
        best_mAP = rl_train(params)
        R = Reward(best_mAP, light_weight_model_parameters, heavy_weight_model_metric[0][2], heavy_weight_model_parameters)
        print("RL_rpoch: {}, R: {}".format(rl_epoch, R))
        controller.update_controller(R)
