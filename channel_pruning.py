import argparse
import os

from models import *
from test import test
from utils.prune_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="cfg/yolov4.cfg", help="cfg file path")
    parser.add_argument("--data", type=str, default="data/voc.data", help="data file path")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help=" model weights")
    parser.add_argument("--global_percent", type=float, default=0.8, help="channel prune percent")
    parser.add_argument("--layer_keep", type=float, default=0.01, help="channel keep percent per layer")
    parser.add_argument("--img_size", type=int, default=416, help="width * height")
    opt = parser.parse_args()

    img_size = opt.img_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.cfg, (img_size, img_size)).to(device)

    if opt.weights.endswith(".pt"):
        model.load_state_dict(torch.load(opt.weights, map_location=device)["model"])
    else:
        _ = load_darknet_weights(model, opt.weights)
    eval_model = lambda model: test(model=model, cfg=opt.cfg, data=opt.data, batch_size=16, img_size=img_size)
    get_parameters = lambda model: sum([param.nelement() for param in model.parameters()])

    with torch.no_grad():
        print("\nHeavy-weight model mAP:")
        heavy_weight_model_metric = eval_model(model)
    heavy_weight_model_parameters = get_parameters(model)

    CBL_idx, Conv_idx, prune_idx, _, _ = parse_module_defs2(model.module_defs)

    bn_weights = gather_bn_weights(model.module_list, prune_idx)

    sorted_bn = torch.sort(bn_weights)[0]
    sorted_bn, sorted_index = torch.sort(bn_weights)
    thresh_index = int(len(bn_weights) * opt.global_percent)
    thresh = sorted_bn[thresh_index].cuda()


    def get_filters_mask(model, thre, CBL_idx, prune_idx):
        pruned = 0
        total = 0
        num_filters = []
        filters_mask = []
        for idx in CBL_idx:
            bn_module = model.module_list[idx][1]
            if idx in prune_idx:

                weight_copy = bn_module.weight.data.abs().clone()

                channels = weight_copy.shape[0]  #
                min_channel_num = (
                    int(channels * opt.layer_keep)
                    if int(channels * opt.layer_keep) > 0
                    else 1
                )
                mask = weight_copy.gt(thresh).float()

                if int(torch.sum(mask)) < min_channel_num:
                    _, sorted_index_weights = torch.sort(weight_copy, descending=True)
                    mask[sorted_index_weights[:min_channel_num]] = 1.0
                remain = int(mask.sum())
                pruned = pruned + mask.shape[0] - remain
            else:
                mask = torch.ones(bn_module.weight.data.shape)
                remain = mask.shape[0]

            total += mask.shape[0]
            num_filters.append(remain)
            filters_mask.append(mask.clone())

        prune_ratio = pruned / total
        print(f"Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}")

        return num_filters, filters_mask


    num_filters, filters_mask = get_filters_mask(model, thresh, CBL_idx, prune_idx)
    CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}
    CBLidx2filters = {idx: filters for idx, filters in zip(CBL_idx, num_filters)}

    for i in model.module_defs:
        if i["type"] == "shortcut":
            i["is_access"] = False
    merge_mask(model, CBLidx2mask, CBLidx2filters)


    def prune_and_eval(model, CBL_idx, CBLidx2mask):
        model_copy = deepcopy(model)

        for idx in CBL_idx:
            bn_module = model_copy.module_list[idx][1]
            mask = CBLidx2mask[idx].cuda()
            bn_module.weight.data.mul_(mask)

        with torch.no_grad():
            mAP = eval_model(model_copy)[0][2]

        print(f"mask the gamma as zero, mAP of the model is {mAP:.4f}")


    for i in CBLidx2mask:
        CBLidx2mask[i] = CBLidx2mask[i].clone().cpu().numpy()

    pruned_model = prune_model_keep_size2(model, prune_idx, CBL_idx, CBLidx2mask)
    with torch.no_grad():
        eval_model(pruned_model)

    for i in model.module_defs:
        if i["type"] == "shortcut":
            i.pop("is_access")

    compact_module_defs = deepcopy(model.module_defs)
    for idx in CBL_idx:
        assert compact_module_defs[idx]["type"] == "convolutional"
        compact_module_defs[idx]["filters"] = str(CBLidx2filters[idx])

    compact_model = Darknet([model.hyperparams.copy()] + compact_module_defs, (img_size, img_size)).to(device)
    compact_nparameters = get_parameters(compact_model)

    init_weights_from_loose_model(compact_model, pruned_model, CBL_idx, Conv_idx, CBLidx2mask)
    random_input = torch.rand((1, 3, img_size, img_size)).to(device)


    def get_inference_time(input, model, repeat=200):
        model.eval()
        start = time.time()
        with torch.no_grad():
            for i in range(repeat):
                output = model(input)
        avg_infer_time = (time.time() - start) / repeat
        return avg_infer_time, output


    print("testing inference time...")
    pruned_forward_time, _ = get_inference_time(random_input, pruned_model)
    light_weight_model_inference_time, _ = get_inference_time(random_input, compact_model)

    print("testing the final model...")
    with torch.no_grad():
        light_weight_model_metric = eval_model(compact_model)

    metric_table = [
        ["Metric", "Before", "After"],
        [
            "mAP",
            f"{heavy_weight_model_metric[0][2]:.6f}",
            f"{light_weight_model_metric[0][2]:.6f}",
        ],
        ["Parameters", f"{heavy_weight_model_parameters}", f"{compact_nparameters}"],
        ["Inference", f"{pruned_forward_time:.4f}", f"{light_weight_model_inference_time:.4f}"],
    ]
    print(AsciiTable(metric_table).table)

    light_weight_cfg_name = os.path.join("cfg", "light_weight_channel.cfg")

    light_weight_cfg_file = write_cfg(light_weight_cfg_name, [model.hyperparams.copy()] + compact_module_defs)
    print(f"Config file has been saved: {light_weight_cfg_file}")

    light_weight_model_name = os.path.join("weights", "light_weight_channel.weights")

    save_weights(compact_model, path=light_weight_model_name)
    print(f"Light_weight model has been saved: {light_weight_model_name}")
