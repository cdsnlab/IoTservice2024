import argparse
import yaml
from easydict import EasyDict
import network
from typing import Dict, Callable, Tuple, Union, List
import argparse
import math


def get_argparser():
    parser = argparse.ArgumentParser(description="DeYO exps")

    parser.add_argument(
        "-c", "--config", default=None, help="name of config file under ./configs/"
    )
    parser.add_argument("--data_root", default=None, help="root for all dataset")
    parser.add_argument(
        "--dset", default=None, type=str, help="ImageNet-C, Waterbirds, ColoredMNIST"
    )
    parser.add_argument(
        "--output", default=None, help="the output directory of this experiment"
    )
    parser.add_argument(
        "--wandb_interval",
        default=None,
        type=int,
        help="print outputs to wandb at given interval.",
    )
    parser.add_argument("--wandb_log", default=None, type=int)
    parser.add_argument("--exp_name", default=None, type=str)

    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument("--gpu", default=None, type=str, help="GPU id to use.")
    parser.add_argument("--debug", default=None, type=bool, help="debug or not.")
    parser.add_argument(
        "--continual", default=None, type=bool, help="continual tta or fully tta"
    )

    # dataloader
    parser.add_argument(
        "--workers",
        default=None,
        type=int,
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--test_batch_size",
        default=None,
        type=int,
        help="mini-batch size for testing, before default value is 4",
    )
    parser.add_argument(
        "--if_shuffle", default=None, type=bool, help="if shuffle the test set."
    )

    # corruption settings
    parser.add_argument(
        "--level", default=None, type=int, help="corruption level of test(val) set."
    )
    parser.add_argument(
        "--corruption", default=None, type=str, help="corruption type of test(val) set."
    )

    # eata settings
    parser.add_argument("--eata_fishers", default=None, type=int)
    parser.add_argument(
        "--fisher_size",
        default=None,
        type=int,
        help="number of samples to compute fisher information matrix.",
    )  # 2000 500
    parser.add_argument(
        "--fisher_alpha",
        type=float,
        default=None,
        help="the trade-off between entropy and regularization loss",
    )  # 2000 100 5000 1
    parser.add_argument(
        "--e_margin",
        type=float,
        default=None,
        help="entropy margin E_0 for filtering reliable samples",
    )
    parser.add_argument(
        "--d_margin",
        type=float,
        default=None,
        help="\epsilon for filtering redundant samples",
    )

    # Exp Settings
    parser.add_argument(
        "--method", default=None, type=str, help="no_adapt, tent, eata, sar, deyo"
    )
    parser.add_argument(
        "--model",
        default=None,
        type=str,
        help="resnet50_gn_timm or resnet50_bn_torch or vitbase_timm or resnet18_bn",
    )
    parser.add_argument(
        "--exp_type",
        default=None,
        type=str,
        help="normal, mix_shifts, bs1, label_shifts, spurious",
    )
    parser.add_argument(
        "--patch_len",
        default=None,
        type=int,
        help="The number of patches per row/column",
    )

    # SAR parameters
    parser.add_argument(
        "--sar_margin_e0",
        default=None,
        type=float,
        help="the threshold for reliable minimization in SAR.",
    )
    parser.add_argument(
        "--imbalance_ratio",
        default=None,
        type=float,
        help="imbalance ratio for label shift exps, selected from [1, 1000, 2000, 3000, 4000, 5000, 500000], 1  denotes totally uniform and 500000 denotes (almost the same to Pure Class Order).",
    )

    # FATA parameters
    parser.add_argument("--aug_type", default=None, type=str, help="patch, pixel, occ")
    parser.add_argument("--occlusion_size", default=None, type=int)
    parser.add_argument("--row_start", default=None, type=int)
    parser.add_argument("--column_start", default=None, type=int)

    parser.add_argument(
        "--fata_margin",
        default=None,
        type=float,
        help="Entropy threshold for sample selection $\tau_\mathrm{Ent}$ in Eqn. (8)",
    )
    parser.add_argument(
        "--fata_margin_e0",
        default=None,
        type=float,
        help="Entropy margin for sample weighting $\mathrm{Ent}_0$ in Eqn. (10)",
    )
    parser.add_argument(
        "--plpd_threshold",
        default=None,
        type=float,
        help="PLPD threshold for sample selection $\tau_\mathrm{PLPD}$ in Eqn. (8)",
    )

    parser.add_argument("--fishers", default=None, type=int)
    parser.add_argument("--filter_ent", default=None, type=int)
    parser.add_argument("--filter_plpd", default=None, type=int)
    parser.add_argument("--reweight_ent", default=None, type=int)
    parser.add_argument("--reweight_plpd", default=None, type=int)

    parser.add_argument("--topk", default=None, type=int)

    parser.add_argument(
        "--wbmodel_name",
        default=None,
        type=str,
        help="Waterbirds pre-trained model path",
    )
    parser.add_argument(
        "--cmmodel_name",
        default=None,
        type=str,
        help="ColoredMNIST pre-trained model path",
    )
    parser.add_argument(
        "--lr_mul", default=None, type=float, help="5 for Waterbirds, ColoredMNIST"
    )

    return parser


def get_args() -> argparse.Namespace:
    parser = get_argparser()
    opts = parser.parse_args()
    return opts


def get_config(args: argparse.Namespace, config_path="configs/config.yaml") -> EasyDict:
    if config_path.split(".")[-1] != "yaml":
        config_path = f"configs/{config_path}.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)
        config = EasyDict(config)

    with open("configs/default.yaml") as f:
        dft = yaml.safe_load(f)
        dft = EasyDict(dft)

    # copy parsed arguments (argument first, unless default value)
    for key in args.__dir__():
        if key[:2] != "__":
            v = getattr(args, key)
            if not hasattr(config, key) or v is not None:
                setattr(config, key, getattr(args, key))

    with open("configs/data.yaml") as f:
        data_conf = yaml.safe_load(f)
        config.datasets = data_conf["datasets"]

    print(f"Config loaded from {config_path}")
    return config


def get_opts(config_path="configs/config.yaml") -> EasyDict:
    opts = get_args()
    return merge_opts(opts, config_path)


# opts.config is preferred to config_path
def merge_opts(opts: Dict, config_path="configs/config.yaml") -> argparse.Namespace:
    return get_config(opts, config_path=opts.config if opts.config else config_path)


def execute_by(branches: Dict[str, Tuple[Callable[[EasyDict], any]]], by="task"):
    opts = get_args()
    if hasattr(opts, by):
        task = getattr(opts, by)
        if task in branches:
            fn = branches[task]
            if type(fn) == tuple:
                fn, config = branches[task]
            opts = merge_opts(opts, config_path=config)
            return fn(opts)

        else:
            raise NotImplementedError(f"Task {by}={task} is not implemented")

    raise NotImplementedError(f"{by} is not in the opts")


def opts_to_dict(opts: EasyDict) -> dict:
    return {
        k: opts_to_dict(v) if isinstance(v, EasyDict) else v
        for k, v in opts.items()
        if not k.startswith("_")
    }


def dump_opts(opts: Dict[str, any]) -> str:
    def _dump_opt(opt: Dict[str, any], level) -> str:
        results = []
        for k, v in opt.items():
            if k.startswith("_"):
                continue
            if isinstance(v, dict):
                v = "\n" + _dump_opt(v, level=level + 1)
            results.append(f'{"    " * level}* {k}: {v}')
        return "\n".join(results)

    return _dump_opt(opts, level=0)
