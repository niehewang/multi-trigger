import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints")
    parser.add_argument("--temps", type=str, default="./temps")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--input_height", type=int, default=None)
    parser.add_argument("--input_width", type=int, default=None)
    parser.add_argument("--input_channel", type=int, default=None)
    parser.add_argument("--num_classes", type=int, default=10)

    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--lr_G", type=float, default=1e-2)
    parser.add_argument("--lr_C", type=float, default=1e-2)
    parser.add_argument("--lr_M", type=float, default=1e-2)
    parser.add_argument("--schedulerG_milestones", type=list, default=[200, 300, 400, 500])
    parser.add_argument("--schedulerC_milestones", type=list, default=[100, 200, 300, 400])
    parser.add_argument("--schedulerM_milestones", type=list, default=[10, 20])
    parser.add_argument("--schedulerG_lambda", type=float, default=0.1)
    parser.add_argument("--schedulerC_lambda", type=float, default=0.1)
    parser.add_argument("--schedulerM_lambda", type=float, default=0.1)
    parser.add_argument("--n_iters", type=int, default=100)
    parser.add_argument("--lambda_div", type=float, default=1)
    parser.add_argument("--lambda_norm", type=float, default=100)
    parser.add_argument("--num_workers", type=float, default=4)

    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--attack_mode", type=str, default="all2one", help="all2one or all2all")
    parser.add_argument("--p_attack", type=float, default=0.1)
    parser.add_argument("--p_cross", type=float, default=0.1)
    parser.add_argument("--mask_density", type=float, default=0.032)
    parser.add_argument("--EPSILON", type=float, default=1e-7)

    parser.add_argument("--random_rotation", type=int, default=10)
    parser.add_argument("--random_crop", type=int, default=5)
    parser.add_argument("--k", type=int, default=4)
    #for wanet
    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--cross_ratio", type=float, default=2)  # rho_a = pc, rho_n = pc * cross_ratio

    parser.add_argument(
        "--grid-rescale", type=float, default=1
    )  # scale grid values to avoid pixel values going out of [-1, 1]. For example, grid-rescale = 0.98
    #end of wanet

    #for badnets
    parser.add_argument('--trigger_path', default="./triggers/trigger_white.png", help='Trigger Path (default: ./triggers/trigger_white.png)')
    parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')

    #end of badnets
    return parser
