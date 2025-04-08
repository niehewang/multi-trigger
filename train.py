import os
import shutil

import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from classifier_models import PreActResNet18, ResNet18
from dataloader import get_dataloader,PostTensorTransform
from networks.models import Generator, NetC_MNIST
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils import progress_bar
from PIL import Image

def create_targets_bd(targets, opt):
    if opt.attack_mode == "all2one":
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif opt.attack_mode == "all2all_mask":
        bd_targets = torch.tensor([(label + 1) % opt.num_classes for label in targets])
    else:
        raise Exception("{} attack mode is not implemented".format(opt.attack_mode))
    return bd_targets.to(opt.device)


def create_bd(inputs, targets, netG, netM, opt):
    bd_targets = create_targets_bd(targets, opt)
    patterns = netG(inputs)
    patterns = netG.normalize_pattern(patterns)

    masks_output = netM.threshold(netM(inputs))
    bd_inputs = inputs + (patterns - inputs) * masks_output
    return bd_inputs, bd_targets, patterns, masks_output


def create_cross(inputs1, inputs2, netG, netM, opt):
    patterns2 = netG(inputs2)
    patterns2 = netG.normalize_pattern(patterns2)
    masks_output = netM.threshold(netM(inputs2))
    inputs_cross = inputs1 + (patterns2 - inputs1) * masks_output
    return inputs_cross, patterns2, masks_output

def badnets(inputs, opt):
    # Badnets
    # Add a backdoor to the image
    inputs_bd = inputs.clone()
    trigger_img = Image.open(opt.trigger_path).convert("RGB")
    trigger_img = trigger_img.resize((opt.trigger_size, opt.trigger_size))
    trigger_tensor = transforms.ToTensor()(trigger_img)
    x_offset = opt.input_width - opt.trigger_size
    y_offset = opt.input_height - opt.trigger_size
    for i in range(inputs_bd.size(0)):
        img_tensor = inputs_bd[i].clone()
        img_tensor[:, y_offset:y_offset + opt.trigger_size, x_offset:x_offset + opt.trigger_size] = trigger_tensor[:, :opt.trigger_size, :opt.trigger_size]
        inputs_bd[i] = img_tensor
    return inputs_bd

def nashville_filter(inputs,opt):
    inputs_bd = inputs.clone()
    for i in range(inputs_bd.size(0)):
        tensor_img = inputs_bd[i].clone()
        tensor_img[0] = torch.clamp(tensor_img[0] * 1.05, 0, 1)  # 调整红色通道
        tensor_img[1] = torch.clamp(tensor_img[1] * 0.93, 0, 1)  # 调整绿色通道
        tensor_img[2] = torch.clamp(tensor_img[2] * 0.82, 0, 1)  # 调整蓝色通道
        
        # 添加淡黄色调
        overlay = torch.tensor([1.0, 0.94, 0.75]).view(3, 1, 1).expand_as(tensor_img)
        overlay = overlay.to(opt.device)
        tensor_img = torch.clamp(tensor_img * 0.8 + overlay * 0.2, 0, 1)
        
        # 增加对比度
        mean = torch.mean(tensor_img, dim=(1, 2), keepdim=True)
        tensor_img = torch.clamp((tensor_img - mean) * 1.3 + mean, 0, 1)
        
        # 增加亮度
        tensor_img = torch.clamp(tensor_img * 1.1, 0, 1)
        inputs_bd[i] = tensor_img
    return inputs_bd

def train_step(
    netC, netG, netM, optimizerC, optimizerG, schedulerC, schedulerG, train_dl1, train_dl2,noise_grid, identity_grid, epoch, opt, tf_writer
):
    netC.train()
    netG.train()
    print(" Training:")
    total = 0
    total_cross = 0
    total_bd = 0
    total_clean = 0

    total_correct_clean = 0
    total_cross_correct = 0
    total_bd_correct = 0

    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    criterion_div = nn.MSELoss(reduction="none")
    for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(train_dl1)), train_dl1, train_dl2):
        optimizerC.zero_grad()

        inputs1, targets1 = inputs1.to(opt.device), targets1.to(opt.device)
        inputs2, targets2 = inputs2.to(opt.device), targets2.to(opt.device)

        bs = inputs1.shape[0]
        num_bd = int(opt.p_attack * bs)
        num_cross = int(opt.p_cross * bs)

        inputs_bd, targets_bd, patterns1, masks1 = create_bd(inputs1[:num_bd], targets1[:num_bd], netG, netM, opt)
        inputs_cross, patterns2, masks2 = create_cross(
            inputs1[num_bd : num_bd + num_cross], inputs2[num_bd : num_bd + num_cross], netG, netM, opt
        )
        # add wanet
        transforms2 = PostTensorTransform(opt).to(opt.device)
        grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        ins = torch.rand(num_cross, opt.input_height, opt.input_height, 2).to(opt.device) * 2 - 1
        grid_temps2 = grid_temps.repeat(num_cross, 1, 1, 1) + ins / opt.input_height
        grid_temps2 = torch.clamp(grid_temps2, -1, 1)
        inputs_bd = F.grid_sample(inputs_bd[:], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
        #targets_bd = torch.ones_like(targets1[:num_bd]) * opt.target_label
        inputs_cross = F.grid_sample(inputs_cross[:], grid_temps2, align_corners=True)
        #end of wanet
        #add badnets
        inputs_bd = badnets(inputs_bd, opt)
        #end of badnets
        # nash
        inputs_bd = nashville_filter(inputs_bd,opt)
        # end of nash

        total_inputs = torch.cat((inputs_bd, inputs_cross, inputs1[num_bd + num_cross :]), 0)
        
        #wanet
        total_inputs = transforms2(total_inputs)
        #end
        total_targets = torch.cat((targets_bd, targets1[num_bd:]), 0)

        preds = netC(total_inputs)
        loss_ce = criterion(preds, total_targets)

        # Calculating diversity loss
        distance_images = criterion_div(inputs1[:num_bd], inputs2[num_bd : num_bd + num_bd])
        distance_images = torch.mean(distance_images, dim=(1, 2, 3))
        distance_images = torch.sqrt(distance_images)

        distance_patterns = criterion_div(patterns1, patterns2)
        distance_patterns = torch.mean(distance_patterns, dim=(1, 2, 3))
        distance_patterns = torch.sqrt(distance_patterns)

        loss_div = distance_images / (distance_patterns + opt.EPSILON)
        loss_div = torch.mean(loss_div) * opt.lambda_div

        total_loss = loss_ce + loss_div
        total_loss.backward()
        optimizerC.step()
        optimizerG.step()

        total += bs
        total_bd += num_bd
        total_cross += num_cross
        total_clean += bs - num_bd - num_cross

        total_correct_clean += torch.sum(
            torch.argmax(preds[num_bd + num_cross :], dim=1) == total_targets[num_bd + num_cross :]
        )
        total_cross_correct += torch.sum(
            torch.argmax(preds[num_bd : num_bd + num_cross], dim=1) == total_targets[num_bd : num_bd + num_cross]
        )
        total_bd_correct += torch.sum(torch.argmax(preds[:num_bd], dim=1) == targets_bd)
        total_loss += loss_ce.detach() * bs
        avg_loss = total_loss / total

        acc_clean = total_correct_clean * 100.0 / total_clean
        acc_bd = total_bd_correct * 100.0 / total_bd
        acc_cross = total_cross_correct * 100.0 / total_cross
        infor_string = "CE loss: {:.4f} - Accuracy: {:.3f} | BD Accuracy: {:.3f} | Cross Accuracy: {:3f}".format(
            avg_loss, acc_clean, acc_bd, acc_cross
        )
        progress_bar(batch_idx, len(train_dl1), infor_string)

        # Saving images for debugging

        if batch_idx == len(train_dl1) - 2:
            dir_temps = os.path.join(opt.temps, opt.dataset)
            if not os.path.exists(dir_temps):
                os.makedirs(dir_temps)
            images = netG.denormalize_pattern(torch.cat((inputs1[:num_bd], patterns1, inputs_bd), dim=2))
            file_name = "{}_{}_images.png".format(opt.dataset, opt.attack_mode)
            file_path = os.path.join(dir_temps, file_name)
            torchvision.utils.save_image(images, file_path, normalize=True, pad_value=1)

    if not epoch % 10:
        # Save figures (tfboard)
        tf_writer.add_scalars(
            "Accuracy/lambda_div_{}/".format(opt.lambda_div),
            {"Clean": acc_clean, "BD": acc_bd, "Cross": acc_cross},
            epoch,
        )

        tf_writer.add_scalars("Loss/lambda_div_{}".format(opt.lambda_div), {"CE": loss_ce, "Div": loss_div}, epoch)

    schedulerC.step()
    schedulerG.step()


def eval(
    netC,
    netG,
    netM,
    optimizerC,
    optimizerG,
    schedulerC,
    schedulerG,
    test_dl1,
    test_dl2,
    noise_grid, 
    identity_grid,
    epoch,
    best_acc_clean,
    best_acc_bd,
    best_acc_cross,
    opt,
):
    netC.eval()
    netG.eval()
    print(" Eval:")
    total = 0.0

    total_correct_clean = 0.0
    total_correct_bd = 0.0
    total_correct_cross = 0.0
    for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(test_dl1)), test_dl1, test_dl2):
        with torch.no_grad():
            inputs1, targets1 = inputs1.to(opt.device), targets1.to(opt.device)
            inputs2, targets2 = inputs2.to(opt.device), targets2.to(opt.device)
            bs = inputs1.shape[0]

            preds_clean = netC(inputs1)
            correct_clean = torch.sum(torch.argmax(preds_clean, 1) == targets1)
            total_correct_clean += correct_clean
    
            inputs_bd, targets_bd, _, _ = create_bd(inputs1, targets1, netG, netM, opt)
            inputs_cross, _, _ = create_cross(inputs1, inputs2, netG, netM, opt)
            #wanet
            grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)

            ins = torch.rand(bs, opt.input_height, opt.input_height, 2).to(opt.device) * 2 - 1
            grid_temps2 = grid_temps.repeat(bs, 1, 1, 1) + ins / opt.input_height
            grid_temps2 = torch.clamp(grid_temps2, -1, 1)
            inputs_bd = F.grid_sample(inputs_bd, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
            #targets_bd = torch.ones_like(targets1) * opt.target_label
            #end
            #add badnets
            inputs_bd = badnets(inputs_bd, opt)
            #end of badnets

            #nash
            inputs_bd = nashville_filter(inputs_bd,opt)
            #end of nash

            preds_bd = netC(inputs_bd)
            correct_bd = torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
            total_correct_bd += correct_bd

            inputs_cross, _, _ = create_cross(inputs1, inputs2, netG, netM, opt)
            #wanet
            
            inputs_cross = F.grid_sample(inputs_cross, grid_temps2, align_corners=True)
            #end
            preds_cross = netC(inputs_cross)
            correct_cross = torch.sum(torch.argmax(preds_cross, 1) == targets1)
            total_correct_cross += correct_cross

            total += bs
            avg_acc_clean = total_correct_clean * 100.0 / total
            avg_acc_cross = total_correct_cross * 100.0 / total
            avg_acc_bd = total_correct_bd * 100.0 / total

            infor_string = "Clean Accuracy: {:.3f} | Backdoor Accuracy: {:.3f} | Cross Accuracy: {:3f}".format(
                avg_acc_clean, avg_acc_bd, avg_acc_cross
            )
            progress_bar(batch_idx, len(test_dl1), infor_string)

    print(
        " Result: Best Clean Accuracy: {:.3f} - Best Backdoor Accuracy: {:.3f} - Best Cross Accuracy: {:.3f}| Clean Accuracy: {:.3f}".format(
            best_acc_clean, best_acc_bd, best_acc_cross, avg_acc_clean
        )
    )
    print(" Saving!!")
    best_acc_clean = avg_acc_clean
    best_acc_bd = avg_acc_bd
    best_acc_cross = avg_acc_cross
    state_dict = {
        "netC": netC.state_dict(),
        "netG": netG.state_dict(),
        "netM": netM.state_dict(),
        "optimizerC": optimizerC.state_dict(),
        "optimizerG": optimizerG.state_dict(),
        "schedulerC": schedulerC.state_dict(),
        "schedulerG": schedulerG.state_dict(),
        "best_acc_clean": best_acc_clean,
        "best_acc_bd": best_acc_bd,
        "best_acc_cross": best_acc_cross,
        "epoch": epoch,
        "opt": opt,
    }
    ckpt_folder = os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode)
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    ckpt_path = os.path.join(ckpt_folder, "{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.dataset))
    torch.save(state_dict, ckpt_path)
    return best_acc_clean, best_acc_bd, best_acc_cross, epoch


# -------------------------------------------------------------------------------------
def train_mask_step(netM, optimizerM, schedulerM, train_dl1, train_dl2, epoch, opt, tf_writer):
    netM.train() 
    print(" Training:")
    total = 0

    total_loss = 0
    criterion_div = nn.MSELoss(reduction="none")
    for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(train_dl1)), train_dl1, train_dl2):
        optimizerM.zero_grad()

        inputs1, targets1 = inputs1.to(opt.device), targets1.to(opt.device)
        inputs2, targets2 = inputs2.to(opt.device), targets2.to(opt.device)

        bs = inputs1.shape[0]
        masks1 = netM(inputs1)
        masks1, masks2 = netM.threshold(netM(inputs1)), netM.threshold(netM(inputs2))

        # Calculating diversity loss
        distance_images = criterion_div(inputs1, inputs2)
        distance_images = torch.mean(distance_images, dim=(1, 2, 3))
        distance_images = torch.sqrt(distance_images)

        distance_patterns = criterion_div(masks1, masks2)
        distance_patterns = torch.mean(distance_patterns, dim=(1, 2, 3))
        distance_patterns = torch.sqrt(distance_patterns)

        loss_div = distance_images / (distance_patterns + opt.EPSILON)
        loss_div = torch.mean(loss_div) * opt.lambda_div

        loss_norm = torch.mean(F.relu(masks1 - opt.mask_density))

        total_loss = opt.lambda_norm * loss_norm + opt.lambda_div * loss_div
        total_loss.backward()
        optimizerM.step()
        infor_string = "Mask loss: {:.4f} - Norm: {:.3f} | Diversity: {:.3f}".format(total_loss, loss_norm, loss_div)
        progress_bar(batch_idx, len(train_dl1), infor_string)

        # Saving images for debugging
        if batch_idx == len(train_dl1) - 2:
            dir_temps = os.path.join(opt.temps, opt.dataset, "masks")
            if not os.path.exists(dir_temps):
                os.makedirs(dir_temps)
            path_masks = os.path.join(dir_temps, "{}_{}_masks.png".format(opt.dataset, opt.attack_mode))
            torchvision.utils.save_image(masks1, path_masks, pad_value=1)

    if not epoch % 10:
        tf_writer.add_scalars(
            "Loss/lambda_norm_{}".format(opt.lambda_norm), {"MaskNorm": loss_norm, "MaskDiv": loss_div}, epoch
        )

    schedulerM.step()


def eval_mask(netM, optimizerM, schedulerM, test_dl1, test_dl2, epoch, opt):
    netM.eval()
    print(" Eval:")
    total = 0.0

    criterion_div = nn.MSELoss(reduction="none")
    for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(test_dl1)), test_dl1, test_dl2):
        with torch.no_grad():
            inputs1, targets1 = inputs1.to(opt.device), targets1.to(opt.device)
            inputs2, targets2 = inputs2.to(opt.device), targets2.to(opt.device)
            bs = inputs1.shape[0]
            masks1, masks2 = netM.threshold(netM(inputs1)), netM.threshold(netM(inputs2))

            # Calculating diversity loss
            distance_images = criterion_div(inputs1, inputs2)
            distance_images = torch.mean(distance_images, dim=(1, 2, 3))
            distance_images = torch.sqrt(distance_images)

            distance_patterns = criterion_div(masks1, masks2)
            distance_patterns = torch.mean(distance_patterns, dim=(1, 2, 3))
            distance_patterns = torch.sqrt(distance_patterns)

            loss_div = distance_images / (distance_patterns + opt.EPSILON)
            loss_div = torch.mean(loss_div) * opt.lambda_div

            loss_norm = torch.mean(F.relu(masks1 - opt.mask_density))

            infor_string = "Norm: {:.3f} | Diversity: {:.3f}".format(loss_norm, loss_div)
            progress_bar(batch_idx, len(test_dl1), infor_string)

    state_dict = {
        "netM": netM.state_dict(),
        "optimizerM": optimizerM.state_dict(),
        "schedulerM": schedulerM.state_dict(),
        "epoch": epoch,
        "opt": opt,
    }
    ckpt_folder = os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode, "mask")
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    ckpt_path = os.path.join(ckpt_folder, "{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.dataset))
    torch.save(state_dict, ckpt_path)
    return epoch


# -------------------------------------------------------------------------------------


def train(opt):
    # Prepare model related things
    if opt.dataset == "cifar10":
        netC = PreActResNet18().to(opt.device)
    elif opt.dataset == "gtsrb":
        netC = PreActResNet18(num_classes=43).to(opt.device)
    elif opt.dataset == "mnist":
        netC = NetC_MNIST().to(opt.device)
    else:
        raise Exception("Invalid dataset")

    netG = Generator(opt).to(opt.device)
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)
    optimizerG = torch.optim.Adam(netG.parameters(), opt.lr_G, betas=(0.5, 0.9))
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizerG, opt.schedulerG_milestones, opt.schedulerG_lambda)

    netM = Generator(opt, out_channels=1).to(opt.device)
    optimizerM = torch.optim.Adam(netM.parameters(), opt.lr_M, betas=(0.5, 0.9))
    schedulerM = torch.optim.lr_scheduler.MultiStepLR(optimizerM, opt.schedulerM_milestones, opt.schedulerM_lambda)

    # For tensorboard
    log_dir = os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_dir = os.path.join(log_dir, "log_dir")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tf_writer = SummaryWriter(log_dir=log_dir)

    # Continue training ?
    ckpt_folder = os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode)
    ckpt_path = os.path.join(ckpt_folder, "{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.dataset))
    # if os.path.exists(ckpt_path):
    #     state_dict = torch.load(ckpt_path)
    #     netC.load_state_dict(state_dict["netC"])
    #     netG.load_state_dict(state_dict["netG"])
    #     netM.load_state_dict(state_dict["netM"])
    #     epoch = state_dict["epoch"] + 1
    #     optimizerC.load_state_dict(state_dict["optimizerC"])
    #     optimizerG.load_state_dict(state_dict["optimizerG"])
    #     schedulerC.load_state_dict(state_dict["schedulerC"])
    #     schedulerG.load_state_dict(state_dict["schedulerG"])
    #     best_acc_clean = state_dict["best_acc_clean"]
    #     best_acc_bd = state_dict["best_acc_bd"]
    #     best_acc_cross = state_dict["best_acc_cross"]
    #     opt = state_dict["opt"]
    #     print("Continue training")
    # else:
        # Prepare mask
    best_acc_clean = 0.0
    best_acc_bd = 0.0
    best_acc_cross = 0.0
    epoch = 1

    # Reset tensorboard
    shutil.rmtree(log_dir)
    os.makedirs(log_dir)
    print("Training from scratch")

    # Prepare dataset
    train_dl1 = get_dataloader(opt, train=True)
    train_dl2 = get_dataloader(opt, train=True)
    test_dl1 = get_dataloader(opt, train=False)
    test_dl2 = get_dataloader(opt, train=False)

    #for wanet,prepare grid
    if not hasattr(opt, 'k'):
        opt.k = 4  # 设置默认值
    ins = torch.rand(1, 2, opt.k, opt.k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = (
        F.upsample(ins, size=opt.input_height, mode="bicubic", align_corners=True)
        .permute(0, 2, 3, 1)
        .to(opt.device)
        )
    array1d = torch.linspace(-1, 1, steps=opt.input_height)
    x, y = torch.meshgrid(array1d, array1d)
    identity_grid = torch.stack((y, x), 2)[None, ...].to(opt.device)
    #end of grid

    if epoch == 1:
        netM.train()
        for i in range(25):
            print(
                "Epoch {} - {} - {} | mask_density: {} - lambda_div: {}  - lambda_norm: {}:".format(
                    epoch, opt.dataset, opt.attack_mode, opt.mask_density, opt.lambda_div, opt.lambda_norm
                )
            )
            train_mask_step(netM, optimizerM, schedulerM, train_dl1, train_dl2, epoch, opt, tf_writer)
            epoch = eval_mask(netM, optimizerM, schedulerM, test_dl1, test_dl2, epoch, opt)
            epoch += 1
    netM.eval()
    netM.requires_grad_(False)

    for i in range(opt.n_iters):
        print(
            "Epoch {} - {} - {} | mask_density: {} - lambda_div: {}:".format(
                epoch, opt.dataset, opt.attack_mode, opt.mask_density, opt.lambda_div
            )
        )
        train_step(
            netC,
            netG,
            netM,
            optimizerC,
            optimizerG,
            schedulerC,
            schedulerG,
            train_dl1,
            train_dl2,
            noise_grid, 
            identity_grid,
            epoch,
            opt,
            tf_writer,
        )

        best_acc_clean, best_acc_bd, best_acc_cross, epoch = eval(
            netC,
            netG,
            netM,
            optimizerC,
            optimizerG,
            schedulerC,
            schedulerG,
            test_dl1,
            test_dl2,
            noise_grid, 
            identity_grid,
            epoch,
            best_acc_clean,
            best_acc_bd,
            best_acc_cross,
            opt,
        )
        epoch += 1
        if epoch > opt.n_iters:
            break


def main():
    opt = config.get_arguments().parse_args()
    if opt.dataset == "mnist" or opt.dataset == "cifar10":
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    elif opt.dataset == "celeba":
        opt.num_classes = 8
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    else:
        raise Exception("Invalid Dataset")
    train(opt)


if __name__ == "__main__":
    main()
