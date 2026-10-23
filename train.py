import time
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from model import U_Net
from dataset import myImageFlodertrainDHID, myImageFlodertest, myImageFlodertrain
import torch.nn as nn
import torch
from PIL import Image
import csv
from metrics import psnr, ssim

torch.backends.cudnn.deterministic = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# ----------------------------------------
# 权重初始化函数
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('ConvBlock') != -1:
        for child in m.children():
            if isinstance(child, torch.nn.Conv2d):
                torch.nn.init.normal_(child.weight.data, 0.0, 0.02)
                if hasattr(child, 'bias') and child.bias is not None:
                    torch.nn.init.constant_(child.bias.data, 0.0)
            elif isinstance(child, torch.nn.BatchNorm2d):
                torch.nn.init.normal_(child.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(child.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def custom_psnr(image1, image2):
    mse = F.mse_loss(image1, image2)
    max_pixel = 1.0
    psnr_value = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr_value.item()

def BatchPSNR(tar_img, prd_img):
    psnr_vals = [custom_psnr(tar, prd) for tar, prd in zip(tar_img, prd_img)]
    return torch.tensor(psnr_vals)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_mse_loss(img1, img2):
    return F.mse_loss(img1, img2)

def compute_l1_loss(input, output):
    return torch.mean(torch.abs(input - output))
# ----------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="RSID", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=16, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0009, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.7, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=40, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=500, help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between model checkpoints')
parser.add_argument('--mse_avg', action='store_true', help='enables mse avg')
parser.add_argument('--data_url', type=str, default="", help='name of the dataset')
parser.add_argument('--init_method', type=str, default="", help='name of the dataset')
parser.add_argument('--train_url', type=str, default="", help='name of the dataset')
# 保留参数定义以免启动脚本报错，但后续不再使用
parser.add_argument('--lambda_coarse', type=float, default=0.2, help='weight for coarse low-frequency supervision')
parser.add_argument('--coarse_scale', type=int, default=4, help='downsample scale for coarse supervision')


if __name__ == "__main__":

    opt = parser.parse_args()
    print(opt)

    os.makedirs('images/ablation/%s' % opt.dataset_name, exist_ok=True)
    # os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)

    cuda = True if torch.cuda.is_available() else False
    EPS = 1e-12

    # 定义损失函数
    criterion_pixelwise = torch.nn.L1Loss()

    generator = U_Net(dim=32)
    pytorch_total_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print("Total_params_model: {}M".format(pytorch_total_params / 1000000.0))

    if cuda:
        generator = generator.to(device)
        criterion_pixelwise.to(device)

    if opt.epoch != 0:
        generator.load_state_dict(torch.load('ablation/RSID/prebest.pth'), strict=True)
    else:
        generator.apply(weights_init_normal)

    if cuda and torch.cuda.device_count() > 1:
        generator = nn.DataParallel(generator)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_G, T_max=300, eta_min=1e-6
    )

    mytransform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_root = '../data/RSID'
    test_roots = ['../data/RSID/test']

    myfolder = myImageFlodertrain(root=train_root, transform=mytransform,
                                  crop=False, resize=False, random_flip=True)
    dataloader = DataLoader(
        myfolder,
        num_workers=opt.n_cpu,
        batch_size=opt.batch_size,
        shuffle=True,
        persistent_workers=True,
    )
    print('data loader finish！')

    test_datasets = {
        os.path.basename(root): myImageFlodertest(root=root, transform=mytransform, resize=False)
        for root in test_roots
    }
    test_dataloaders = {
        name: DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, persistent_workers=True)
        for name, dataset in test_datasets.items()
    }

    def sample_images(epoch, i, real_A, real_B, fake_B):
        data, pred, label = real_A * 255, fake_B * 255, real_B * 255
        data = data.cpu()
        pred = pred.cpu()
        label = label.cpu()
        pred = torch.clamp(pred.detach(), 0, 255)
        data, pred, label = data.int(), pred.int(), label.int()
        h, w = pred.shape[-2], pred.shape[-1]
        img = np.zeros((h, 1 * 3 * w, 3))
        for idx in range(0, 1):
            row = idx * h
            tmplist = [data[idx], pred[idx], label[idx]]
            for k in range(3):
                col = k * w
                tmp = np.transpose(tmplist[k], (1, 2, 0))
                img[row:row + h, col:col + w] = np.array(tmp)
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        img.save("./images/ablation/RSID/%03d_%06d.png" % (epoch, i))

    def evaluate_model(generator, test_dataloaders):
        generator.eval()
        total_psnr = {name: 0 for name in test_dataloaders}
        total_loss = {name: 0 for name in test_dataloaders}
        count = {name: 0 for name in test_dataloaders}
        criterion = nn.L1Loss()

        for name, dataloader in test_dataloaders.items():
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    real_A, real_B, _ = batch
                    real_A, real_B = real_A.to(device), real_B.to(device)

                    # ------------- 改：修复解包报错 -------------
                    fake_B = generator(real_A)
                    # --------------------------------------

                    psnr_val = BatchPSNR(fake_B, real_B)
                    loss = criterion(fake_B, real_B).mean().item()
                    total_psnr[name] += psnr_val.mean().item()
                    total_loss[name] += loss
                    count[name] += 1

        avg_psnr = {name: total_psnr[name] / count[name] for name in total_psnr}
        avg_loss = {name: total_loss[name] / count[name] for name in total_loss}
        mean_psnr = np.mean(list(avg_psnr.values()))
        mean_loss = np.mean(list(avg_loss.values()))
        generator.train()
        return mean_psnr, mean_loss, avg_psnr, avg_loss

    prev_time = time.time()
    step = 0
    best_psnr = -float('inf')
    save_dir = 'ablation/RSID'
    os.makedirs(save_dir, exist_ok=True)

    run_cfg = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device),
        'dataset_name': opt.dataset_name,
        'train_root': train_root,
        'test_roots': ';'.join(test_roots),
        'start_epoch': opt.epoch,
        'total_epochs': opt.n_epochs,
        'batch_size': opt.batch_size,
        'learning_rate': opt.lr,
        'betas': f'({opt.b1}, {opt.b2})',
        'img_size': f'{opt.img_height}x{opt.img_width}',
        'channels': opt.channels,
        'optimizer': 'Adam',
        'scheduler': 'CosineAnnealingLR(T_max=200, eta_min=1e-6)',
        'model': 'U_Net(dim=32)',
        'trainable_params_M': round(pytorch_total_params / 1000000.0, 3),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
    }
    remark_text = '; '.join([f"{k}={v}" for k, v in run_cfg.items()])

    csv_file = os.path.join(save_dir, 'training_log.csv')
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Mean_PSNR', 'Mean_Loss',
                         'Test_Mean_PSNR', 'Test_Mean_Loss', 'Remark'])

    for epoch in range(opt.epoch, opt.n_epochs):
        epoch_psnr = []
        epoch_loss = []

        with tqdm(total=len(dataloader), desc=f"Epoch {epoch}/{opt.n_epochs}", unit="batch", colour="white") as pbar:
            for i, batch in enumerate(dataloader):
                step += 1
                real_A, real_B = batch[0].to(device), batch[1].to(device)

                optimizer_G.zero_grad()

                # ------------- 改：修复解包报错 -------------
                fake_B = generator(real_A)
                # --------------------------------------

                loss_pixel = criterion_pixelwise(fake_B, real_B).mean()
                loss_mse = compute_mse_loss(fake_B, real_B).mean()
                p0 = 1 - ssim(fake_B, real_B).mean()

                # ------------- 删除了 loss_coarse -----------------

                # 总损失 (仅保留 pixel, mse, p0)
                loss_G = loss_pixel + loss_mse + p0

                loss_G.backward()
                optimizer_G.step()

                psnr_val = BatchPSNR(fake_B, real_B)
                epoch_psnr.append(psnr_val.mean().item())
                epoch_loss.append(loss_G.item())

                # 同步更新进度条显示，去除 loss_coarse
                pbar.set_postfix({
                    'loss_pixel': f"{loss_pixel.item():.6f}",
                    'loss_mse': f"{loss_mse.item():.6f}",
                    'p0': f"{p0.item():.6f}",
                    'loss_G': f"{loss_G.item():.6f}"
                })
                pbar.update(1)

                if i % max(1, len(dataloader) // 2) == 0:
                    sample_images(epoch, i, real_A, real_B, fake_B)

        mean_psnr = np.mean(epoch_psnr)
        mean_loss = np.mean(epoch_loss)
        print(f"Epoch [{epoch}/{opt.n_epochs}], Mean PSNR: {mean_psnr:.6f}, Mean Loss: {mean_loss:.6f}")

        test_mean_psnr, test_mean_loss, test_avg_psnr, test_avg_loss = evaluate_model(generator, test_dataloaders)
        print(f"Epoch [{epoch}/{opt.n_epochs}], Test Mean PSNR: {test_mean_psnr:.6f}, Test Mean Loss: {test_mean_loss:.6f}")

        if test_mean_psnr > 26:
            torch.save(
                generator.module.state_dict() if isinstance(generator, nn.DataParallel) else generator.state_dict(),
                os.path.join(save_dir, f'generator_epoch_{epoch}.pth')
            )

        if test_mean_psnr > best_psnr:
            best_psnr = test_mean_psnr
            best_model_state = generator.module.state_dict() if isinstance(generator, nn.DataParallel) else generator.state_dict()
            torch.save(best_model_state, os.path.join(save_dir, 'prebest.pth'))

        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, mean_psnr, mean_loss, test_mean_psnr, test_mean_loss, remark_text])

        scheduler_G.step()