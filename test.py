import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import lpips as lpips_pkg

from model import U_Net
from dataset import myImageFlodertest
from metrics import ssim, psnr, lpips
from contextlib import nullcontext

# 复现性设置
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    new_sd = {}
    for k, v in state_dict.items():
        new_sd[k.replace("module.", "")] = v
    return new_sd


def load_checkpoint(model, ckpt_path, device):
    sd = torch.load(ckpt_path, map_location=device)
    sd = _strip_module_prefix(sd)
    model.load_state_dict(sd, strict=True)
    return model


def _to_float(x):
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def compute_metrics(pred, gt, lpips_model):
    ssim_val = _to_float(ssim(pred, gt))
    psnr_val = _to_float(psnr(pred, gt))
    lpips_val = _to_float(lpips(pred, gt, lpips_model))
    return ssim_val, psnr_val, lpips_val


def save_reconstructed_image(pred, img_name, ssim_val, psnr_val, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # 直接使用传进来的 img_name 作为前缀
    fname = os.path.join(out_dir, f"{img_name}_SSIM_{ssim_val:.4f}_PSNR_{psnr_val:.2f}.png")
    save_image(pred, fname)


def test(dataset_path, ckpt_path, output_dir="test-results", num_workers=4, use_amp=False):
    model = U_Net(dim=32).to(device)
    model = load_checkpoint(model, ckpt_path, device)
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = myImageFlodertest(root=dataset_path, transform=transform, resize=False)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    lpips_model = lpips_pkg.LPIPS(net='alex').to(device)
    lpips_model.eval()

    total_ssim, total_psnr, total_lpips = 0.0, 0.0, 0.0
    count = 0

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "metrics.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # 表头增加 ciede2000
        writer.writerow(["image_name", "ssim", "psnr", "lpips"])

        autocast = torch.cuda.amp.autocast if (use_amp and device.type == "cuda") else nullcontext

        with torch.inference_mode():
            for idx, batch in enumerate(dataloader):

                blurry, clear, img_names = batch
                current_name = img_names[0]

                blurry = blurry.to(device, non_blocking=True)
                clear = clear.to(device, non_blocking=True)

                _, _, h, w = blurry.size()
                pad_h = (32 - h % 32) % 32
                pad_w = (32 - w % 32) % 32

                if pad_h > 0 or pad_w > 0:
                    blurry_padded = F.pad(blurry, (0, pad_w, 0, pad_h), mode='reflect')
                else:
                    blurry_padded = blurry

                with autocast():
                    pred_padded = model(blurry_padded)

                if pad_h > 0 or pad_w > 0:
                    pred = pred_padded[:, :, :h, :w]
                else:
                    pred = pred_padded

                pred = torch.clamp(pred, 0, 1)
                pred = (pred * 255.0).round() / 255.0

                ssim_val, psnr_val, lpips_val = compute_metrics(pred, clear, lpips_model)

                total_ssim += ssim_val
                total_psnr += psnr_val
                total_lpips += lpips_val
                count += 1

                save_reconstructed_image(pred, current_name, ssim_val, psnr_val, output_dir)

                writer.writerow([current_name, ssim_val, psnr_val, lpips_val])

                if (idx + 1) % 10 == 0 or (idx + 1) == len(dataloader):
                    print(f"[{idx + 1}/{len(dataloader)}] Name={current_name} SSIM={ssim_val:.4f} PSNR={psnr_val:.2f} "
                          f"LPIPS={lpips_val:.4f}")

    avg_ssim = total_ssim / max(count, 1)
    avg_psnr = total_psnr / max(count, 1)
    avg_lpips = total_lpips / max(count, 1)

    print("\n========= TEST SUMMARY =========")
    print(f"Average SSIM : {avg_ssim:.4f}")
    print(f"Average PSNR : {avg_psnr:.2f} dB")
    print(f"Average LPIPS: {avg_lpips:.4f}")
    print("================================\n")


if __name__ == "__main__":
    data_root = "/opt/data/private/wb/data/LHID/TestA"
    ckpt = "/opt/data/private/wb/see241/train_result_LHID_05_100_testA/prebest.pth"
    test(data_root, ckpt_path=ckpt, output_dir="./1", use_amp=False)