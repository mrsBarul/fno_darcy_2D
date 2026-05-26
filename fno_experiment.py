import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities3 import (
    MatReader,
    UnitGaussianNormalizer,
    LpLoss,
    count_params
)

# Настройка русского шрифта
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# SPECTRAL CONV
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(
            in_channels, out_channels, modes1, modes2, dtype=torch.cfloat
        ))
        self.weights2 = nn.Parameter(scale * torch.randn(
            in_channels, out_channels, modes1, modes2, dtype=torch.cfloat
        ))

    def compl_mul2d(self, x, w):
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x):
        B = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            B, x.shape[1], x.shape[2], x.shape[3] // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )
        return torch.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]))


# FNO
class FNO2d(nn.Module):
    def __init__(self, modes, width, layers=4, fc=128):
        super().__init__()
        self.width = width
        self.fc0 = nn.Linear(3, width)
        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()
        for _ in range(layers):
            self.conv.append(SpectralConv2d(width, width, modes, modes))
            self.w.append(nn.Conv1d(width, width, 1))
        self.fc1 = nn.Linear(width, fc)
        self.fc2 = nn.Linear(fc, 1)

    def forward(self, x):
        B, Nx, Ny, _ = x.shape
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        for conv, w in zip(self.conv, self.w):
            x1 = conv(x)
            x2 = w(x.view(B, self.width, -1)).view(B, self.width, Nx, Ny)
            x = F.relu(x1 + x2)
        x = x.permute(0, 2, 3, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ВИЗУАЛИЗАЦИЯ 1: Скорость и давление (поля)
def plot_fields(y_true, y_pred, name, save_dir):
    """Визуализация полей решения (скорость/давление)"""
    os.makedirs(save_dir, exist_ok=True)
    idx = 0
    y_t = y_true[idx].detach().cpu().numpy()
    y_p = y_pred[idx].detach().cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im1 = axes[0].imshow(y_t, cmap='viridis', aspect='auto')
    axes[0].set_title("Истинное решение", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Координата X (узлы сетки)", fontsize=11)
    axes[0].set_ylabel("Координата Y (узлы сетки)", fontsize=11)
    plt.colorbar(im1, ax=axes[0], label="Значение поля")
    
    im2 = axes[1].imshow(y_p, cmap='viridis', aspect='auto')
    axes[1].set_title("Предсказанное решение", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Координата X (узлы сетки)", fontsize=11)
    axes[1].set_ylabel("Координата Y (узлы сетки)", fontsize=11)
    plt.colorbar(im2, ax=axes[1], label="Значение поля")
    
    error = np.abs(y_t - y_p)
    im3 = axes[2].imshow(error, cmap='hot', aspect='auto')
    axes[2].set_title(f"Абсолютная ошибка\n(макс: {error.max():.4f})", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("Координата X (узлы сетки)", fontsize=11)
    axes[2].set_ylabel("Координата Y (узлы сетки)", fontsize=11)
    plt.colorbar(im3, ax=axes[2], label="Ошибка")
    
    plt.suptitle(f"Поля скорости/давления - {name}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{name}_fields.png", dpi=150, bbox_inches='tight')
    plt.close()

# ВИЗУАЛИЗАЦИЯ 2: FFT высокие и низкие частоты
def plot_high_vs_low_frequencies(true_field, pred_field, name, save_dir, cutoff_ratio=0.25):
    """Анализ высоких и низких частот через FFT"""
    os.makedirs(save_dir, exist_ok=True)
    
    if torch.is_tensor(true_field):
        true_field = true_field.cpu().numpy()
    if torch.is_tensor(pred_field):
        pred_field = pred_field.cpu().numpy()
    
    F_true = np.fft.fftshift(np.fft.fft2(true_field))
    F_pred = np.fft.fftshift(np.fft.fft2(pred_field))
    amp_true, amp_pred = np.abs(F_true), np.abs(F_pred)
    
    h, w = true_field.shape
    center_h, center_w = h // 2, w // 2
    cutoff_radius = int(min(h, w) * cutoff_ratio)
    
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center_w)**2 + (y - center_h)**2)
    mask_low = distance <= cutoff_radius
    mask_high = distance > cutoff_radius
    
    error_low = np.sum(np.abs(amp_true[mask_low] - amp_pred[mask_low])) / (np.sum(amp_true[mask_low]) + 1e-8)
    error_high = np.sum(np.abs(amp_true[mask_high] - amp_pred[mask_high])) / (np.sum(amp_true[mask_high]) + 1e-8)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    im1 = axes[0,0].imshow(np.log(amp_true + 1e-8), cmap='hot', aspect='auto')
    axes[0,0].set_title('Спектр истинного поля (log)', fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('Пространственная частота X', fontsize=10)
    axes[0,0].set_ylabel('Пространственная частота Y', fontsize=10)
    plt.colorbar(im1, ax=axes[0,0], label="Логарифм амплитуды")
    
    im2 = axes[0,1].imshow(np.log(amp_pred + 1e-8), cmap='hot', aspect='auto')
    axes[0,1].set_title('Спектр предсказанного поля (log)', fontsize=12, fontweight='bold')
    axes[0,1].set_xlabel('Пространственная частота X', fontsize=10)
    axes[0,1].set_ylabel('Пространственная частота Y', fontsize=10)
    plt.colorbar(im2, ax=axes[0,1], label="Логарифм амплитуды")
    
    im3 = axes[0,2].imshow(np.log(np.abs(amp_true - amp_pred) + 1e-8), cmap='hot', aspect='auto')
    axes[0,2].set_title('Спектральная ошибка (log)', fontsize=12, fontweight='bold')
    axes[0,2].set_xlabel('Пространственная частота X', fontsize=10)
    axes[0,2].set_ylabel('Пространственная частота Y', fontsize=10)
    plt.colorbar(im3, ax=axes[0,2], label="Логарифм ошибки")
    
    axes[1,0].imshow(mask_low, cmap='Blues', alpha=0.7, aspect='auto')
    axes[1,0].set_title(f'Низкие частоты (r ≤ {cutoff_radius})', fontsize=12, fontweight='bold')
    axes[1,0].set_xlabel('Пространственная частота X', fontsize=10)
    axes[1,0].set_ylabel('Пространственная частота Y', fontsize=10)
    
    axes[1,1].imshow(mask_high, cmap='Reds', alpha=0.7, aspect='auto')
    axes[1,1].set_title(f'Высокие частоты (r > {cutoff_radius})', fontsize=12, fontweight='bold')
    axes[1,1].set_xlabel('Пространственная частота X', fontsize=10)
    axes[1,1].set_ylabel('Пространственная частота Y', fontsize=10)
    
    bars = axes[1,2].bar(['Низкие частоты', 'Высокие частоты'], [error_low, error_high], 
                        color=['blue', 'red'], alpha=0.7, edgecolor='black')
    axes[1,2].set_ylabel('Относительная спектральная ошибка', fontsize=11)
    axes[1,2].set_title(f'Ошибка по частотам\nОтношение Выс/Низк: {error_high/error_low:.2f}', 
                        fontsize=12, fontweight='bold')
    axes[1,2].grid(True, alpha=0.3, axis='y')
    axes[1,2].set_ylim(0, max(error_low, error_high) * 1.2)
    
    for bar, err in zip(bars, [error_low, error_high]):
        axes[1,2].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{err:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle(f'FFT анализ высоких и низких частот - {name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{name}_high_vs_low_freq.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return {'error_low': error_low, 'error_high': error_high, 'ratio': error_high / error_low}


# ЗАПУСК ЭКСПЕРИМЕНТА
def run(cfg):
    TRAIN = "piececonst_r421_N1024_smooth1.mat"
    TEST = "piececonst_r421_N1024_smooth2.mat"
    ntrain, ntest = 200, 50
    batch = 20
    epochs = 30
    s = 29
    r = (421 - 1) // (s - 1)

    reader = MatReader(TRAIN)
    x_train = reader.read_field("coeff")[:ntrain, ::r, ::r][:, :s, :s]
    y_train = reader.read_field("sol")[:ntrain, ::r, ::r][:, :s, :s]

    reader.load_file(TEST)
    x_test = reader.read_field("coeff")[:ntest, ::r, ::r][:, :s, :s]
    y_test = reader.read_field("sol")[:ntest, ::r, ::r][:, :s, :s]

    x_norm = UnitGaussianNormalizer(x_train)
    y_norm = UnitGaussianNormalizer(y_train)

    x_train = x_norm.encode(x_train)
    x_test = x_norm.encode(x_test)
    y_train = y_norm.encode(y_train)

    grid = np.linspace(0, 1, 421)[::r]
    grid = np.stack(np.meshgrid(grid, grid), -1)
    grid = torch.tensor(grid[None], dtype=torch.float)

    x_train = torch.cat([x_train[..., None], grid.repeat(ntrain,1,1,1)], dim=-1)
    x_test = torch.cat([x_test[..., None], grid.repeat(ntest,1,1,1)], dim=-1)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch
    )

    model = FNO2d(cfg["modes"], cfg["width"], cfg["layers"], cfg["fc"]).to(device)
    params = count_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    l2 = LpLoss()

    hist_mse, hist_l2 = [], []
    best_l2 = 1e9

    for ep in range(epochs):
        model.train()
        mse_sum = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x).squeeze(-1)
            loss = F.mse_loss(out, y)
            loss.backward()
            opt.step()
            mse_sum += loss.item()

        model.eval()
        l2_sum = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).squeeze(-1)
                pred = y_norm.decode(pred)
                val = l2(pred.view(len(x), -1), y.view(len(x), -1)).item()
                l2_sum += val
                best_l2 = min(best_l2, val)

        hist_mse.append(mse_sum / len(train_loader))
        hist_l2.append(l2_sum / len(test_loader))

    # Сохраняем поля и FFT анализ только для baseline модели
    os.makedirs("out", exist_ok=True)
    
    with torch.no_grad():
        x_sample = x_test[:1].to(device)
        y_true_sample = y_test[:1]
        pred_sample = y_norm.decode(model(x_sample).squeeze(-1))
        y_true_decoded = y_norm.decode(y_true_sample)
        
        # Поля только для baseline
        if cfg["name"] == "Baseline":
            plot_fields(y_true_decoded, pred_sample, cfg['name'], "out")
            plot_high_vs_low_frequencies(
                y_true_decoded[0], pred_sample[0], 
                cfg['name'], "out", cutoff_ratio=0.25
            )

    return {
        "name": cfg["name"],
        "mse": hist_mse,
        "l2": hist_l2,
        "params": params,
        "best_l2": best_l2,
        "final_mse": hist_mse[-1],
        "final_l2": hist_l2[-1]
    }

# MAIN
def main():
    experiments = [
        {"name": "Baseline", "layers": 4, "width": 32, "modes": 12, "fc": 128},
        {"name": "Layers_2", "layers": 2, "width": 32, "modes": 12, "fc": 128},
        {"name": "Layers_6", "layers": 6, "width": 32, "modes": 12, "fc": 128},
        {"name": "Width_64", "layers": 4, "width": 64, "modes": 12, "fc": 128},
        {"name": "Modes_8", "layers": 4, "width": 32, "modes": 8, "fc": 128}
    ]

    results = {}
    
    for e in experiments:
        print(f"\n--- Обучение модели: {e['name']} ---")
        results[e["name"]] = run(e)

    #  ВЫВОД РЕЗУЛЬТАТОВ 

    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ")
    
    for name, res in results.items():
        print(f"\nМодель: {name}:")
        print(f"   Количество параметров: {res['params']:,}")
        print(f"   Наилучшая L2 ошибка: {res['best_l2']:.6f}")
        print(f"   Финальная MSE: {res['final_mse']:.6f}")
        print(f"   Финальная L2: {res['final_l2']:.6f}")
    
    #ОБЩИЙ ГРАФИК LOSS FUNCTION (MSE + L2 для всех моделей)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Левый график: MSE
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for idx, (name, res) in enumerate(results.items()):
        ax1.plot(res["mse"], label=f"{name}", linewidth=2, color=colors[idx])
    ax1.set_title("Среднеквадратичная ошибка (MSE)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Эпоха обучения", fontsize=12, fontweight='bold')
    ax1.set_ylabel("MSE", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Правый график: L2
    for idx, (name, res) in enumerate(results.items()):
        ax2.plot(res["l2"], label=f"{name}", linewidth=2, color=colors[idx])
    ax2.set_title("L2 ошибка (норма ошибки)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Эпоха обучения", fontsize=12, fontweight='bold')
    ax2.set_ylabel("L2 ошибка", fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.suptitle("Функция потерь (Loss Function) - сравнение всех моделей", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("out/COMMON_loss_function_all_models.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # ОБЩИЙ ГРАФИК MSE (только MSE для всех моделей)
    plt.figure(figsize=(12, 7))
    for idx, (name, res) in enumerate(results.items()):
        plt.plot(res["mse"], label=f"{name} (лучшая L2: {res['best_l2']:.5f})", 
                linewidth=2, color=colors[idx])
    plt.title("Сравнение среднеквадратичной ошибки (MSE) всех моделей", fontsize=16, fontweight='bold')
    plt.xlabel("Эпоха обучения", fontsize=13, fontweight='bold')
    plt.ylabel("Среднеквадратичная ошибка (MSE)", fontsize=13, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("out/COMMON_MSE_all_models.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Дополнительный график: сравнение L2
    plt.figure(figsize=(12, 7))
    for idx, (name, res) in enumerate(results.items()):
        plt.plot(res["l2"], label=f"{name} (лучшая: {res['best_l2']:.5f})", 
                linewidth=2, color=colors[idx])
    plt.title("Сравнение L2 ошибки всех моделей", fontsize=16, fontweight='bold')
    plt.xlabel("Эпоха обучения", fontsize=13, fontweight='bold')
    plt.ylabel("L2 ошибка", fontsize=13, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("out/COMMON_L2_all_models.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Лучшая модель
    best = min(results.items(), key=lambda x: x[1]['best_l2'])
    print("\n" + "="*80)
    print(f"ЛУЧШАЯ МОДЕЛЬ: {best[0]}")
    print(f"   Наилучшая L2 ошибка: {best[1]['best_l2']:.6f}")
    print(f"   Количество параметров: {best[1]['params']:,}")
    
    print("\nВсе результаты сохранены в папку 'out'")
    print("   Созданные графики:")
    print("   1. Baseline_fields.png - Поля скорости/давления (только для Baseline)")
    print("   2. Baseline_high_vs_low_freq.png - FFT анализ (только для Baseline)")
    print("   3. COMMON_loss_function_all_models.png - ОБЩИЙ график функции потерь (MSE + L2)")
    print("   4. COMMON_MSE_all_models.png - ОБЩИЙ график MSE")
    print("   5. COMMON_L2_all_models.png - ОБЩИЙ график L2 ошибки")


if __name__ == "__main__":
    main()