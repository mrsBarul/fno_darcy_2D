import numpy as np
from utilities3 import MatReader
import matplotlib.pyplot as plt
import os

os.makedirs('dataset_visualization', exist_ok=True)

TRAIN_PATH1 = 'piececonst_r421_N1024_smooth1.mat'
TRAIN_PATH2 = 'piececonst_r421_N1024_smooth2.mat'

reader1 = MatReader(TRAIN_PATH1)
coeff1 = reader1.read_field('coeff')
sol1 = reader1.read_field('sol')

reader2 = MatReader(TRAIN_PATH2)
coeff2 = reader2.read_field('coeff')
sol2 = reader2.read_field('sol')

if hasattr(coeff1, 'numpy'):
    coeff1 = coeff1.numpy()
    sol1 = sol1.numpy()
    coeff2 = coeff2.numpy()
    sol2 = sol2.numpy()

print('НАБОР ДАННЫХ 1: piececonst_r421_N1024_smooth1.mat')
print(f'форма coeff: {coeff1.shape}')
print(f'форма sol: {sol1.shape}')
print(f'диапазон coeff: [{coeff1.min():.6f}, {coeff1.max():.6f}]')
print(f'диапазон sol: [{sol1.min():.6f}, {sol1.max():.6f}]')
print()

print('НАБОР ДАННЫХ 2: piececonst_r421_N1024_smooth2.mat')
print(f'форма coeff: {coeff2.shape}')
print(f'форма sol: {sol2.shape}')
print(f'диапазон coeff: [{coeff2.min():.6f}, {coeff2.max():.6f}]')
print(f'диапазон sol: [{sol2.min():.6f}, {sol2.max():.6f}]')
print()

print('СРАВНЕНИЕ:')
print(f'макс. абсолютная разница в coeff: {np.abs(coeff1 - coeff2).max():.6f}')
print(f'макс. абсолютная разница в sol: {np.abs(sol1 - sol2).max():.6f}')
print('='*60)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# График 1: Dataset1 - coeff[0]
im1 = axes[0,0].imshow(coeff1[0], cmap='jet')
axes[0,0].set_title('Поле ПРОНИЦАЕМОСТИ\nНабор данных 1 (smooth1)\ncoeff[0]', fontsize=9, fontweight='bold')
axes[0,0].set_xlabel('Ось X (координата, пиксель) | от 0 до 1023', fontsize=8)
axes[0,0].set_ylabel('Ось Y (координата, пиксель)', fontsize=8)
cbar1 = plt.colorbar(im1, ax=axes[0,0])
cbar1.set_label('Значение проницаемости', fontsize=8)

# График 2: Dataset1 - sol[0]
im2 = axes[0,1].imshow(sol1[0], cmap='jet')
axes[0,1].set_title('Поле ДАВЛЕНИЯ\nНабор данных 1 (smooth1)\nsol[0]', fontsize=9, fontweight='bold')
axes[0,1].set_xlabel('Ось X (координата, пиксель) | от 0 до 1023', fontsize=8)
axes[0,1].set_ylabel('Ось Y (координата, пиксель)', fontsize=8)
cbar2 = plt.colorbar(im2, ax=axes[0,1])
cbar2.set_label('Значение давления', fontsize=8)

# График 3: Dataset2 - coeff[0]
im3 = axes[0,2].imshow(coeff2[0], cmap='jet')
axes[0,2].set_title('Поле ПРОНИЦАЕМОСТИ\nНабор данных 2 (smooth2)\ncoeff[0]', fontsize=9, fontweight='bold')
axes[0,2].set_xlabel('Ось X (координата, пиксель) | от 0 до 1023', fontsize=8)
axes[0,2].set_ylabel('Ось Y (координата, пиксель)', fontsize=8)
cbar3 = plt.colorbar(im3, ax=axes[0,2])
cbar3.set_label('Значение проницаемости', fontsize=8)

# График 4: Dataset2 - sol[0]
im4 = axes[0,3].imshow(sol2[0], cmap='jet')
axes[0,3].set_title('Поле ДАВЛЕНИЯ\nНабор данных 2 (smooth2)\nsol[0]', fontsize=9, fontweight='bold')
axes[0,3].set_xlabel('Ось X (координата, пиксель) | от 0 до 1023', fontsize=8)
axes[0,3].set_ylabel('Ось Y (координата, пиксель)', fontsize=8)
cbar4 = plt.colorbar(im4, ax=axes[0,3])
cbar4.set_label('Значение давления', fontsize=8)

# График 5: |coeff1 - coeff2|
diff_coeff = np.abs(coeff1[0] - coeff2[0])
im5 = axes[1,0].imshow(diff_coeff, cmap='hot')
axes[1,0].set_title('РАЗНИЦА в проницаемости\nмежду датасетами 1 и 2\n|coeff1 - coeff2|', fontsize=9, fontweight='bold')
axes[1,0].set_xlabel('Ось X (координата, пиксель) | от 0 до 1023', fontsize=8)
axes[1,0].set_ylabel('Ось Y (координата, пиксель)', fontsize=8)
cbar5 = plt.colorbar(im5, ax=axes[1,0])
cbar5.set_label('Абсолютная разница', fontsize=8)

# График 6: |sol1 - sol2|
diff_sol = np.abs(sol1[0] - sol2[0])
im6 = axes[1,1].imshow(diff_sol, cmap='hot')
axes[1,1].set_title('РАЗНИЦА в давлении\nмежду датасетами 1 и 2\n|sol1 - sol2|', fontsize=9, fontweight='bold')
axes[1,1].set_xlabel('Ось X (координата, пиксель) | от 0 до 1023', fontsize=8)
axes[1,1].set_ylabel('Ось Y (координата, пиксель)', fontsize=8)
cbar6 = plt.colorbar(im6, ax=axes[1,1])
cbar6.set_label('Абсолютная разница', fontsize=8)

# График 7: Гистограмма распределения coeff
axes[1,2].hist(coeff1[0].ravel(), bins=50, alpha=0.5, label='Датасет 1 (smooth1)', density=True, color='blue')
axes[1,2].hist(coeff2[0].ravel(), bins=50, alpha=0.5, label='Датасет 2 (smooth2)', density=True, color='orange')
axes[1,2].set_title('РАСПРЕДЕЛЕНИЕ значений\nпроницаемости (coeff)', fontsize=9, fontweight='bold')
axes[1,2].set_xlabel('Значение проницаемости', fontsize=8)
axes[1,2].set_ylabel('Плотность вероятности', fontsize=8)
axes[1,2].legend(fontsize=7)
axes[1,2].grid(True, alpha=0.3)

# График 8: Гистограмма распределения sol
axes[1,3].hist(sol1[0].ravel(), bins=50, alpha=0.5, label='Датасет 1 (smooth1)', density=True, color='blue')
axes[1,3].hist(sol2[0].ravel(), bins=50, alpha=0.5, label='Датасет 2 (smooth2)', density=True, color='orange')
axes[1,3].set_title('РАСПРЕДЕЛЕНИЕ значений\nдавления (sol)', fontsize=9, fontweight='bold')
axes[1,3].set_xlabel('Значение давления', fontsize=8)
axes[1,3].set_ylabel('Плотность вероятности', fontsize=8)
axes[1,3].legend(fontsize=7)
axes[1,3].grid(True, alpha=0.3)

plt.suptitle('ВИЗУАЛИЗАЦИЯ И СРАВНЕНИЕ ДАТАСЕТОВ\nПроницаемость (coeff) и Давление (sol) | Размер сетки: 1024×1024 пикселя', 
             fontsize=12, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('dataset_visualization/datasets_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
