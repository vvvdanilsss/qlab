import numpy as np
import cv2
import pandas as pd
import os
from matplotlib import pyplot as plt


def load_intensity_from_csv(filename):
    df = pd.read_csv(filename, header=None)
    return df[0].values


def load_image(filename):
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)


def calculate_snr(G_normalized, signal_threshold):
    # Пиксели выше порога считаем сигналом, ниже - шумом
    signal_pixels = G_normalized[G_normalized > signal_threshold]
    noise_pixels = G_normalized[G_normalized <= signal_threshold]

    # Средняя интенсивность сигнальных и шумовых пикселей
    mean_signal = np.mean(signal_pixels)
    mean_noise = np.mean(noise_pixels)

    # Стандартное отклонение шумовых пикселей
    std_noise = np.std(noise_pixels)

    # Расчет SNR
    snr = (mean_signal - mean_noise) / std_noise if std_noise != 0 else float('inf')
    return snr


def calculate_G(B_values, I_values):
    avg_B = np.mean(B_values)
    avg_I = np.mean(I_values, axis=0)

    sum_BI = np.zeros_like(avg_I, dtype=np.float64)
    for i, I in enumerate(I_values):
        sum_BI += B_values[i] * I
    avg_BI = sum_BI / len(B_values)

    G = avg_BI - avg_B * avg_I
    return G


def estimate_snr_from_histogram(G_normalized):
    histogram, _ = np.histogram(G_normalized, bins=256, range=(0, 256))

    # Определение диапазонов для пиков шума и сигнала
    noise_peak = np.argmax(histogram[:150])
    signal_peak = np.argmax(histogram[150:250]) + 150

    # Вычисление SNR
    snr_estimate = histogram[signal_peak] / histogram[noise_peak] if histogram[noise_peak] != 0 else np.inf
    return snr_estimate


def main():
    speckle_sizes = [8, 16, 32]
    snr_values = []

    for speckle_size in speckle_sizes:
        image_dir = f'images_{speckle_size}'
        intensity_file = f'intensity_{speckle_size}.csv'

        B_values = load_intensity_from_csv(intensity_file)
        I_values = []

        for i in range(1, 4000):
            image_path = os.path.join(image_dir, f'{i}.png')
            I_values.append(load_image(image_path))

        I_values = np.array(I_values, dtype=np.float64)
        G = calculate_G(B_values, I_values)

        G_normalized = ((G - np.min(G)) / (np.max(G) - np.min(G)) * 255).astype(np.uint8)

        # Сохраняем изображение, если нужно
        cv2.imwrite(f'G_{speckle_size}.png', G_normalized)

        # Предположим, что порог определения сигнала был установлен на уровне 100
        signal_threshold = 100
        snr = calculate_snr(G_normalized, signal_threshold)
        snr_values.append(snr / 60.87)
        print(f'SNR for speckle size {speckle_size}: {snr}')

        # Строим гистограмму для текущего размера спеклов
        plt.figure()
        plt.hist(G_normalized.flatten(), bins=256, range=(0, 256), alpha=0.75, color='blue', edgecolor='black')
        plt.title(f'Гистограмма интенсивностей для спеклов размером {speckle_size}')
        plt.xlabel('Интенсивность пикселя')
        plt.ylabel('Количество пикселей')
        plt.grid(True)
        plt.show()

    # Построение графика
    plt.figure(figsize=(10, 5))
    plt.plot(speckle_sizes, snr_values, 'o-', color='blue')
    plt.title('Зависимость SNR от размера спекла')
    plt.xlabel('Размер спекла')
    plt.ylabel('SNR')
    plt.grid(True)
    plt.xticks(speckle_sizes)
    plt.show()

    # Определение размера спеклов с наилучшим SNR
    best_speckle_size_index = np.argmax(snr_values)
    best_speckle_size = speckle_sizes[best_speckle_size_index]
    print(f'Лучшее соотношение сигнал/шум для размера спеклов: {best_speckle_size}')

    # Пересчет SNR для разного количества спеклов
    num_speckles_list = [2000, 2500, 3000, 3500, 4000]
    snr_for_best_speckle = []

    image_dir = f'images_{best_speckle_size}'
    intensity_file = f'intensity_{best_speckle_size}.csv'
    B_values = load_intensity_from_csv(intensity_file)

    for num_speckles in num_speckles_list:
        I_values_subset = [load_image(os.path.join(image_dir, f'{i}.png')) for i in range(1, num_speckles + 1)]
        I_values_subset = np.array(I_values_subset, dtype=np.float64)

        G_subset = calculate_G(B_values[:num_speckles], I_values_subset)
        G_normalized_subset = ((G_subset - np.min(G_subset)) / (np.max(G_subset) - np.min(G_subset)) * 255).astype(
            np.uint8)

        # Строим и выводим гистограмму для текущего подмножества спеклов
        plt.figure()
        plt.hist(G_normalized_subset.flatten(), bins=256, range=(0, 256), alpha=0.75, color='green', edgecolor='black')
        plt.title(f'Гистограмма интенсивностей для {num_speckles} спеклов размером {best_speckle_size}')
        plt.xlabel('Интенсивность пикселя')
        plt.ylabel('Количество пикселей')
        plt.grid(True)
        plt.show()

    for num_speckles in num_speckles_list:
        I_values_subset = [load_image(os.path.join(image_dir, f'{i}.png')) for i in range(1, num_speckles + 1)]
        I_values_subset = np.array(I_values_subset, dtype=np.float64)

        G_subset = calculate_G(B_values[:num_speckles], I_values_subset)
        G_normalized_subset = ((G_subset - np.min(G_subset)) / (np.max(G_subset) - np.min(G_subset)) * 255).astype(
            np.uint8)

        # Оцениваем SNR для текущего подмножества спеклов
        snr_estimate = estimate_snr_from_histogram(G_normalized_subset)
        snr_for_best_speckle.append(snr_estimate)

    # Построение графика для лучшего размера спеклов
    plt.figure(figsize=(10, 5))
    plt.plot(num_speckles_list, snr_for_best_speckle, 'o-', color='red')
    plt.title(f'Зависимость SNR от количества спеклов для размера {best_speckle_size}')
    plt.xlabel('Количество спеклов')
    plt.ylabel('SNR')
    plt.grid(True)
    plt.xticks(num_speckles_list)
    plt.show()


if __name__ == '__main__':
    main()
