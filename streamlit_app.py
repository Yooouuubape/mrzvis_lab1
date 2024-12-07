"""
Автор: Салюков Глеб Геннадьевич
Группа: 121731
Лабораторная работа №3
Вариант 11: Реализация модели линейной рекуркуляционной сети с постоянным коэффициентом обучения и ненормированными весами.
Тема: Линейная рекуркуляционная сеть (Линейный автоэнкодер)
Дата: 13.11.2024
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
import tempfile  # Убедитесь, что импортируете tempfile
import streamlit as st
from skimage.metrics import peak_signal_noise_ratio as psnr_metric, structural_similarity as ssim_metric
import io

# Настройки страницы Streamlit
st.set_page_config(
    page_title="Линейная Рекуркуляционная Сеть (Линейный автоэнкодер)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Стиль для заголовков и текста
st.markdown("""
    <style>
    .main-title {
        font-size: 50px;
        color: #4CAF50;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0px;
    }
    .info-text {
        font-size: 20px;
        color: #333333;
        text-align: center;
        margin-top: 0px;
    }
    .section-divider {
        border-top: 2px solid #4CAF50;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Заголовок приложения
st.markdown('<p class="main-title">Лабораторная работа №3: Линейная Рекуркуляционная Сеть (Линейный автоэнкодер)</p>', unsafe_allow_html=True)

# Информация об авторе и деталях
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<p class="info-text"><strong>Автор:</strong> Салюков Глеб Геннадьевич</p>', unsafe_allow_html=True)
with col2:
    st.markdown('<p class="info-text"><strong>Группа:</strong> 121731</p>', unsafe_allow_html=True)
with col3:
    st.markdown('<p class="info-text"><strong>Вариант:</strong> 11</p>', unsafe_allow_html=True)
with col4:
    st.markdown('<p class="info-text"><strong>Дата:</strong> 13.11.2024</p>', unsafe_allow_html=True)

# Горизонтальная линия
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# Функции

def convert_to_bmp_if_needed(file_bytes, filename):
    """
    Проверяет, является ли файл BMP. Если нет, конвертирует его в BMP.

    Параметры:
    - file_bytes: байты загруженного файла.
    - filename: имя загруженного файла.

    Возвращает:
    - bmp_path: путь к BMP-файлу (либо оригинальный, если он уже BMP).
    """
    try:
        _, ext = os.path.splitext(filename)
        if ext.lower() != '.bmp':
            with tempfile.NamedTemporaryFile(delete=False, suffix='.bmp') as tmp_file:
                with Image.open(io.BytesIO(file_bytes)) as img:
                    img.save(tmp_file.name, format='BMP')
            st.info(f"Изображение конвертировано в BMP формат.")
            return tmp_file.name
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.bmp') as tmp_file:
                tmp_file.write(file_bytes)
            st.info(f"Изображение уже в BMP формате.")
            return tmp_file.name
    except Exception as e:
        st.error(f"Ошибка при конвертации изображения: {e}")
        st.stop()

def load_image(filepath):
    """
    Загружает изображение формата BMP и нормализует пиксели в диапазоне [-1, 1].

    Параметры:
    - filepath: путь к файлу изображения.

    Возвращает:
    - image_array: массив изображения с нормализованными значениями [-1, 1].
    """
    try:
        with Image.open(filepath) as img:
            img = img.convert('RGB')  # Конвертация в RGB
            image_array = (np.asarray(img).astype(np.float32) / 127.5) - 1.0  # Нормализация в [-1, 1]
        return image_array
    except Exception as e:
        st.error(f"Ошибка при загрузке изображения: {e}")
        st.stop()

def split_into_patches(image, r, m, stride):
    """
    Разбивает изображение на патчи размером r x m с заданным шагом (stride).

    Параметры:
    - image: массив изображения.
    - r: высота патча.
    - m: ширина патча.
    - stride: шаг при разбиении на патчи.

    Возвращает:
    - patches: массив патчей, каждый патч — вектор.
    - padded_shape: форма дополненного изображения.
    """
    h, w, c = image.shape
    pad_h = (stride - (h - r) % stride) % stride
    pad_w = (stride - (w - m) % stride) % stride

    image_padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    new_h, new_w, _ = image_padded.shape
    num_patches_h = (new_h - r) // stride + 1
    num_patches_w = (new_w - m) // stride + 1

    patches = []
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            patch = image_padded[i * stride:i * stride + r, j * stride:j * stride + m, :]
            patches.append(patch.flatten())

    patches = np.array(patches)
    return patches, image_padded.shape

def initialize_weights(n, p, scale=1.0):
    """
    Инициализирует ненормированные веса с использованием стандартного нормального распределения и масштаба.

    Параметры:
    - n: количество входных нейронов.
    - p: количество скрытых нейронов.
    - scale: масштаб инициализации весов.

    Возвращает:
    - Wf: матрица весов прямого распространения.
    - Wb: матрица весов обратного распространения.
    """
    Wf = np.random.randn(n, p) * scale  # Ненормированные веса с масштабом
    Wb = np.random.randn(p, n) * scale  # Ненормированные веса с масштабом
    return Wf, Wb

def train_autoencoder(X, Wf, Wb, bf, bb, alpha, epochs, patience=100):
    """
    Обучает линейный автоэнкодер без регуляризации с ранней остановкой.

    Параметры:
    - X: входные патчи.
    - Wf: матрица весов прямого распространения.
    - Wb: матрица весов обратного распространения.
    - bf: смещение прямого распространения.
    - bb: смещение обратного распространения.
    - alpha: скорость обучения.
    - epochs: количество эпох обучения.
    - patience: количество эпох без улучшения перед остановкой.

    Возвращает:
    - Wf, Wb, bf, bb: обновлённые параметры модели.
    - errors: список значений ошибки по эпохам.
    """
    L, n = X.shape
    p = Wf.shape[1]
    errors = []
    best_mse = float('inf')
    no_improve_epochs = 0

    for epoch in range(epochs):
        # Прямое распространение
        Y = X.dot(Wf) + bf  # Кодирование
        X_r = Y.dot(Wb) + bb  # Декодирование

        # Вычисление ошибки (MSE)
        mse = np.mean((X_r - X) ** 2)
        errors.append(mse)

        # Проверка на улучшение
        if mse < best_mse:
            best_mse = mse
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # Проверка на раннюю остановку
        if no_improve_epochs >= patience:
            st.warning(f"Ранняя остановка: ошибка не улучшалась в течение {patience} эпох.")
            break

        # Проверка на NaN
        if np.isnan(mse):
            st.error(f"Эпоха {epoch + 1}: Ошибка стала NaN. Остановка обучения.")
            st.stop()

        # Обратное распространение ошибок
        dX_r = (X_r - X) / L  # Градиент по декодированному выходу
        dWb = Y.T.dot(dX_r)  # Градиент по Wb
        dbb = np.sum(dX_r, axis=0, keepdims=True)  # Градиент по bb

        dY = dX_r.dot(Wb.T)  # Градиент по Y
        dWf = X.T.dot(dY)  # Градиент по Wf
        dbf = np.sum(dY, axis=0, keepdims=True)  # Градиент по bf

        # Обновление весов и смещений
        Wf -= alpha * dWf
        Wb -= alpha * dWb
        bf -= alpha * dbf
        bb -= alpha * dbb

        # Вывод информации каждые 100 эпох
        if (epoch + 1) % 100 == 0 or epoch == 0:
            st.write(f"Эпоха {epoch + 1}, Ошибка: {mse:.6f}")

    return Wf, Wb, bf, bb, errors

def reconstruct_image(X_r, image_shape, r, m, stride):
    """
    Восстанавливает полное изображение из восстановленных патчей с усреднением перекрытий.

    Параметры:
    - X_r: восстановленные патчи.
    - image_shape: форма дополненного изображения.
    - r: высота патча.
    - m: ширина патча.
    - stride: шаг при размещении патчей.

    Возвращает:
    - reconstructed_image: восстановленное изображение.
    """
    h_padded, w_padded, c = image_shape
    reconstructed_sum = np.zeros(image_shape)
    reconstructed_count = np.zeros(image_shape)

    num_patches_h = (h_padded - r) // stride + 1
    num_patches_w = (w_padded - m) // stride + 1
    idx = 0

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            patch = X_r[idx].reshape((r, m, c))
            reconstructed_sum[i * stride:i * stride + r, j * stride:j * stride + m, :] += patch
            reconstructed_count[i * stride:i * stride + r, j * stride:j * stride + m, :] += 1
            idx += 1

    # Избегаем деления на ноль
    reconstructed_count[reconstructed_count == 0] = 1
    reconstructed_image = reconstructed_sum / reconstructed_count

    return reconstructed_image

def compute_metrics(original, reconstructed):
    """
    Вычисляет метрики качества восстановления: MSE, PSNR и SSIM.

    Параметры:
    - original: исходное изображение.
    - reconstructed: восстановленное изображение.

    Возвращает:
    - mse: среднеквадратичная ошибка.
    - psnr: Peak Signal-to-Noise Ratio.
    - ssim: Structural Similarity Index.
    """
    mse = np.mean((original - reconstructed) ** 2)
    psnr = psnr_metric(original, reconstructed, data_range=2.0)  # Диапазон [-1,1] => разница 2
    if original.shape[2] == 1:
        ssim = ssim_metric(original[:, :, 0], reconstructed[:, :, 0], data_range=2.0)
    else:
        ssim = 0
        for c in range(original.shape[2]):
            ssim += ssim_metric(original[:, :, c], reconstructed[:, :, c], data_range=2.0)
        ssim /= original.shape[2]
    return mse, psnr, ssim

def save_reconstructed_image(reconstructed_image):
    """
    Преобразует восстановленное изображение из диапазона [-1,1] в [0,255] и возвращает его как объект BytesIO.

    Параметры:
    - reconstructed_image: массив восстановленного изображения.

    Возвращает:
    - img_bytes: объект BytesIO с изображением.
    """
    img = Image.fromarray(((reconstructed_image + 1.0) * 127.5).astype(np.uint8))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='BMP')
    img_bytes.seek(0)
    return img_bytes

def save_weights(Wf, Wb, bf, bb):
    """
    Сохраняет матрицы весов и смещений в BytesIO.

    Параметры:
    - Wf: матрица весов прямого распространения.
    - Wb: матрица весов обратного распространения.
    - bf: смещение прямого распространения.
    - bb: смещение обратного распространения.

    Возвращает:
    - Wf_bytes, Wb_bytes, bf_bytes, bb_bytes: объекты BytesIO с весовыми матрицами и смещениями.
    """
    def save_numpy_array(array):
        bytes_io = io.BytesIO()
        np.save(bytes_io, array)
        bytes_io.seek(0)
        return bytes_io

    Wf_bytes = save_numpy_array(Wf)
    Wb_bytes = save_numpy_array(Wb)
    bf_bytes = save_numpy_array(bf)
    bb_bytes = save_numpy_array(bb)

    return Wf_bytes, Wb_bytes, bf_bytes, bb_bytes

def visualize_results(original, reconstructed, errors, mse, psnr, ssim, r, m, stride, X, X_r, num_patches=5):
    """
    Визуализирует оригинальное и восстановленное изображение, график ошибки и несколько патчей с дополнительной информацией.

    Параметры:
    - original: исходное изображение.
    - reconstructed: восстановленное изображение.
    - errors: список ошибок по эпохам.
    - mse: среднеквадратичная ошибка.
    - psnr: Peak Signal-to-Noise Ratio.
    - ssim: Structural Similarity Index.
    - r: высота патча.
    - m: ширина патча.
    - stride: шаг при разбиении на патчи.
    - X: массив исходных патчей.
    - X_r: массив восстановленных патчей.
    - num_patches: количество патчей для визуализации.
    """
    # Создание вкладок для разных визуализаций
    tab1, tab2, tab3 = st.tabs(["Изображения", "График Ошибок", "Патчи"])

    with tab1:
        st.subheader("Сравнение Оригинального и Восстановленного Изображений")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.image((original + 1.0) / 2.0, caption="Оригинальное изображение", use_column_width=True)

        with col2:
            st.image((reconstructed + 1.0) / 2.0, caption="Восстановленное изображение", use_column_width=True)

        with col3:
            difference = np.abs(original - reconstructed)
            st.image(difference / 2.0, caption="Разница (Оригинал - Восстановление)", use_column_width=True)

        with col4:
            st.markdown(f"""
            **Метрики Качества Восстановления:**
            - **MSE:** {mse:.6f}
            - **PSNR:** {psnr:.2f} dB
            - **SSIM:** {ssim:.4f}
            """)

    with tab2:
        st.subheader("График Изменения Ошибки по Эпохам")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(1, len(errors) + 1), errors, color='blue', label='MSE', linewidth=2)
        ax.set_xlabel('Эпоха обучения')
        ax.set_ylabel('Среднеквадратичная ошибка (MSE)')
        ax.set_title('График изменения ошибки по эпохам')
        ax.legend()
        ax.grid(True, linestyle='--', linewidth=0.5)
        # Добавление вертикальных линий для ключевых эпох
        key_epochs = [1, len(errors) // 4, len(errors) // 2, 3 * len(errors) // 4, len(errors)]
        for epoch in key_epochs:
            ax.axvline(x=epoch, color='red', linestyle='--', linewidth=1)
            ax.text(epoch, ax.get_ylim()[1]*0.95, f'Epoch {epoch}', rotation=90, color='red', fontsize=8, ha='right')
        st.pyplot(fig)

    with tab3:
        st.subheader("Примеры Патчей")
        num_patches_to_show = min(num_patches, X.shape[0])
        indices = np.random.choice(X.shape[0], num_patches_to_show, replace=False)
        for idx in indices:
            original_patch = X[idx].reshape((r, m, -1))
            reconstructed_patch = X_r[idx].reshape((r, m, -1))
            patch_difference = np.abs(original_patch - reconstructed_patch)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            # Оригинальный патч
            axes[0].imshow((original_patch + 1.0) / 2.0)
            axes[0].set_title('Оригинальный патч')
            axes[0].axis('off')

            # Восстановленный патч
            axes[1].imshow((reconstructed_patch + 1.0) / 2.0)
            axes[1].set_title('Восстановленный патч')
            axes[1].axis('off')

            # Разница патчей
            axes[2].imshow(patch_difference / 2.0)
            axes[2].set_title('Разница патчей')
            axes[2].axis('off')

            # Добавление метрики для патча
            patch_mse = np.mean((original_patch - reconstructed_patch) ** 2)
            fig.suptitle(f'Патч №{idx + 1}: MSE = {patch_mse:.6f}', fontsize=14)
            st.pyplot(fig)

# Главная функция Streamlit приложения
def main():
    # Боковая панель для ввода параметров
    st.sidebar.header("Параметры Модели и Обучения")

    # Загрузка изображения
    uploaded_file = st.sidebar.file_uploader("Загрузите изображение (PNG, JPG, BMP)", type=["png", "jpg", "jpeg", "bmp"])

    if uploaded_file is not None:
        filename = uploaded_file.name
        file_bytes = uploaded_file.read()
        # Конвертация в BMP, если необходимо
        bmp_path = convert_to_bmp_if_needed(file_bytes, filename)

        # Ввод параметров модели
        st.sidebar.subheader("Параметры Разбиения на Патчи")
        r = st.sidebar.number_input("Высота патча (r)", min_value=4, value=8, step=1)
        m = st.sidebar.number_input("Ширина патча (m)", min_value=4, value=8, step=1)
        stride = st.sidebar.number_input("Шаг (stride)", min_value=1, value=4, step=1)

        st.sidebar.subheader("Параметры Обучения")
        alpha = st.sidebar.number_input("Скорость обучения (alpha)", min_value=0.00001, max_value=1.0, value=0.0001, step=0.00001, format="%.5f")
        epochs = st.sidebar.number_input("Количество эпох", min_value=1, max_value=100000, value=3000, step=100)
        patience = st.sidebar.number_input("Параметр ранней остановки (patience)", min_value=10, max_value=1000, value=100, step=10)

        if st.sidebar.button("Начать Обучение"):
            with st.spinner("Обучение модели... Это может занять некоторое время..."):
                # Загрузка и нормализация изображения
                image = load_image(bmp_path)
                h, w, c = image.shape

                # Проверка параметров патча
                if not (4 <= r <= h and 4 <= m <= w):
                    st.error(f"Параметры патча некорректны: 4 <= r <= {h}, 4 <= m <= {w}")
                    st.stop()

                # Разбиение на патчи
                X, padded_shape = split_into_patches(image, r, m, stride)
                st.success(f"Изображение разбито на патчи размером {r}x{m} с шагом {stride}. Общее количество патчей: {X.shape[0]}")

                # Инициализация весов и смещений
                n = X.shape[1]
                p = max(1, n // 2 + 50)  # Размер скрытого слоя, можно настроить
                Wf, Wb = initialize_weights(n, p, scale=0.1)  # Уменьшение масштаба до 0.1
                bf = np.zeros((1, p))
                bb = np.zeros((1, n))

                st.write(f"**Инициализированы веса:**")
                st.write(f"- **Wf.shape:** {Wf.shape}")
                st.write(f"- **Wb.shape:** {Wb.shape}")
                st.write(f"**Параметры обучения:**")
                st.write(f"- **alpha:** {alpha}")
                st.write(f"- **epochs:** {epochs}")
                st.write(f"- **patience:** {patience}")

                # Засекаем время начала обучения
                start_time = time.time()

                # Обучение автоэнкодера
                Wf, Wb, bf, bb, errors = train_autoencoder(X, Wf, Wb, bf, bb, alpha, epochs, patience)

                # Засекаем время завершения обучения
                end_time = time.time()
                training_time = end_time - start_time

                st.success(f"Обучение завершено за {training_time:.2f} секунд.")

                # Восстановление изображения
                Y = X.dot(Wf) + bf
                X_r = Y.dot(Wb) + bb
                X_r = np.clip(X_r, -1, 1)  # Ограничение значений в [-1, 1]
                reconstructed_image = reconstruct_image(X_r, padded_shape, r, m, stride)
                h_original, w_original = image.shape[:2]
                reconstructed_image = reconstructed_image[:h_original, :w_original, :]

                # Вычисление метрик
                mse, psnr, ssim_val = compute_metrics(image, reconstructed_image)
                st.write(f"**Метрики Качества Восстановления:**")
                st.write(f"- **MSE:** {mse:.6f}")
                st.write(f"- **PSNR:** {psnr:.2f} dB")
                st.write(f"- **SSIM:** {ssim_val:.4f}")

                # Сохранение восстановленного изображения в BytesIO
                reconstructed_img_bytes = save_reconstructed_image(reconstructed_image)

                # Визуализация результатов
                visualize_results(image, reconstructed_image, errors, mse, psnr, ssim_val, r, m, stride, X, X_r, num_patches=5)

                # Сохранение весов в BytesIO
                Wf_bytes, Wb_bytes, bf_bytes, bb_bytes = save_weights(Wf, Wb, bf, bb)

                # Отображение кнопок для скачивания
                st.subheader("Скачать Результаты")
                st.download_button(
                    label="Скачать Восстановленное Изображение",
                    data=reconstructed_img_bytes,
                    file_name="reconstructed_image.bmp",
                    mime="image/bmp",
                )
                st.download_button(
                    label="Скачать Весовую Матрицу Wf",
                    data=Wf_bytes,
                    file_name="Wf.npy",
                    mime="application/octet-stream",
                )
                st.download_button(
                    label="Скачать Весовую Матрицу Wb",
                    data=Wb_bytes,
                    file_name="Wb.npy",
                    mime="application/octet-stream",
                )
                st.download_button(
                    label="Скачать Смещение bf",
                    data=bf_bytes,
                    file_name="bf.npy",
                    mime="application/octet-stream",
                )
                st.download_button(
                    label="Скачать Смещение bb",
                    data=bb_bytes,
                    file_name="bb.npy",
                    mime="application/octet-stream",
                )

                # Итоговая статистика
                st.subheader("Итоговые Результаты")
                st.write(f"- **Время обучения:** {training_time:.2f} секунд")
                st.write(f"- **Количество эпох:** {epochs}")
                st.write(f"- **Начальная ошибка:** {errors[0]:.6f}")
                st.write(f"- **Конечная ошибка:** {errors[-1]:.6f}")
                st.write(f"- **Среднеквадратичная ошибка (MSE):** {mse:.6f}")
                st.write(f"- **Пиковое отношение сигнал/шум (PSNR):** {psnr:.2f} dB")
                st.write(f"- **Структурное сходство (SSIM):** {ssim_val:.4f}")
    else:
        st.info("Пожалуйста, загрузите изображение для начала работы.")

if __name__ == "__main__":
    main()
