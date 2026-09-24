from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOURCE = Path("data_4W_sem.csv")
OUTPUT = Path("signal_analysis")
BASELINE_BINS = 80
SIMILARITY_TOLERANCE = 0.005
OUTLIER_THRESHOLD_MAD = 5.0
AMPLITUDE_RANGES = ((0, 50), (50, 100), (100, 150), (150, 200))


def signal_features(signal: np.ndarray) -> tuple[float, int, int, float, float]:
    """Return amplitude, peak bin, FWHM, normalized area and tail ratio."""
    amplitude = float(signal.max())
    peak = int(signal.argmax())
    half_height = amplitude / 2

    left_candidates = np.flatnonzero(signal[: peak + 1] <= half_height)
    left = int(left_candidates[-1]) if left_candidates.size else 0
    right_candidates = np.flatnonzero(signal[peak:] <= half_height)
    right = peak + int(right_candidates[0]) if right_candidates.size else len(signal) - 1
    fwhm = right - left

    positive = np.clip(signal, 0, None)
    normalized_area = float(positive.sum() / max(amplitude, 1e-12))
    leading_area = float(positive[max(0, peak - 50) : peak + 1].sum())
    trailing_area = float(positive[peak : min(len(signal), peak + 150)].sum())
    tail_ratio = trailing_area / max(leading_area, 1e-12)
    return amplitude, peak, fwhm, normalized_area, tail_ratio


def robust_z_scores(values: np.ndarray) -> np.ndarray:
    median = np.median(values, axis=0)
    mad = 1.4826 * np.median(np.abs(values - median), axis=0)
    mad[mad == 0] = 1.0
    return np.abs((values - median) / mad)


def choose_dense_reference(amplitudes: np.ndarray, mask: np.ndarray) -> tuple[int, np.ndarray]:
    """Choose the signal having the largest 0.5%-amplitude neighbourhood."""
    candidates = np.flatnonzero(mask)
    if not candidates.size:
        raise ValueError("Empty amplitude range")
    counts = np.array(
        [
            np.sum(mask & (np.abs(amplitudes - amplitudes[i]) <= SIMILARITY_TOLERANCE * amplitudes[i]))
            for i in candidates
        ]
    )
    best = candidates[np.argmax(counts)]
    group = mask & (np.abs(amplitudes - amplitudes[best]) <= SIMILARITY_TOLERANCE * amplitudes[best])
    return int(best), group


def save_outlier_plot(signals: np.ndarray, features: pd.DataFrame) -> None:
    selected = features.index[features["is_outlier"]].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx in selected:
        axes[0].plot(signals[idx], lw=1.8, label=f"форма {idx}, A={features.loc[idx, 'amplitude']:.2f}")
        axes[1].plot(
            signals[idx] / features.loc[idx, "amplitude"],
            lw=1.8,
            label=f"форма {idx}",
        )
    axes[0].set_title("Выбросы: сигнал после вычитания базовой линии")
    axes[1].set_title("Выбросы: форма после нормировки по амплитуде")
    for ax in axes:
        ax.set_xlabel("Отсчёт")
        ax.set_ylabel("Амплитуда")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT / "01_outlier_shapes.png", dpi=180)
    plt.close(fig)


def save_feature_plot(features: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    normal = features[~features["is_outlier"]]
    outlier = features[features["is_outlier"]]
    axes[0].hist(normal["amplitude"], bins=40, color="#4c78a8", alpha=0.85)
    axes[0].scatter(outlier["amplitude"], np.zeros(len(outlier)), color="#e45756", s=60, zorder=3)
    axes[0].set_title("Распределение амплитуд уникальных сигналов")
    axes[0].set_xlabel("Амплитуда")
    axes[0].set_ylabel("Количество")
    axes[1].scatter(normal["amplitude"], normal["fwhm"], s=18, alpha=0.55, label="обычные")
    axes[1].scatter(outlier["amplitude"], outlier["fwhm"], s=70, color="#e45756", label="выбросы")
    axes[1].set_title("Ширина импульса и амплитуда")
    axes[1].set_xlabel("Амплитуда")
    axes[1].set_ylabel("FWHM, отсчёты")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "02_amplitude_and_width.png", dpi=180)
    plt.close(fig)


def save_similar_plot(
    signals: np.ndarray, features: pd.DataFrame, reference: int, group_mask: np.ndarray
) -> None:
    selected = np.flatnonzero(group_mask)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx in selected:
        label = f"форма {idx}, A={features.loc[idx, 'amplitude']:.2f}"
        axes[0].plot(signals[idx], alpha=0.8, lw=1.4, label=label)
        axes[1].plot(signals[idx] / features.loc[idx, "amplitude"], alpha=0.8, lw=1.4)
    ref_amp = features.loc[reference, "amplitude"]
    axes[0].set_title(f"Амплитуды в пределах ±0,5% от {ref_amp:.2f}")
    axes[1].set_title("Те же сигналы после нормировки")
    for ax in axes:
        ax.set_xlabel("Отсчёт")
        ax.set_ylabel("Амплитуда")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT / "03_similar_amplitude_shapes.png", dpi=180)
    plt.close(fig)


def save_range_plot(signals: np.ndarray, features: pd.DataFrame, groups: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    for ax, entry in zip(axes.ravel(), groups):
        selected = np.flatnonzero(entry["mask"])
        for idx in selected:
            ax.plot(
                signals[idx] / features.loc[idx, "amplitude"],
                lw=1.2,
                alpha=0.8,
                label=f"{features.loc[idx, 'amplitude']:.2f}",
            )
        lo, hi = entry["range"]
        ax.set_title(
            f"Диапазон {lo}–{hi}; опорная A={entry['reference_amplitude']:.2f}; "
            f"форм: {len(selected)}"
        )
        ax.set_xlabel("Отсчёт")
        ax.set_ylabel("Нормированная амплитуда")
        ax.grid(alpha=0.25)
        ax.legend(title="A", fontsize=7, title_fontsize=8)
    fig.suptitle("Форма сигналов с близкой амплитудой в разных диапазонах", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUTPUT / "04_ranges_similar_shapes.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    raw = pd.read_csv(SOURCE, header=None)
    values = raw.to_numpy(dtype=float)

    # Work on unique waveforms so repeated rows do not dominate the analysis.
    first_occurrence = ~raw.duplicated()
    unique_raw = raw.loc[first_occurrence]
    original_indices = unique_raw.index.to_numpy()
    unique_values = unique_raw.to_numpy(dtype=float)
    baselines = np.median(unique_values[:, :BASELINE_BINS], axis=1)
    signals = unique_values - baselines[:, None]

    metric_names = ["amplitude", "peak_bin", "fwhm", "normalized_area", "tail_ratio"]
    metric_values = np.array([signal_features(row) for row in signals])
    feature_table = pd.DataFrame(metric_values, columns=metric_names)
    feature_table["peak_bin"] = feature_table["peak_bin"].astype(int)
    feature_table["fwhm"] = feature_table["fwhm"].astype(int)
    feature_table["robust_distance"] = robust_z_scores(metric_values).max(axis=1)
    feature_table["is_outlier"] = feature_table["robust_distance"] > OUTLIER_THRESHOLD_MAD
    feature_table.insert(0, "unique_shape_id", np.arange(len(feature_table)))
    feature_table.insert(1, "first_original_index", original_indices)

    keys = [row.tobytes() for row in np.ascontiguousarray(values)]
    unique_keys = [row.tobytes() for row in np.ascontiguousarray(unique_values)]
    key_to_unique = {key: idx for idx, key in enumerate(unique_keys)}
    unique_ids = np.array([key_to_unique[key] for key in keys])
    repeat_counts = np.bincount(unique_ids, minlength=len(unique_values))
    feature_table["repeat_count"] = repeat_counts

    good = ~feature_table["is_outlier"].to_numpy()
    amplitudes = feature_table["amplitude"].to_numpy()
    reference, similar_mask = choose_dense_reference(amplitudes, good)

    range_groups: list[dict] = []
    range_rows: list[dict] = []
    for lo, hi in AMPLITUDE_RANGES:
        range_mask = good & (amplitudes >= lo) & (amplitudes < hi)
        ref, group_mask = choose_dense_reference(amplitudes, range_mask)
        range_groups.append(
            {
                "range": (lo, hi),
                "reference": ref,
                "reference_amplitude": amplitudes[ref],
                "mask": group_mask,
            }
        )
        for idx in np.flatnonzero(group_mask):
            range_rows.append(
                {
                    "amplitude_range": f"{lo}-{hi}",
                    "reference_shape_id": ref,
                    "reference_amplitude": amplitudes[ref],
                    "unique_shape_id": idx,
                    "amplitude": amplitudes[idx],
                    "relative_difference_pct": 100 * abs(amplitudes[idx] - amplitudes[ref]) / amplitudes[ref],
                    "repeat_count": repeat_counts[idx],
                }
            )

    feature_table.to_csv(OUTPUT / "unique_signal_metrics.csv", index=False)
    feature_table.loc[feature_table["is_outlier"]].to_csv(OUTPUT / "outlier_unique_shapes.csv", index=False)

    all_metrics = pd.DataFrame(
        {
            "original_index": np.arange(len(values)),
            "unique_shape_id": unique_ids,
        }
    ).merge(
        feature_table[
            [
                "unique_shape_id",
                "amplitude",
                "peak_bin",
                "fwhm",
                "normalized_area",
                "tail_ratio",
                "robust_distance",
                "is_outlier",
            ]
        ],
        on="unique_shape_id",
        how="left",
    )
    all_metrics.to_csv(OUTPUT / "all_signal_metrics.csv", index=False)

    similar_rows = feature_table.loc[similar_mask].copy()
    similar_rows["reference_shape_id"] = reference
    similar_rows["reference_amplitude"] = amplitudes[reference]
    similar_rows["relative_difference_pct"] = (
        100 * np.abs(similar_rows["amplitude"] - amplitudes[reference]) / amplitudes[reference]
    )
    similar_rows.to_csv(OUTPUT / "similar_amplitude_group.csv", index=False)
    pd.DataFrame(range_rows).to_csv(OUTPUT / "range_groups.csv", index=False)

    save_outlier_plot(signals, feature_table)
    save_feature_plot(feature_table)
    save_similar_plot(signals, feature_table, reference, similar_mask)
    save_range_plot(signals, feature_table, range_groups)

    outlier_rows = all_metrics.loc[all_metrics["is_outlier"], "original_index"].tolist()
    report = f"""# Анализ формы сигналов

Источник: `{SOURCE.name}` — {len(values)} строк по {values.shape[1]} отсчётов. Уникальных форм: {len(unique_values)}.

## Методика

- Базовая линия — медиана первых {BASELINE_BINS} отсчётов.
- Амплитуда — максимум после вычитания базовой линии.
- Форма описана пятью параметрами: амплитуда, положение максимума, FWHM, площадь/амплитуда и отношение площади хвоста к площади фронта.
- Выброс: хотя бы один параметр отклоняется более чем на {OUTLIER_THRESHOLD_MAD:g} робастных стандартных отклонений (MAD) от медианы. Анализ ведётся по уникальным формам, затем результат переносится на повторы.
- Близкая амплитуда: относительное отличие не более {100 * SIMILARITY_TOLERANCE:.1f}% от опорного сигнала.

## Задача 1. Выбросы

Обнаружено {int(feature_table['is_outlier'].sum())} уникальных выброса; с учётом повторов — {len(outlier_rows)} строк.

Индексы исходных строк (нумерация с нуля): {outlier_rows}.

Это слабые импульсы с амплитудами {', '.join(f'{x:.3f}' for x in feature_table.loc[feature_table['is_outlier'], 'amplitude'])}; их максимум расположен существенно раньше основной массы сигналов. График: `01_outlier_shapes.png`.

## Задача 2. Сигналы с амплитудами в пределах 0,5%

Опорная уникальная форма: {reference}; амплитуда {amplitudes[reference]:.3f}. В группу вошло {int(similar_mask.sum())} уникальных форм и {int(repeat_counts[similar_mask].sum())} строк с повторами.

После нормировки по амплитуде формы близки, но сохраняется различие в ширине и скорости спада. Это показывает, что одинаковая амплитуда не гарантирует одинаковую временную форму. График: `03_similar_amplitude_shapes.png`.

## Задача 3. Разные диапазоны амплитуд

| Диапазон | Опорная амплитуда | Уникальных форм в ±0,5% | Строк с повторами |
|---:|---:|---:|---:|
"""
    for item in range_groups:
        count_unique = int(item["mask"].sum())
        count_all = int(repeat_counts[item["mask"]].sum())
        lo, hi = item["range"]
        report += f"| {lo}–{hi} | {item['reference_amplitude']:.3f} | {count_unique} | {count_all} |\n"
    report += """

Нормированные кривые на `04_ranges_similar_shapes.png` показывают систематическое изменение формы: с ростом амплитуды импульсы становятся шире, а хвост — длиннее. Следовательно, сравнивать форму корректнее отдельно внутри амплитудных диапазонов.

## Файлы результатов

- `all_signal_metrics.csv` — метрики и флаг выброса для всех 3000 строк.
- `unique_signal_metrics.csv` — метрики 583 уникальных форм и число повторов.
- `outlier_unique_shapes.csv` — только уникальные выбросы.
- `similar_amplitude_group.csv` — выборка для задачи 2.
- `range_groups.csv` — выборки для задачи 3.
"""
    (OUTPUT / "README.md").write_text(report, encoding="utf-8")

    print(f"Rows: {len(values)}; unique shapes: {len(unique_values)}")
    print(f"Unique outliers: {int(feature_table['is_outlier'].sum())}; rows marked: {len(outlier_rows)}")
    print(f"Task 2 reference amplitude: {amplitudes[reference]:.6f}; unique shapes: {int(similar_mask.sum())}")
    for item in range_groups:
        print(
            f"Range {item['range'][0]}-{item['range'][1]}: "
            f"A={item['reference_amplitude']:.6f}; unique shapes={int(item['mask'].sum())}"
        )


if __name__ == "__main__":
    main()
