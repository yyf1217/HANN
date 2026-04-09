#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计以下目录中的实验结果文件最后一行的 val / test，并输出为一个 Excel 文件：
- result_count_add
- result_count_add_5d
- result_count_add_5d_2
- result_count_baseline

要求：
1. 只统计 MODELS = {"L", "LF"} 的结果；
2. 第1列为 {model}-{task}；
3. 第2列为 source（baseline / result_count_add / result_count_add_5d / result_count_add_5d_2）；
4. 第3列为 val，第4列为 test；
5. 若某一 our 版本的 val 或 test 比 baseline 更大，则对应单元格标红。

示例：
    python collect_results_multi.py \
        --baseline_dir result.count_baseline \
        --our_dirs result_count_add result_count_add_5d result_count_add_5d_2 \
        --output summary_results_multi.xlsx
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

MODELS = {"L", "LF"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline_dir",
        type=str,
        default="result_count_baseline",
        help="baseline result directory",
    )
    parser.add_argument(
        "--our_dirs",
        nargs="+",
        default=["result_count_add", "result_count_add_5d", "result_count_add_5d_2", "result_count_add_5d_3", "result_count_add_5d_4", "result_count_add_5d_5", "result_count_add_5d_6", "result_count_fuse_5d", "result_count_fuse_5d_2"],
        help="our result directories",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="summary_results_multi5.xlsx",
        help="output xlsx path",
    )
    return parser.parse_args()


def split_filename_stem(stem: str) -> Tuple[str, str]:
    """
    从文件名 stem 中解析出:
        model-task_key

    例如：
        L-['boat', 'g']-5-5x96-512-0.001-1
        -> model='L', task_key='boat-g'

        LF-cycle3-g-5-5x96-512-0.001-1
        -> model='LF', task_key='cycle3-g'
    """
    parts = stem.split("-")
    if not parts:
        raise ValueError(f"Invalid filename stem: {stem}")

    model = parts[0]
    if model not in MODELS:
        raise ValueError(f"Skip unsupported model in filename: {stem}")

    if len(parts) >= 2:
        second = parts[1]
        try:
            parsed = ast.literal_eval(second)
            if isinstance(parsed, list) and len(parsed) == 2:
                task_name = f"{parsed[0]}-{parsed[1]}"
                return model, task_name
        except Exception:
            pass

    if len(parts) >= 3:
        task_name = f"{parts[1]}-{parts[2]}"
        return model, task_name

    raise ValueError(f"Cannot parse task info from filename: {stem}")


def read_last_val_test(file_path: Path) -> Tuple[float, float]:
    """
    每个 txt 文件每行格式为: lr \t loss \t val \t test
    读取最后一行的 val 和 test
    """
    last_line: Optional[str] = None
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line

    if last_line is None:
        raise ValueError(f"Empty file: {file_path}")

    cols = re.split(r"\s+", last_line)
    if len(cols) < 4:
        raise ValueError(f"Invalid last line in {file_path}: {last_line}")

    val = float(cols[-2])
    test = float(cols[-1])
    return val, test


def collect_one_dir(result_dir: str) -> Dict[str, Tuple[float, float]]:
    """
    返回：
        {
            "L-boat-g": (val, test),
            "LF-cycle3-g": (val, test),
            ...
        }
    只保留 MODELS 中的模型结果。
    """
    path = Path(result_dir)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {result_dir}")

    results: Dict[str, Tuple[float, float]] = {}

    for file_path in sorted(path.glob("*.txt")):
        try:
            model, task = split_filename_stem(file_path.stem)
        except ValueError:
            continue

        key = f"{model}-{task}"
        val, test = read_last_val_test(file_path)
        results[key] = (val, test)

    return results


def sort_key(exp_key: str) -> Tuple[str, str]:
    model, task = exp_key.split("-", 1)
    return task, model


def write_excel(
    output_path: str,
    baseline_results: Dict[str, Tuple[float, float]],
    our_results_by_dir: Dict[str, Dict[str, Tuple[float, float]]],
) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "summary"

    ws.append(["model-task", "source", "val", "test"])

    all_keys = set(baseline_results.keys())
    for results in our_results_by_dir.values():
        all_keys.update(results.keys())
    all_keys = sorted(all_keys, key=sort_key)

    source_order = ["baseline"] + list(our_results_by_dir.keys())

    # 样式
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    center = Alignment(horizontal="center", vertical="center")
    red_fill = PatternFill("solid", fgColor="FFC7CE")
    red_font = Font(color="9C0006")

    for key in all_keys:
        baseline_val, baseline_test = baseline_results.get(key, (None, None))

        for source in source_order:
            if source == "baseline":
                val, test = baseline_val, baseline_test
            else:
                val, test = our_results_by_dir[source].get(key, (None, None))

            ws.append([key, source, val, test])
            row_idx = ws.max_row

            # 对比 baseline：更大则标红（仅 our 版本参与比较）
            if source != "baseline" and baseline_val is not None and val is not None and val > baseline_val:
                ws.cell(row=row_idx, column=3).fill = red_fill
                ws.cell(row=row_idx, column=3).font = red_font

            if source != "baseline" and baseline_test is not None and test is not None and test > baseline_test:
                ws.cell(row=row_idx, column=4).fill = red_fill
                ws.cell(row=row_idx, column=4).font = red_font

    # 表头样式
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center

    # 内容样式
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=1, max_col=4):
        row[0].alignment = Alignment(vertical="center")
        row[1].alignment = center
        row[2].alignment = center
        row[3].alignment = center

    # 数值格式
    for row in range(2, ws.max_row + 1):
        ws.cell(row=row, column=3).number_format = "0.000000"
        ws.cell(row=row, column=4).number_format = "0.000000"

    # 列宽
    widths = {1: 28, 2: 22, 3: 14, 4: 14}
    for col_idx, width in widths.items():
        ws.column_dimensions[get_column_letter(col_idx)].width = width

    ws.freeze_panes = "A2"
    wb.save(output_path)



def main() -> None:
    args = parse_args()

    baseline_results = collect_one_dir(args.baseline_dir)
    our_results_by_dir = {dir_name: collect_one_dir(dir_name) for dir_name in args.our_dirs}

    write_excel(args.output, baseline_results, our_results_by_dir)

    print(f"Saved Excel to: {args.output}")
    print(f"baseline files: {len(baseline_results)}")
    for dir_name, results in our_results_by_dir.items():
        print(f"{dir_name}: {len(results)}")


if __name__ == "__main__":
    main()
