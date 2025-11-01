"""
Программа тестирует pandas, polars и dask,
измеряет время и примерную память и записывает единый JSON с результатами.
"""

import os
import time # Для вычисления времени
import gc # Сборщик мусора
import json
import statistics # Для удобного вычисления
from pathlib import Path

# Библиотеки для проверки
import pandas as pd
import polars as pl
import dask.dataframe as dd
import psutil

def main():
    input_path = "StudentsPerformance.csv"  # Имя файла-датасета StudentsPerformance.csv data_sme.csv
    outdir = "./bench_results"              # Имя выходного файла
    encoding = "utf-8"                      # Кодировка
    runs = 1000                                # Количество тестов
    sep = ","                               # Разделитель
    
    # Создаем папку, если нет
    os.makedirs(outdir, exist_ok=True)
    
    result_filename = os.path.join(outdir, "bench_result.json")

    # Если нет датасета выводим ошибку в файл
    if not Path(input_path).exists():
        result = {"error": "input file not found", "input": input_path}
        with open(result_filename, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        return 2
    # Словарь с результатами
    results = {
        "input": input_path,
        "runs": runs,
        "separator": sep,
        "libs": {}
    }

    # Библиотеки
    libs = {
        "pandas": read_write_pandas,
        "polars": read_write_polars, 
        "dask": read_write_dask
    }

    # Основной цикл
    for lib_name, lib_func in libs.items():
        # На каждую библиотеку выбираем функцию
        runs_data = []
        # Делаем хотя бы 1 тест
        for i in range(max(1, runs)):
            # Создаем временный файл для записи
            temp_out = os.path.join(outdir, f"tmp_{lib_name}_{i}.csv")
            try:
                # Вызываем функцию отдельно для каждой функции
                res = lib_func(input_path, temp_out, encoding, sep)
                runs_data.append(res)
            except Exception as e:
                runs_data.append({"error": repr(e)})
            
            try:
                if os.path.exists(temp_out):
                    os.remove(temp_out)
            except Exception:
                pass
        
        # Считаем среднее значение по всем успешным запускам
        successful_runs = [r for r in runs_data if "error" not in r]
        if successful_runs:
            # Собираем все значения по каждому параметру
            read_times = [r["read_time_s"] for r in successful_runs]
            write_times = [r["write_time_s"] for r in successful_runs]
            read_mems = [r["read_mem_mb"] for r in successful_runs]
            write_mems = [r["write_mem_mb"] for r in successful_runs]
            
            results["libs"][lib_name] = {
                "read_time_s": statistics.mean(read_times),
                "write_time_s": statistics.mean(write_times),
                "read_mem_mb": statistics.mean(read_mems),
                "write_mem_mb": statistics.mean(write_mems),
                "runs_used": len(successful_runs)  # Сколько запусков усреднялось
            }
        else:
            results["libs"][lib_name] = {"error": "all runs failed"}

    with open(result_filename, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    
    return 0

def measure(func, *args, **kwargs):
    """Измеряет время и память"""
    proc = psutil.Process()
    # Начальное время и память
    mem0 = proc.memory_info().rss
    t0 = time.perf_counter()

    res = func(*args, **kwargs)
    # Время и память после выполнения
    t1 = time.perf_counter()
    mem1 = proc.memory_info().rss
    # Вычисляем время выполнения и средную затраченную память
    elapsed = t1 - t0
    avg_mem = ((mem0 + mem1) / 2) / (1024 * 1024)
    # Очищаем память
    gc.collect()
    return res, elapsed, avg_mem

def read_write_pandas(path: str, out_tmp: str, encoding: str, sep: str):
    """Читаем и записываем на pandas"""
    try:
        # Получаем значения чтения из measure с помощью анонимной функции
        (df, _), t_read, mem_read = measure(lambda: (
            pd.read_csv(path, sep=sep, encoding=encoding, engine="python"), sep))
        # Тоже для записи
        _, t_write, mem_write = measure(lambda: df.to_csv(out_tmp, index=False, encoding=encoding))
        
        return {
            "read_time_s": t_read,
            "write_time_s": t_write,
            "read_mem_mb": mem_read,
            "write_mem_mb": mem_write
        }
    except Exception as e:
        return {"error": repr(e)}

def read_write_polars(path: str, out_tmp: str, encoding: str, sep: str):
    """Читаем и записываем на polars"""
    try:
        (df, _), t_read, mem_read = measure(lambda: (
            pl.read_csv(path, separator=sep, encoding=encoding), sep))
        
        _, t_write, mem_write = measure(lambda: df.write_csv(out_tmp))
        
        return {
            "read_time_s": t_read,
            "write_time_s": t_write,
            "read_mem_mb": mem_read,
            "write_mem_mb": mem_write
        }
    except Exception as e:
        return {"error": repr(e)}

def read_write_dask(path: str, out_tmp: str, encoding: str, sep: str):
    """Читаем и записываем на dask"""
    try:
        (ddf, _), t_read_lazy, mem_read_lazy = measure(lambda: (
            dd.read_csv(path, sep=sep, encoding=encoding), sep))
        
        (pdf,), t_compute, mem_compute = measure(lambda: (ddf.compute(),))
        
        _, t_write, mem_write = measure(lambda: pdf.to_csv(out_tmp, index=False))
        
        return {
            "read_time_s": t_read_lazy + t_compute,
            "write_time_s": t_write,
            "read_mem_mb": max(mem_read_lazy, mem_compute),
            "write_mem_mb": mem_write
        }
    except Exception as e:
        return {"error": repr(e)}

if __name__ == "__main__":
    raise SystemExit(main())