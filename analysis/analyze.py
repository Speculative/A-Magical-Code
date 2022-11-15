import os.path as path
import re
from os import listdir, makedirs
from csv import reader
from time import time
import matplotlib.pyplot as plt
import numpy as np
from pickle import load, dump

PRECOMP_FILE = "stats.pkl"
RESULTS_DIR = "A_Magical_Code_tournament_results"
PLOT_DIR = "plots"

RESULT_PATTERN = re.compile(
    r"results_(?P<group>\d)_(?P<domain>\d)_(?P<batch>\d)_(?P<n>\d+)\.csv"
)
NULL_PATTERN = re.compile(r"results_(?P<group>\d)_nulls_(?P<n>\d+)\.csv")
COLS = ["agent", "seed", "n", "message", "decoded", "score"]
COL_TYPES = [int, int, int, str, str, float]
GROUPS = list(range(1, 9))
DOMAINS = [str(i) for i in range(1, 9)] + ["null"]
NS = list(range(0, 101, 10))

if path.exists(PRECOMP_FILE):
    with open(PRECOMP_FILE, "rb") as f:
        per_group_domain_n = load(f)
else:
    results = {group: {domain: [] for domain in DOMAINS} for group in GROUPS}
    lost_rows = 0
    total_rows = 0

    load_start = time()
    for file in listdir(RESULTS_DIR):
        if match := RESULT_PATTERN.match(file):
            group = int(match.group("group"))
            domain = match.group("domain")
            n = int(match.group("n"))
        elif match := NULL_PATTERN.match(file):
            group = int(match.group("group"))
            domain = "null"
            n = match.group("n")
        else:
            print("Failed to match", file)
            continue

        with open(path.join(RESULTS_DIR, file), newline="") as f:
            try:
                csvfile = reader(f)
                # Skip header row
                next(csvfile)
                for line_no, fields in enumerate(csvfile):
                    total_rows += 1
                    try:
                        if len(fields) != len(COLS):
                            lost_rows += 1
                            continue
                        converted_fields = [
                            convert(field) for convert, field in zip(COL_TYPES, fields)
                        ]
                        result = {
                            col: value for col, value in zip(COLS, converted_fields)
                        }
                        results[group][domain].append(result)
                    except Exception as e:
                        lost_rows += 1
                        print(f"Exception when processing row {file}:{line_no}", e)
            except Exception as e:
                print(f"Exception when reading file {file}", e)
                continue
    load_end = time()

    print("Total rows:", total_rows)
    print("Lost rows:", lost_rows)
    print("Load time:", round(load_end - load_start))

    compute_start = time()
    per_group_domain_n: dict[int, dict[str, dict[int, tuple[int, float]]]] = {
        group: {domain: {n: np.array([0, 0.0]) for n in NS} for domain in DOMAINS}
        for group in GROUPS
    }
    for group in GROUPS:
        for domain in DOMAINS:
            for result in results[group][domain]:
                n = result["n"]
                per_group_domain_n[group][domain][n] += np.array([1, result["score"]])
    compute_end = time()
    print("Crunched numbers in", round(compute_end - compute_start))

    with open(PRECOMP_FILE, "w+b") as f:
        dump(per_group_domain_n, f)


makedirs(PLOT_DIR, exist_ok=True)
avg_per_group = [
    sum(per_group_domain_n[group][domain][n][1] for n in NS for domain in DOMAINS)
    / max(
        sum(per_group_domain_n[group][domain][n][0] for n in NS for domain in DOMAINS),
        1,
    )
    for group in GROUPS
]
plt.bar(GROUPS, avg_per_group)
plt.gca().set_ylabel("Average Score")
plt.gca().set_xlabel("Group")
plt.title("Average Score Per-Group")
plt.savefig(path.join(PLOT_DIR, "per_group.png"), dpi=300)

avg_per_group_domain = {
    group: {
        domain: sum(per_group_domain_n[group][domain][n][1] for n in NS)
        / max(sum(per_group_domain_n[group][domain][n][0] for n in NS), 1)
        for domain in DOMAINS
    }
    for group in GROUPS
}
plt.clf()
for group in GROUPS:
    num_bar_groups = len(DOMAINS)
    num_bars = len(GROUPS)
    x = np.arange(num_bar_groups)
    width = 0.1
    bars = plt.bar(
        x - (width * num_bars) / 2 + (group * width),
        [avg_per_group_domain[group][domain] for domain in DOMAINS],
        width,
        label=f"G{group}",
    )
plt.title(f"Average Score Per-Group Per-Domain (all n)")
plt.gca().set_ylabel("Average Score")
plt.gca().set_xticks(x, DOMAINS)
plt.gca().set_xlabel("Domain")
plt.gca().legend()
plt.savefig(path.join(PLOT_DIR, "per_group_per_domain.png"), dpi=300)

for n in NS:
    avg_per_group_per_domain = {
        group: {
            domain: per_group_domain_n[group][domain][n][1]
            / max(per_group_domain_n[group][domain][n][0], 1)
            for domain in DOMAINS
        }
        for group in GROUPS
    }
    plt.clf()
    for group in GROUPS:
        num_bar_groups = len(DOMAINS)
        num_bars = len(GROUPS)
        x = np.arange(num_bar_groups)
        width = 0.1
        bars = plt.bar(
            x - (width * num_bars) / 2 + (group * width),
            [avg_per_group_per_domain[group][domain] for domain in DOMAINS],
            width,
            label=f"G{group}",
        )
    plt.title(f"Average Score Per-Group Per-Domain (n={n})")
    plt.gca().set_ylabel("Average Score")
    plt.gca().set_xticks(x, DOMAINS)
    plt.gca().set_xlabel("Domain")
    plt.gca().legend()
    plt.savefig(path.join(PLOT_DIR, f"per_group_per_domain_n={n}.png"), dpi=300)

for domain in DOMAINS:
    avg_per_group_per_n = {
        group: {
            n: per_group_domain_n[group][domain][n][1]
            / max(per_group_domain_n[group][domain][n][0], 1)
            for n in NS
        }
        for group in GROUPS
    }
    plt.clf()
    for group in GROUPS:
        num_bar_groups = len(NS)
        num_bars = len(GROUPS)
        x = np.arange(num_bar_groups)
        width = 0.1
        bars = plt.bar(
            x - (width * num_bars) / 2 + (group * width),
            [avg_per_group_per_n[group][n] for n in NS],
            width,
            label=f"G{group}",
        )
    plt.title(f"Average Score Per-Group Per-n (domain={domain})")
    plt.gca().set_ylabel("Average Score")
    plt.gca().set_xticks(x, NS)
    plt.gca().set_xlabel("n")
    plt.gca().legend()
    plt.savefig(path.join(PLOT_DIR, f"per_group_per_n_domain={domain}.png"), dpi=300)
