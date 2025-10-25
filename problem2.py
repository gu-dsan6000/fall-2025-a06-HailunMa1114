#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 2 — Cluster Usage Analysis

Outputs (under data/output/):
  - problem2_timeline.csv           : per-application timeline (Spark writes a DIR with part-*.csv)
  - problem2_cluster_summary.csv    : per-cluster summary (DIR with part-*.csv)
  - problem2_stats.txt              : overall stats (plain text)
  - problem2_bar_chart.png          : bar chart (applications per cluster)
  - problem2_density_plot.png       : duration distribution for the largest cluster
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def build_spark(app_name: str, master: str):
    """Create a SparkSession with a given master URL."""
    from pyspark.sql import SparkSession
    b = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
    )
    return b.getOrCreate()


def parse_args():
    p = argparse.ArgumentParser(description="Analyze cluster usage over time")
    p.add_argument("master", nargs="?", default="local[*]",
                   help="Spark master URL (e.g., spark://host:7077). Default: local[*]")
    p.add_argument("--net-id", default="netid",
                   help="Your NETID (only used to tag the Spark app name)")
    p.add_argument("--input", default="data/raw",
                   help="Base input folder containing raw logs (local or S3A).")
    p.add_argument("--output", default="data/output",
                   help="Output folder for CSVs and figures")
    p.add_argument("--skip-spark", action="store_true",
                   help="Skip Spark and only regenerate charts from existing CSVs")
    return p.parse_args()


def run_with_spark(master: str, net_id: str, input_dir: str, output_dir: str):
    """
    Read all log lines recursively and construct:
      1) Application timeline (cluster_id, application_id, app_number, start_time, end_time)
      2) Cluster summary (num apps, first/last timestamps)
    Notes:
      - application directory name looks like 'application_<clusterId>_<seq>'
      - we derive cluster_id from application_id
      - timestamps appear in multiple formats; we collect the raw string and normalize later in pandas
    """
    import pyspark.sql.functions as F
    from pyspark.sql.functions import col, regexp_extract, min as min_, max as max_, input_file_name
    from pyspark.sql.window import Window

    spark = build_spark(f"A06-Problem2-{net_id}", master)
    spark.sparkContext.setLogLevel("WARN")

    # Read all files 
    df = (
        spark.read
        .option("recursiveFileLookup", "true")
        .text(input_dir)
        .withColumn("path", input_file_name())
    )

    app_re = r".*/(application_\d+_\d+)/.*"
    df = df.withColumn("application_id", regexp_extract(col("path"), app_re, 1))

    df = df.withColumn("cluster_id", regexp_extract(col("application_id"), r"application_(\d+)_\d+", 1))

    ts_re = r"((\d{4}[-/]\d{2}[-/]\d{2}|\d{2}/\d{2}/\d{2}) \d{2}:\d{2}:\d{2})"
    df = df.withColumn("ts", regexp_extract(col("value"), ts_re, 1))
    df = df.filter((col("cluster_id") != "") & (col("application_id") != "") & (col("ts") != ""))

    # per-application
    per_app = (
        df.groupBy("cluster_id", "application_id")
          .agg(min_("ts").alias("start_time"),
               max_("ts").alias("end_time"))
    )

    # assign app_number per cluster by sorting application_id lexicographically
    w = Window.partitionBy("cluster_id").orderBy("application_id")
    per_app = per_app.withColumn("app_number", F.row_number().over(w))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save timeline 
    (
        per_app.select("cluster_id", "application_id", "app_number", "start_time", "end_time")
              .coalesce(1)
              .write.mode("overwrite")
              .option("header", True)
              .csv(str(out / "problem2_timeline.csv"))
    )

    # Cluster summary
    cluster_summary = (
        per_app.groupBy("cluster_id")
               .agg(F.count("*").alias("num_applications"),
                    F.min("start_time").alias("cluster_first_app"),
                    F.max("end_time").alias("cluster_last_app"))
    )

    (
        cluster_summary.coalesce(1)
                       .write.mode("overwrite")
                       .option("header", True)
                       .csv(str(out / "problem2_cluster_summary.csv"))
    )

    # Simple overall stats (bring to driver)
    pdf = cluster_summary.orderBy(F.desc("num_applications")).toPandas()
    total_clusters = len(pdf)
    total_apps = int(pdf["num_applications"].sum()) if total_clusters else 0
    avg = (total_apps / total_clusters) if total_clusters else 0.0

    lines = [
        f"Total unique clusters: {total_clusters}",
        f"Total applications: {total_apps}",
        f"Average applications per cluster: {avg:.2f}",
        "",
        "Most heavily used clusters:"
    ]
    for _, row in pdf.head(10).iterrows():
        lines.append(f"  Cluster {row['cluster_id']}: {int(row['num_applications'])} applications")

    (out / "problem2_stats.txt").write_text("\n".join(lines) + "\n")

    spark.stop()

def _pick_part_csv(dir_path: Path) -> Path:
    """Pick a part-*.csv file from a Spark output directory."""
    parts = sorted(dir_path.glob("part-*.csv"))
    if not parts:
        raise FileNotFoundError(f"No part-*.csv under {dir_path}")
    return parts[0]


def make_plots(output_dir: str):
    """
    Regenerate both visualizations from saved CSVs:
      - Bar chart: applications per cluster
      - Duration histogram (log x-axis): for the cluster with most apps
    """
    out = Path(output_dir)

    # Timeline / summary may be directories (Spark) or already flattened single files.
    tl_dir = out / "problem2_timeline.csv"
    cs_dir = out / "problem2_cluster_summary.csv"
    tl_csv = _pick_part_csv(tl_dir) if tl_dir.is_dir() else tl_dir
    cs_csv = _pick_part_csv(cs_dir) if cs_dir.is_dir() else cs_dir

    timeline = pd.read_csv(tl_csv)
    summary  = pd.read_csv(cs_csv)

    # Plot 
    plt.figure()
    ordered = summary.sort_values("num_applications", ascending=False)
    bars = plt.bar(ordered["cluster_id"].astype(str), ordered["num_applications"])
    for b in bars:
        plt.text(b.get_x() + b.get_width()/2, b.get_height(), f"{int(b.get_height())}",
                 ha="center", va="bottom")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Cluster ID")
    plt.ylabel("Applications")
    plt.title("Applications per Cluster")
    plt.tight_layout()
    plt.savefig(out / "problem2_bar_chart.png", dpi=160)
    plt.close()

    # Plot 2
    timeline["start_time"] = pd.to_datetime(timeline["start_time"], errors="coerce")
    timeline["end_time"]   = pd.to_datetime(timeline["end_time"], errors="coerce")
    timeline["duration_s"] = (timeline["end_time"] - timeline["start_time"]).dt.total_seconds()
    timeline = timeline.dropna(subset=["duration_s"])

    if len(timeline) == 0:
        print("No durations available; skipping density plot.")
        return

    top_cluster = (timeline.groupby("cluster_id")["application_id"]
                          .count()
                          .sort_values(ascending=False)
                          .index[0])
    subset = timeline[timeline["cluster_id"] == top_cluster].copy()

    plt.figure()
    plt.hist(subset["duration_s"], bins=30, alpha=0.7)
    plt.xscale("log")
    plt.xlabel("Duration (seconds, log scale)")
    plt.ylabel("Count")
    plt.title(f"Duration Distribution — Cluster {top_cluster} (n={len(subset)})")
    plt.tight_layout()
    plt.savefig(out / "problem2_density_plot.png", dpi=160)
    plt.close()


def main():
    args = parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)

    if not args.skip_spark:
        run_with_spark(args.master, args.net_id, args.input, args.output)
        
    make_plots(args.output)
    print("Problem 2 outputs are in:", args.output)


if __name__ == "__main__":
    main()

