#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 1 — Log Level Distribution
Counts occurrences of INFO/WARN/ERROR/DEBUG across all log lines,
saves (1) counts CSV, (2) 10 random samples CSV, and (3) a summary TXT.

Usage examples:
  spark-submit problem1.py --input s3://<bucket>/sample/ --output s3://<bucket>/outputs/p1
  spark-submit problem1.py --input s3://<bucket>/data/   --output s3://<bucket>/outputs/p1
"""

import os
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, col, count, rand

def main():
    parser = argparse.ArgumentParser(description="Count log levels in Spark/YARN logs")
    parser.add_argument("--input", required=True, help="Input directory or prefix (local or S3)")
    parser.add_argument("--output", required=True, help="Output base path (local or S3)")
    args = parser.parse_args()

    # Start Spark 
    spark = (
        SparkSession.builder
        .appName("A06-Problem1-LogLevels")
        .getOrCreate()
    )

    # Read raw log lines as a single 'value' column 
    # Accepts wildcards 
    logs_df = (
    spark.read
    .option("recursiveFileLookup", "true")   
    .text(args.input)
)


    # Extract log level token from each line 
    # The regex pulls one of INFO|WARN|ERROR|DEBUG into 'log_level'
    parsed_df = (
        logs_df
        .withColumn("log_level", regexp_extract(col("value"), r"\b(INFO|WARN|ERROR|DEBUG)\b", 1))
        .filter(col("log_level") != "")  
    )

    # Aggregate counts by level 
    counts_df = parsed_df.groupBy("log_level").agg(count("*").alias("count"))

    # Write counts to a single CSV part for easy grading
    (counts_df.coalesce(1)
              .write.mode("overwrite")
              .option("header", True)
              .csv(f"{args.output}/problem1_counts.csv"))

    # Write 10 random samples 
    sample_df = parsed_df.orderBy(rand()).limit(10) \
                         .select(col("value").alias("log_entry"), col("log_level"))
    (sample_df.coalesce(1)
              .write.mode("overwrite")
              .option("header", True)
              .csv(f"{args.output}/problem1_sample.csv"))

    # Build and write a human-readable summary 
    total_lines   = parsed_df.count()
    unique_levels = counts_df.count()
    level_pairs   = {r["log_level"]: r["count"] for r in counts_df.collect()}

    summary_lines = [
        f"Total log lines processed: {total_lines}",
        f"Unique log levels found: {unique_levels}",
        "",
        "Log level distribution:"
    ]
    for lvl, cnt in level_pairs.items():
        summary_lines.append(f"{lvl}: {cnt}")

        local_summary = "problem1_summary.txt"
    with open(local_summary, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    # If output is S3 or S3A, copy the summary file up to the same prefix
    if args.output.startswith(("s3://", "s3a://")):
        # aws cli 只认 s3://，把 s3a:// 转成 s3://
        s3_cli_url = args.output.replace("s3a://", "s3://", 1)
        os.system(f"aws s3 cp {local_summary} {s3_cli_url}/problem1_summary.txt --only-show-errors")

    spark.stop()

if __name__ == "__main__":
    main()
