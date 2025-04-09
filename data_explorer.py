import sys
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, isnan, when

def create_spark_session():
    return SparkSession.builder.appName("PySpark Data Explorer").getOrCreate()


def read_csv(spark, file_path):
    return spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)


def calculate_missing_values(df):
    missing_counts = {}
    
    for column in df.columns:
        null_count = df.filter( col(column).isNull() |  isnan(col(column)) | (col(column) == "")).count()
        missing_counts[column] = null_count
    
    return missing_counts

def save_json(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def main():

    if len(sys.argv) != 3:
        print("Usage: python mvp_explorer.py <input_csv> <output_json>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:

        spark = create_spark_session()

        df = read_csv(spark, input_file)

        row_count = df.count()
        column_count = len(df.columns)

        missing_values = calculate_missing_values(df)

        report = {
            "metadata": {
                "row_count": row_count,
                "column_count": column_count
            },
            "missing_values": missing_values
        }

        save_json(report, output_file)
        print(f"Report saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
    finally:
        breakpoint()
        if 'spark' in locals():
            spark.stop()

if __name__ == "__main__":
    main()