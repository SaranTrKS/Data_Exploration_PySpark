import sys
import json
import traceback

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, countDistinct, isnan, when
from pyspark.sql.types import StringType, IntegerType, LongType, DoubleType, BooleanType


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

def detect_column_types(df):

    CATEGORICAL = "categorical"
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    TEXT = "text"

    column_types = {}

    for column in df.columns:

        spark_type = df.schema[column].dataType

        if isinstance(spark_type, StringType):

            distinct_count = df.select(countDistinct(col(column))).collect()[0][0]
            total_rows = df.count()

            if distinct_count <= 20 or (total_rows > 0 and distinct_count / total_rows < 0.1):
                column_types[column] = CATEGORICAL
            else:
                column_types[column] = TEXT

        elif isinstance(spark_type, (IntegerType, LongType)):
           
            distinct_count = df.select(countDistinct(col(column))).collect()[0][0]
            total_rows = df.count()
            
            # Integer columns with few unique values could be categorical or discrete
            if distinct_count <= 20:
                # Check if this looks like a category code
                # Column names containing 'id', 'code', 'type', 'category', 'class' hint at categorical
                column_lower = column.lower()
                categorical_hints = ['id', 'code', 'type', 'category', 'class', 'status', 'level', 'grade']
                
                if any(hint in column_lower for hint in categorical_hints) or distinct_count <= 5:
                    column_types[column] = CATEGORICAL
                else:
                    column_types[column] = DISCRETE
            else:
                # Many distinct values, likely continuous
                column_types[column] = CONTINUOUS
        
        elif isinstance(spark_type, DoubleType):
            column_types[column] = CONTINUOUS
        elif isinstance(spark_type, BooleanType):
            column_types[column] = CATEGORICAL
        
        else:
            column_types[column] = CATEGORICAL # default to categorical for unknown types

    
    return column_types
        



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

        column_types = detect_column_types(df)
        # breakpoint()
        column_info = {}
        for column in df.columns:
            column_info[column] = {
                "type": column_types[column],
                "missing_count": missing_values[column]
            }

        report = {
            "metadata": {
                "row_count": row_count,
                "column_count": column_count
            },
            "columns": column_info
        }

        save_json(report, output_file)

        print(f"Report saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
    finally:
        # breakpoint()
        if 'spark' in locals():
            spark.stop()

if __name__ == "__main__":
    main()