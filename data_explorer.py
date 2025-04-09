import sys
import json
import traceback

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, countDistinct, isnan, when, min as spark_min, max as spark_max
from pyspark.sql.types import StringType, IntegerType, LongType, DoubleType, BooleanType



##### CONSTANTS ####
CATEGORICAL = "categorical"
DISCRETE = "discrete"
CONTINUOUS = "continuous"
TEXT = "text"


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
                
                if any(hint in column_lower for hint in categorical_hints) or distinct_count / total_rows < 0.1:
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
        
def analyze_categorical_discrete(df, column):
    """Analyze a categorical or discrete column."""
    # Get value frequencies
    value_counts = df.groupBy(column).count().orderBy("count", ascending=False)
    
    # Convert to dictionary (limit to top 100 values for performance)
    values_with_counts = value_counts.limit(100).collect()
    result = {
        "distinct_count": df.select(countDistinct(col(column))).collect()[0][0],
        "value_frequencies": {str(row[0]): row[1] for row in values_with_counts}
    }
    
    return result

def analyze_continuous(df, column, num_bins = 10):
    """Analyze a continuous column."""
    # Get min, max, mean, stddev
    min_max = df.agg(
        spark_min(col(column)).alias("min"),
        spark_max(col(column)).alias("max")
    ).collect()[0]
    
    min_val = min_max["min"]
    max_val = min_max["max"]

    result = {
        "min": min_val,
        "max": max_val
    }

    if min_val is not None and max_val is not None and min_val != max_val:

        try:
            bin_width = (max_val - min_val) / num_bins

            bins = []

            for i in range(num_bins):
                bin_start = min_val + i * bin_width
                bin_end = min_val + (i + 1) * bin_width if i < num_bins - 1 else max_val

                bin_count = df.filter(
                    (col(column) >= bin_start) & (col(column) <= bin_end if i == num_bins-1 else col(column) < bin_end)
                ).count()

                bins.append(
                    {
                        "bin_start": bin_start,
                        "bin_end": bin_end,
                        "count": bin_count
                    }
                )
            result["histogram"] = bins
        except Exception as e:
            print(f"Error creating histogram for column {column}: {e}")
            result["histogram"] = str(e)

    return result

def analyze_text(df, column):
    # Just get a count of non-null values for now
    result = {
        "non_null_count": df.filter(col(column).isNotNull()).count()
    }
    
    return result

def analyze_column(df, column, col_type):

    if col_type in [CATEGORICAL, DISCRETE]:
        return analyze_categorical_discrete(df, column)
    elif col_type == CONTINUOUS:
        return analyze_continuous(df, column)
    elif col_type == TEXT:
        return analyze_text(df, column)
    else:
        return {}

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

        
        column_info = {}
        for column in df.columns:
            col_type = column_types[column]
            analysis = analyze_column(df, column, col_type)

            column_info[column] = {
                "type": col_type,
                "missing_count": missing_values[column],
                "analysis": analysis
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