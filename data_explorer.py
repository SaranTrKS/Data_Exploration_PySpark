
import traceback

import sys
import json
from argparse import ArgumentParser, ArgumentTypeError
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, countDistinct, isnan, when, min as spark_min, max as spark_max,
    split, explode, length, lower, size
)
from pyspark.sql.types import StringType, IntegerType, LongType, DoubleType, BooleanType


##### CONSTANTS ####
CATEGORICAL = "categorical"
DISCRETE = "discrete"
CONTINUOUS = "continuous"
TEXT = "text"


def create_spark_session():
    return SparkSession.builder.appName("PySpark Data Explorer").getOrCreate()


def read_csv(spark, args):
    df = spark.read.option("header", str(args.header).lower()) \
                       .option("delimiter", args.delimiter) \
                       .option("inferSchema", "true") \
                       .csv(args.input_file)
    return df


def calculate_missing_values(df):
    missing_counts = {}
    try:
        for column in df.columns:
            null_count = df.filter( col(column).isNull() |  isnan(col(column)) | (col(column) == "")).count()
            missing_counts[column] = null_count
    except Exception as e:
        logging.warning(f"Error calculating missing values for column '{column}': {str(e)}")
        # Default to counting just nulls if the more complex condition fails
        missing_counts[column] = df.filter(col(column).isNull()).count()

    return missing_counts

def detect_column_types(df):
    column_types = {}
    logger = logging.getLogger("DataExplorer")

    for column in df.columns:

        try:

            spark_type = df.schema[column].dataType

            if isinstance(spark_type, StringType):
                distinct_count = df.select(countDistinct(col(column))).collect()[0][0]
                total_rows = df.count()

                if distinct_count <= 20 or (total_rows > 0 and distinct_count / total_rows < 0.1):
                    column_types[column] = CATEGORICAL
                else:
                    try:
                        # Calculate average word count per value
                        word_counts = df.select(size(split(col(column), r'\s+')).alias("words"))
                        avg_words = word_counts.agg({"words": "avg"}).collect()[0][0] or 0
                        
                        # Calculate average string length
                        avg_length = df.select(length(col(column)).alias("length")).agg({"length": "avg"}).collect()[0][0] or 0
                        
                        # If the average contains multiple words or is long, classify as text
                        if avg_words >= 1.5 or avg_length > 100:
                            column_types[column] = TEXT
                        else:
                            column_types[column] = CATEGORICAL
                    except Exception as e:
                        logger.warning(f"Error calculating text metrics for '{column}': {str(e)}")
                        # Default to categorical if text detection fails
                        column_types[column] = CATEGORICAL

            # Numeric Columns
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

            logger.info(f"Column '{column}' classified as {column_types[column]}")

        except Exception as e:
            logger.error(f"Error detecting type for column '{column}': {str(e)}")
            # Default to CATEGORICAL for any errors
            column_types[column] = CATEGORICAL

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
    """Analyze a text column with word count information."""
    try:

        df_filtered = df.filter(col(column).isNotNull())

        if df_filtered.count() == 0:
            return {"non_null_count": 0}
        
        non_null_count = df_filtered.count()

        words_df = df_filtered.select(
            explode(
                split(lower(col(column)), r'\W+')
            ).alias("word")
        )
        # Count non-empty words
        words_df = words_df.filter(length(col("word")) > 0)
        # Count word frequencies
        word_counts = words_df.groupby("word").count().orderBy("count", ascending=False)
        # Get top words
        top_words = word_counts.limit(20).collect()

        # count total words
        total_words = words_df.count()
        unique_words = word_counts.count()

        result  = {
            "non_null_count": non_null_count,
            "total_words": total_words,
            "unique_words": unique_words,
            "top_words": {row["word"]: row["count"] for row in top_words}
        }

        return result
    except Exception as e:
        print(f"Error analyzing text column {column}: {e}")
        return {
            "non_null_count": df.filter(col(column).isNotNull()).count(),
            "analysis_error": str(e)
        }


def analyze_column(df, column, col_type, num_bins=10):
    if col_type in [CATEGORICAL, DISCRETE]:
        return analyze_categorical_discrete(df, column)
    elif col_type == CONTINUOUS:
        return analyze_continuous(df, column, num_bins)
    elif col_type == TEXT:
        return analyze_text(df, column)
    else:
        return {}

def save_json(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)


def is_csv_file(filepath):
    if not filepath.lower().endswith('.csv'):
        raise ArgumentTypeError(f"{filepath} is not a valid CSV file.")
    return filepath

def is_json_file(filepath):
    if not filepath.lower().endswith('.json'):
        raise ArgumentTypeError(f"{filepath} is not a valid JSON file.")
    return filepath


def main():

    parser = ArgumentParser()
    parser.add_argument("-i", "--input-file", type=is_csv_file, dest="input_file", required=True, help="Path to the input CSV file")
    parser.add_argument("-o", "--output-file", type=is_json_file, dest="output_file", required=True, help="Path to save the JSON report")
    parser.add_argument("-b", "--num-bins", type=int, dest="num_bins", default=10, help="Number of bins for histograms (default: 10)")

    parser.add_argument("--header", dest="header", action="store_true", help="Specify if the CSV has a header row")
    parser.add_argument("--no-header", dest="header", action="store_false", help="Specify if the CSV does not have a header row")
    parser.set_defaults(header=True)
    parser.add_argument("-d", "--delimiter", type=str, dest="delimiter", default=",", help="CSV delimiter character (default: ',')")
    parser.add_argument("-s", "--sample-size", type=int, dest="sample_size", default=1000, help="Number of rows to sample for analysis (default: 1000)")
    parser.add_argument("-D", "--debug", action="store_true", dest="debug", help="Enable debug logging")

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("DataExplorer")

    try:

        logger.info(f"[Logger-Log] - Creating Spark session")
        spark = create_spark_session()

        logger.info(f"[Logger-Log] - Reading CSV file: {args.input_file}")
        df = read_csv(spark, args)

        # Sample if needed
        if args.sample_size > 0 and df.count() > args.sample_size:
            logger.info(f"[Logger-Log] - Sampling {args.sample_size} rows for analysis")
            df_sample = df.sample(fraction=args.sample_size/df.count(), seed=42)
        else:
            df_sample = df

        logger.info("C[Logger-Log] - alculating row and column counts")
        row_count = df.count()
        column_count = len(df.columns)

        logger.info("C[Logger-Log] - alculating missing values")
        missing_values = calculate_missing_values(df)

        logger.info("D[Logger-Log] - etecting column types")
        column_types = detect_column_types(df_sample)

        logger.info("A[Logger-Log] - nalyzing columns")
        column_info = {}
        for column in df.columns:
            logger.info(f"[Logger-Log] - Analyzing column: {column}")
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

        output_file = args.output_file
        logger.info(f"[Logger-Log] - Saving report to {output_file}")
        save_json(report, output_file)
        print(f"Report saved to {output_file}")

        logger.info("A[Logger-Log] - nalysis completed successfully")

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        print(f"An error occurred: {e}")
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
    finally:
        # breakpoint()
        if 'spark' in locals():
            logger.info("S[Logger-Log] - topping Spark session")
            spark.stop()

if __name__ == "__main__":
    main()