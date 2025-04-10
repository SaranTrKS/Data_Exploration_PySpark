# PySpark Data Explorer

A comprehensive system for exploring and analyzing CSV data using PySpark. This tool automatically detects column types, calculates statistics, and generates detailed JSON reports.

## Features

- **Automatic column type detection:** Categorizes columns as Categorical, Discrete, Continuous, or Text
- **Missing value analysis:** Calculates missing value counts for each column
- **Type-specific analysis:**
  - **Categorical/Discrete:** Extracts distinct values and their frequencies
  - **Continuous:** Calculates min/max values and generates histograms
  - **Text:** Provides word count information and most frequent words
- **Comprehensive error handling:** Gracefully handles exceptions with detailed logging
- **Configurable options:** Sample size, histogram bins, CSV delimiter, etc.

## Requirements

- Python 3.6+
- PySpark 3.0+
- Java 8+ (required for Spark)

## Installation

1. Ensure you have Java installed and `JAVA_HOME` environment variable set.
2. Install PySpark:
   ```bash
   pip install pyspark
   ```
3. Download the `data_explorer.py` script to your local machine.

## Usage

### Basic Usage

```bash
python data_explorer.py -i input.csv -o output.json
```

### Command Line Options

```
-i, --input-file      Path to the input CSV file (required)
-o, --output-file     Path to save the JSON report (required)
-b, --num-bins        Number of bins for histograms (default: 10)
--header              Specify if the CSV has a header row (default: True)
--no-header           Specify if the CSV does not have a header row
-d, --delimiter       CSV delimiter character (default: ',')
-s, --sample-size     Number of rows to sample for analysis (default: 1000)
-D, --debug           Enable debug logging
```

### Examples

Basic analysis:
```bash
python data_explorer.py -i Sample_Input.csv -o report.json
```

Advanced options:
```bash
python data_explorer.py -i Sample_Input.csv -o report.json -b 15 -s 500 -D --no-header -d ";"
```

## Output Format

The tool generates a JSON file with the following structure:

```json
{
  "metadata": {
    "row_count": 1000,
    "column_count": 10
  },
  "columns": {
    "column1": {
      "type": "categorical",
      "missing_count": 5,
      "analysis": {
        "distinct_count": 3,
        "value_frequencies": {
          "value1": 400,
          "value2": 350,
          "value3": 245
        }
      }
    },
    "column2": {
      "type": "continuous",
      "missing_count": 2,
      "analysis": {
        "min": 0.5,
        "max": 100.3,
        "histogram": [
          {
            "bin_start": 0.5,
            "bin_end": 10.5,
            "count": 50
          },
          // more bins...
        ]
      }
    },
    "column3": {
      "type": "text",
      "missing_count": 0,
      "analysis": {
        "non_null_count": 1000,
        "total_words": 5230,
        "unique_words": 1280,
        "top_words": {
          "the": 320,
          "and": 290,
          // more words...
        }
      }
    }
    // more columns...
  }
}
```

## Column Type Detection

The system uses the following heuristics to determine column types:

1. **Categorical:**
   - String columns with low cardinality (≤ 20 distinct values or distinct/total ratio < 0.1)
   - Boolean columns
   - Integer columns with column names containing hints like 'id', 'code', 'type', etc.
   - Integer columns with few distinct values relative to total rows

2. **Discrete:**
   - Integer columns with limited distinct values (≤ 20) that don't match categorical patterns

3. **Continuous:**
   - Numeric columns with many distinct values
   - Float/double columns

4. **Text:**
   - String columns with average word count ≥ 1.5 words per value
   - String columns with average length > 100 characters

## Error Handling

The tool includes comprehensive error handling:
- Each stage of processing is wrapped in try/except blocks
- Errors are logged with detailed messages
- The program continues processing when possible, with fallback strategies
- Debug mode provides verbose logging for troubleshooting

## Performance Considerations

For large datasets:
- Use the sample-size parameter to analyze a subset of data
- The tool limits frequency tables to top 100 values to prevent memory issues
- Consider running on a more powerful Spark cluster for very large files


## Extending the Code

To extend the functionality:

### Adding a New Column Type

1. Add a new constant (e.g., `DATE = "date"`)
2. Update `detect_column_types()` to detect the new type
3. Create a new analysis function (e.g., `analyze_date()`)
4. Update `analyze_column()` to call your new function

### Adding New Metrics

1. Modify the relevant analysis function
2. Add your new metrics to the result dictionary

### Supporting New Input Formats

1. Create a new reading function (e.g., `read_parquet()`)
2. Update the main function to select the appropriate reader based on file extension

### Improving Performance

1. Add caching for intermediate results
2. Use more efficient Spark operations
3. Implement more sophisticated sampling strategies

### Batch Processing

For analyzing multiple files:

```bash
for file in *.csv; do
  python data_explorer.py -i "$file" -o "${file%.csv}_report.json"
done
```

## Implementation Notes

1. The tool uses dynamic type inference when reading CSVs to detect proper data types
2. Type detection uses sampling for better performance with large datasets
3. Frequency tables are limited to prevent memory issues with high-cardinality columns
4. Word frequency analysis is case-insensitive