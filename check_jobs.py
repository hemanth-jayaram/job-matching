import csv
import os
from collections import defaultdict

def analyze_jobs_file(filepath, sample_size=5):
    """Analyze the job postings CSV file and print its structure and sample data."""
    print(f"\nAnalyzing job postings file: {filepath}")
    print("-" * 80)
    
    # Check if file exists and is accessible
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return
    
    # Get file size
    file_size = os.path.getsize(filepath) / (1024 * 1024)  # in MB
    print(f"File size: {file_size:.2f} MB")
    
    # Try to read the file with different encodings
    encodings = ['utf-8', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                # Try to read the first line to check if it's a header
                first_line = f.readline().strip()
                if not first_line:
                    print(f"File appears to be empty")
                    return
                    
                # Try to parse the first line as CSV to get column names
                try:
                    reader = csv.reader([first_line])
                    headers = next(reader)
                    print(f"\nDetected {len(headers)} columns with encoding '{encoding}':")
                    for i, header in enumerate(headers, 1):
                        print(f"  {i}. {header}")
                    
                    # Read a few more lines to get sample data
                    print("\nSample data (first few rows):")
                    print("-" * 80)
                    
                    # Reset file pointer and create a new reader
                    f.seek(0)
                    reader = csv.DictReader(f)
                    
                    # Print sample rows
                    for i, row in enumerate(reader):
                        if i >= sample_size:
                            break
                        print(f"Row {i+1}:")
                        for key, value in row.items():
                            value_preview = str(value)[:100] + ('...' if len(str(value)) > 100 else '')
                            print(f"  {key}: {value_preview}")
                        print("-" * 40)
                    
                    # Count total rows
                    f.seek(0)
                    total_rows = sum(1 for _ in csv.reader(f)) - 1  # subtract 1 for header
                    print(f"\nTotal rows (estimated): {total_rows:,}")
                    
                    # Successfully read the file with this encoding
                    return
                    
                except Exception as e:
                    print(f"Error parsing as CSV with encoding '{encoding}': {str(e)}")
                    print(f"First 200 characters of first line: {first_line[:200]}")
                    continue
                    
        except UnicodeDecodeError:
            print(f"Failed to read with encoding '{encoding}'") 
            continue
        except Exception as e:
            print(f"Unexpected error with encoding '{encoding}': {str(e)}")
            continue
    
    print("\nFailed to read the file with any of the attempted encodings.")
    print("The file may be corrupted, in a different format, or use an unexpected encoding.")

if __name__ == "__main__":
    import sys
    
    # Default file path
    default_file = "job_market_data.csv"
    
    # Use command line argument if provided, otherwise use default
    file_to_analyze = sys.argv[1] if len(sys.argv) > 1 else default_file
    
    print(f"Job Market Data Analyzer")
    print("=" * 40)
    
    analyze_jobs_file(file_to_analyze)
    
    print("\nAnalysis complete. Check the output above for details about the job postings file.")
