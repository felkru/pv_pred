import os
import pandas as pd
import io
import sys
from datetime import datetime

def analyze_single_csv(file_path):
    """
    Reads a CSV file, performs the requested analysis (describe, info, head, unique
    of first column), and returns the captured output as a string.
    """
    try:
        # Read the CSV file
        data = pd.read_csv(file_path)

        # Use StringIO to capture the output of print statements and data.info()
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output

        # Perform the analysis and print the results
        print('\ncsv description and info\n')
        print(data.describe())
        
        # data.info() prints directly to stdout, so it's captured by the redirection
        print('\n') # Add a newline before info for better separation
        data.info() 

        print('\n\ncsv head\n')
        print(data.head())

        print('\n\nunique classes\n')
        # Check if the DataFrame is not empty and has at least one column
        if not data.empty and data.shape[1] > 0:
            print(data.iloc[:, 0].unique())
        else:
            print("CSV is empty or has no columns to analyze for unique classes.")

        # Get the captured output
        analysis_output = redirected_output.getvalue()

        # Restore original stdout
        sys.stdout = old_stdout

        return analysis_output

    except pd.errors.EmptyDataError:
        return f"Error: The CSV file '{file_path}' is empty.\n"
    except pd.errors.ParserError as e:
        return f"Error parsing CSV file '{file_path}': {e}\n"
    except Exception as e:
        return f"An unexpected error occurred while processing '{file_path}': {e}\n"

def generate_directory_tree(root_dir):
    """
    Generates a simplified ASCII tree representation of the directory,
    focusing on CSV files.
    """
    tree_str = f"Directory structure for '{root_dir}':\n"
    if not os.path.isdir(root_dir):
        return f"Error: Directory '{root_dir}' not found.\n"

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Calculate current depth for indentation
        depth = dirpath[len(root_dir):].count(os.sep)
        indent = '    ' * depth
        
        # Add current directory to the tree
        if depth == 0:
            # For the root, just list its name
            tree_str += f"└── {os.path.basename(root_dir)}/\n"
        else:
            tree_str += f"{indent}└── {os.path.basename(dirpath)}/\n"
        
        # List CSV files within the current directory
        for f in sorted(filenames):
            if f.endswith('.csv'):
                tree_str += f"{indent}    ├── {f}\n"
        
        # Add placeholders for subdirectories that might contain CSVs
        for d in sorted(dirnames):
            # Only add if it's not a hidden directory and we haven't already processed it
            if not d.startswith('.'): # and there's a CSV deeper? (simplified for now)
                pass # The os.walk will handle the recursive listing naturally


    return tree_str

def main(root_directory="."):
    """
    Recursively finds all CSV files in the given directory and applies the
    analysis function, printing the results in the specified format.
    """
    print("Beginning CSV analysis...\n")

    # Generate and print the directory structure
    # Note: This will only list directories/files that os.walk finds,
    # so it might not be a complete tree if some branches have no CSVs.
    print(generate_directory_tree(root_directory))
    print("\n" + "=" * 50)
    print("CSV File Analysis Reports")
    print("=" * 50 + "\n")

    csv_files_found = []
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.endswith('.csv'):
                full_path = os.path.join(dirpath, filename)
                csv_files_found.append(full_path)

    if not csv_files_found:
        print(f"No CSV files found in '{root_directory}' or its subdirectories.")
        return

    # Process and print analysis for each CSV file
    for csv_file_path in sorted(csv_files_found): # Sort for consistent output order
        print("=" * 50)
        print(f"File: {csv_file_path}")
        print("=" * 50)
        print(analyze_single_csv(csv_file_path))
        print("\n\n") # Add extra newlines for separation between files

    print("\n" + "=" * 50)
    print("CSV Analysis Complete.")
    print("=" * 50)


if __name__ == '__main__':
    # You can specify the root directory to scan.
    # By default, it will scan the current directory where the script is run.
    # Example: To scan a folder named 'my_data_folder' in the same directory as the script:
    # main('my_data_folder')
    main('.') # Scans the current directory
