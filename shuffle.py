import csv
import random

def clean_csv(input_file, output_file):
    # Open the input CSV file for reading
    with open(input_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        # Read all rows except the header
        rows = list(reader)[1:]
        
        # Shuffle the rows
        random.shuffle(rows)
        
        # Open the output CSV file for writing
        with open(output_file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            # Write the header row to the output file
            writer.writerow(next(csv.reader(open(input_file))))
            # Write the shuffled rows to the output file
            writer.writerows(rows)

# Input and output file paths
input_file = 'csv/ems_data.csv'
output_file = 'csv/ems_data_randomized.csv'

# Call the function to clean the CSV file
clean_csv(input_file, output_file)

print("CSV file cleaned successfully!")
