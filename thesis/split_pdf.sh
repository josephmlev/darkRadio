#!/bin/bash

# Prompt for input PDF file
read -p "Enter the input file name (without .pdf if omitted): " input_file

# Add .pdf extension if not provided
if [[ $input_file != *.pdf ]]; then
    input_file="${input_file}.pdf"
fi

# Check if the input file exists
if [[ ! -f $input_file ]]; then
    echo "Error: File '$input_file' not found."
    exit 1
fi

# Prompt for output file name
read -p "Enter the output file name (without .pdf if omitted): " output_file

# Add .pdf extension if not provided
if [[ $output_file != *.pdf ]]; then
    output_file="${output_file}.pdf"
fi

# Prompt for start page
read -p "Enter the start page: " start_page

# Prompt for stop page
read -p "Enter the stop page: " stop_page

# Run pdftk with the specified parameters
pdftk "$input_file" cat "$start_page"-"$stop_page" output "$output_file"

# Confirmation message
if [[ $? -eq 0 ]]; then
    echo "Successfully created '$output_file' containing pages $start_page to $stop_page from '$input_file'."
else
    echo "Error: Failed to create the PDF file."
fi

