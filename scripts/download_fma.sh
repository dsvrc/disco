#!/bin/bash

set -e

echo "Downloading FMA (Free Music Archive) dataset..."

mkdir -p raw_data/fma
cd raw_data/fma

# FMA dataset URLs from https://github.com/mdeff/fma
# You can choose different subsets based on your needs:

# FMA Small (8 GiB, 8,000 tracks of 30s, 8 balanced genres)
# echo "Downloading FMA Small dataset..."
# wget -c https://os.unil.cloud.switch.ch/fma/fma_small.zip

# Uncomment below if you need larger subsets:

# FMA Medium (25 GiB, 25,000 tracks of 30s, 16 unbalanced genres)
echo "Downloading FMA Medium dataset..."
wget -c https://os.unil.cloud.switch.ch/fma/fma_medium.zip

# FMA Large (93 GiB, 106,574 tracks of 30s, 161 unbalanced genres)
# echo "Downloading FMA Large dataset..."
# wget -c https://os.unil.cloud.switch.ch/fma/fma_large.zip

# FMA Full (879 GiB, 106,574 untrimmed tracks, 161 unbalanced genres)
# echo "Downloading FMA Full dataset..."
# wget -c https://os.unil.cloud.switch.ch/fma/fma_full.zip

# Download metadata
echo "Downloading FMA metadata..."
wget -c https://os.unil.cloud.switch.ch/fma/fma_metadata.zip

echo "Extracting archives..."
unzip -n fma_medium
unzip -n fma_metadata

# Clean up zip files (optional)
# rm fma_small.zip fma_metadata.zip

cd ../..

echo "FMA dataset download completed!"
echo "Dataset location: raw_data/fma/"
echo ""
echo "Dataset structure:"
echo "  fma_small/    - audio files organized by genre"
echo "  fma_metadata/ - tracks.csv, genres.csv, etc."
