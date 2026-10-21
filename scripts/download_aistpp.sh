
mkdir -p raw_data/aistpp

# please check your network connection to Google
gdown 1RzqSbSnbMEwLUagV0GThfpm9JJXePGkV -O raw_data/edge_aistpp.zip
unzip raw_data/aistpp/edge_aistpp.zip -d raw_data/aistpp/
rm raw_data/edge_aistpp.zip