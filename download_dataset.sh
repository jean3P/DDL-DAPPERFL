#!/usr/bin/env bash
#
# download_office_caltech.sh
#
# 1) Creates a "resources" directory if it doesn't exist
# 2) Downloads OfficeCaltechDomainAdaptation-master.tar.gz from Google Drive
# 3) Extracts it into ./resources
# 4) Copies the images to a new folder named Office_Caltech_10
# 5) Cleans up.

set -e  # Exit immediately on error

# 1) Create resources folder if not present
mkdir -p ./resources

# 2) Download the tar.gz
echo "Downloading OfficeCaltechDomainAdaptation-master.tar.gz..."
gdown --id 1zF-ZR-_lC67YvuGKGXcNG-OyqOcOKhYn \
      -O OfficeCaltechDomainAdaptation-master.tar.gz

# 3) Extract the archive into ./resources
echo "Extracting to ./resources..."
tar -xzf OfficeCaltechDomainAdaptation-master.tar.gz -C ./resources

# Remove macOS resource-fork files if present
find ./resources/OfficeCaltechDomainAdaptation-master/images/ \
     -type f -name '._*' -delete

# 4) Create the Office_Caltech_10 folder
mkdir -p ./resources/Office_Caltech_10

# 5) Copy only the contents of `images/` there
cp -r ./resources/OfficeCaltechDomainAdaptation-master/images/* \
      ./resources/Office_Caltech_10

# Clean up
echo "Cleaning up..."
rm OfficeCaltechDomainAdaptation-master.tar.gz
rm -rf ./resources/OfficeCaltechDomainAdaptation-master

echo "Done! The images are now in ./resources/Office_Caltech_10"
