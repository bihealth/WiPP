# wipp_dev


# Installation
git clone https://github.com/bihealth/wipp_dev.git \
cd wipp_dev \
make


# Running of test project

# Change in directory
cd ./projects/example_project
# Call training data generation with n nodes (example: n = 4)
../../run_WiPP.sh tr -n 4 
# Call peak annotation
../../run_WiPP.sh an
# Call peak picking with n nodes (example: n = 4)
../../run_WiPP.sh pp -n 4 


TODO <YG, 2018.09.06> Update
