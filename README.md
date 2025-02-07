# Pipeline Impact Indicator: A geospatial and machine learning-based tool for assessing environmental impacts on mangrove ecosystems in the Niger Delta.

Quantifying the Impact of Crude Oil Spills on the Mangrove Ecosystem in the Niger Delta Using AI and Earth Observation <br>
Remote Sens. 2025, 17(3), 358; https://doi.org/10.3390/rs17030358

<br>

### Dependencies
Please see the required_pkgs.txt file for required python libraries to run this end-to-end. Additionally for export_ee.ipynb you'll need to make a copy of the following for Sentinel-1 processing - https://github.com/adugnag/gee_s1_ard 

### Usage Instructions
1) First run *export_ee.ipynb* to download processed Sentinel-1 composites for your ROI onto your Drive and then download the resulting tifs onto your machine
2) Run through *data_processing* to create production-ready NetCDF stacks from your tifs
3) Finally *model_run* to run the model on your data. This also handles scaling and shaping with the model_run.py script. This notebook also includes setup and running of the Hidden Markov Model. 
