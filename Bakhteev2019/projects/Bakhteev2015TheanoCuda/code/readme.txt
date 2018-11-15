Usage:
To preprocess data to experiment format from data you can use generate_data.py

Run algorithm syntax:
python launch.py WISDM_filename RBM_layer_size AE_layer_size Softmax_layer_size RBM_epochs AE_epochs Softmax_epochs part_of_data_for_training number_of_iterations 

Note, that test dataset always contains 25% of the original dataset
example:
python launch.py ../data/WISDM.npy 378 225 117 500 500 3000 0.75 1

The final results of expertiment (mean and std of test and train errors) are showed in stdout. In order to write result in file you can redirect stdout, for example:
python -u launch.py ../data/WISDM.npy 378 225 117 500 500 3000 0.75 1 >results.txt

Sample of run-script for CPU and GPU launces are:
run8CPU.sh (for 8 cores)
runGPU.sh
