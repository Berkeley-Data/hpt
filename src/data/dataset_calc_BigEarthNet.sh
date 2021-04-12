
### S-1 data
declare -a s1_band=("VV" "VH")
path="/home/cjrd/data/bigearthnet/BigEarthNet-S1-v1.0/"
data_index_dir="/home/taeil/hpt_k/src/data"
declare -i numworkers=30
declare -i batchsize=256

for band in ${s1_band[@]}
do
    python dataset_calc_BigEarthNet.py\
     --path $path\
     --data_index_dir $data_index_dir\
     --numworkers $numworkers\
     --batchsize $batchsize\
     --use_s1\
     --band $band

done

### S-2 data
declare -a s2_band=("B01" "B02" "B03" "B04" "B05" "B06" "B07" "B08" "B09" "B11" "B12" "B8A")
path="/home/cjrd/data/bigearthnet/BigEarthNet-v1.0/"
data_index_dir="/home/taeil/hpt_k/src/data"
declare -i numworkers=30
declare -i batchsize=256

for band in ${s2_band[@]}
do
    python dataset_calc_BigEarthNet.py\
     --path $path\
     --data_index_dir $data_index_dir\
     --numworkers $numworkers\
     --batchsize $batchsize\
     --use_s2\
     --band $band

done

