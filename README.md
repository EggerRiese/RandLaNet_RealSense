# RandLaNet_RealSense

### (1) Setup
 
- Clone the repository 
```
git clone
```
- Setup python environment
```
conda create -n randlanet python=3.5
source activate randlanet
pip install -r helper_requirements.txt
sh compile_op.sh
```

### (2) RealSense
RealSense dataset can be found 
<a href="https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1">here</a>. 
Uncompress the folder and move it to 
`/data/RealSense`.

- Preparing the dataset:
```
python utils/data_prepare_RealSense.py
```
- Start training:
```
python main_RealSense.py --gpu 0 --mode train
```
- Start testing:
```
python main_RealSense.py --gpu 0 --mode test
```
- Move all the generated results (*.ply) in `/test` folder to `/data/RealSense/results`, calculate the final mean IoU results (not tested for now):
```
python utils/6_fold_cv.py
```



### Acknowledgment
-  The code refers to <a href="https://github.com/QingyongHu/RandLA-Net">RandLaNet</a>. The Network was customized to fit the Intel RealSense data.
-  Part of their code refers to <a href="https://github.com/jlblancoc/nanoflann">nanoflann</a> library and the the recent work <a href="https://github.com/HuguesTHOMAS/KPConv">KPConv</a>.
