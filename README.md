# Light-SERNet

This is the Tensorflow 2.x implementation of our paper ["Light-SERNet: A lightweight fully convolutional neural network for speech emotion recognition"](https://arxiv.org/abs/2110.03435), submitted in ICASSP 2022. 

<div align=center>
<img width=95% src="https://github.com/AryaAftab/LIGHT-SERNET/blob/master/pics/Architecture.png"/>
</div>
In this paper, we propose an efficient and lightweight fully convolutional neural network(FCNN) for speech emotion recognition in systems with limited hardware resources. In the proposed FCNN model, various feature maps are extracted via three parallel paths with different filter sizes. This helps deep convolution blocks to extract high-level features, while ensuring sufficient separability. The extracted features are used to classify the emotion of the input speech segment. While our model has a smaller size than that of the state-of-the-art models, it achieves a higher performance on the IEMOCAP and EMO-DB datasets.




## Run
### 1. Clone Repository
```bash
$ git clone https://github.com/AryaAftab/LIGHT-SERNET.git
$ cd LIGHT-SERNET/
```
### 2. Requirements
- Tensorflow >= 2.3.0
- Numpy >= 1.19.2
- Tqdm >= 4.50.2
- Matplotlib> = 3.3.1
- Scikit-learn >= 0.23.2

```bash
$ pip install -r requirements.txt
```

### 3. Data:
* Download **[EMO-DB](http://emodb.bilderbar.info/download/download.zip)** and **[IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm)**(requires permission to access) datasets
* extract them in [data](https://github.com/AryaAftab/LIGHT-SERNET/tree/master/data) folder
### 4. Prepare datasets :
Use the following code to convert each dataset to the desired size(second):
```bash
$ python utils/segment/segment_dataset.py -dp data/{dataset_folder} -ip utils/DATASET_INFO.json -d {datasetname_in_jsonfile} -l {desired_size(seconds)}
```
For example, for EMO-DB Dataset :
```bash
$ python utils/segment/segment_dataset.py -dp data/EMO-DB -ip utils/DATASET_INFO.json -d EMO-DB -l 3
```
### 5. Set hyperparameters and training config :
You only need to change the constants in the [hyperparameters.py](https://github.com/AryaAftab/LIGHT-SERNET/blob/master/hyperparameters.py) to set the hyperparameters and the training config.

### 6. Strat training:
Use the following code to train the model on the desired dataset with the desired cost function.
- Note 1: The database name is the name of the database folder after segmentation.
- Note 2: The results for the confusion matrix are saved in the [result](https://github.com/AryaAftab/LIGHT-SERNET/tree/master/result) folder.
```bash
$ python train.py -dn {dataset_name_after_segmentation} -ln {cost_function_name}
```
For example, for EMO-DB Dataset :
```bash
$ python train.py -dn EMO-DB_3s_Segmented -ln focal
```

## Citation

If you find our code useful for your research, please consider citing:
```bibtex
@article{aftab2021light,
  title={Light-SERNet: A lightweight fully convolutional neural network for speech emotion recognition},
  author={Aftab, Arya and Morsali, Alireza and Ghaemmaghami, Shahrokh and Champagne, Benoit},
  journal={arXiv preprint arXiv:2110.03435},
  year={2021}
}
```