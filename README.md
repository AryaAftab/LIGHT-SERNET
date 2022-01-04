# Light-SERNet

This is the Tensorflow 2.x implementation of our paper ["Light-SERNet: A lightweight fully convolutional neural network for speech emotion recognition"](https://arxiv.org/abs/2110.03435), submitted in ICASSP 2022. 

<div align=center>
<img width=95% src="./pics/Architecture.png"/>
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
* extract them in [data](./data) folder

### 4. Set hyperparameters and training config :
You only need to change the constants in the [hyperparameters.py](./hyperparameters.py) to set the hyperparameters and the training config.

### 6. Strat training:
Use the following code to train the model on the desired dataset, cost function, and input length(second).
- Note 1: The input is automatically cut or padded to the desired size and stored in the [data](./data) folder.
- Note 2: The best model are saved in the [result](./result) folder.
- Note 3: The results for the confusion matrix are saved in the [result](./result) folder.
```bash
$ python train.py -dn {dataset_name} \
                  -id {input durations} \
                  -at {audio_type} \
                  -ln {cost function name} \
                  -v {verbose for training bar} \
                  -it {type of input(mfcc, spectrogram, mel_spectrogram)}
```
#### Example:

EMO-DB Dataset:
```bash
python train.py -dn "EMO-DB" \
                -id 3 \
                -at "all" \
                -ln "focal" \
                -v 1 \
                -it "mfcc"
```

IEMOCAP Dataset:
```bash
python train.py -dn "IEMOCAP" \
                -id 7 \
                -at "impro" \
                -ln "cross_entropy" \
                -v 1 \
                -it "mfcc"
```
**Note : For all experiments just run ```run.sh```**
```bash
sh run.sh
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
