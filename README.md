# Deep-Learning-for-Audio-Super-Resolution
This is my master's degree thesis project in Data Science.

###### Abstract 
> Audio Super-Resolution is the problem of predicting the missing high-frequency content of a given signal from its low frequencies. Several recent studies have shown that Deep Learning algorithms are able to achieve remarkable results by modeling audio Super-Resolution as a regression task. A large variety of approaches have been proposed in literature, including convolutional and recurrent architectures to capture both local and long-term dependencies between audio frames. Furthermore, some research show that significant improvements may be achieved by processing the input signal not only in the time, but also in the frequency domain by exploiting the Fourier transform operations as an integral part of the neural network configuration. This thesis project aims not only to deal with the study of these approaches, but also to combine them in a principled way in order to explore a novel model architecture.


### Introduction
The key thrust of this thesis is on the implementation of a novel model architecture inspired by some of the state-of-the-art techniques. The two studies from which most of the proposed methods in this work
derives are the following:
- Sawyer Birnbaum et al. "Temporal FiLM: Capturing Long-Range Sequence Dependencies with Feature-Wise Modulations". In: *Advances in Neural Information Processing Systems. 2019, pp. 10287-10298*. ([github repo](https://github.com/kuleshov/audio-super-res))
- Teck Yian Lim et al. "Time-frequency networks for audio super-resolution". In: *2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE. 2018, pp. 646-650*. ([github repo](https://github.com/moodoki/tfnet))

### Requirements & Setup
The main packages used in this work include Tensorflow (1.x), Scipy (1.2.1) and Librosa (0.6.0). We use Google Colaboratory (Colab) to train the models, so the scripts are organized into notebooks.

As for the data, we use the Voice Cloning Toolkit Corpus (VCTK) [dataset](https://datashare.ed.ac.uk/handle/10283/3443). 

Furthermore, we process audio data with one of the state-of-the-art open-source STT engines, i.e. Deep Speech (([github repo](https://github.com/mozilla/DeepSpeech))). 

### Documents
You can read my [thesis](docs/thesis_latex/Bini_Giorgio_Tesi_LMDS_24032021.pdf) and look at my [presentation](docs/Presentation/Presentazione.pptx).

![plot](docs/Presentation/funny_img.png)
