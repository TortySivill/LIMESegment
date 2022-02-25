# LIMESegment
Code to fully reproduce benchmark results (and to extend for your own purposes) from the paper: <b>LIMESegment: Meaningful, Realistic Time Series Explanations</b>

## Goal
The goal of this package is to provide a modular way of adapting LIME to Time Series data. It provides methods for:
<ul>
  <li> Segmenting a time series </li>
  <li> Perturbing a time series </li>
  <li> Measuring similarity between time series </li>
  <li> Putting these all together to find explanations in the form 'This segment was most import for the overall classification'
  </ul>

## Installation

<ul> 
  <li> Clone repository </li>
  <li> Install dependencies 'pip3 install -r requirements.txt' </li>

## Content
<ul>
  <li> Utils: 
    <ul> 
      <li> data.py : Contains code generating all synthetic datasets used for experiments & loading UCR datasets used for expleriments </li>
      <li> models.py : Contains code generating and training all classification models used to test explanations:
        <ul> 
          <li> KNN: K Nearest Neighbour Algorithm implemented with Scikit learn </li>
          <li> CNN:1D Convolutional Neural Network implemented with keras </li>
          <li> LSTMFCN: State of the Art Hybrid LSTM and and LSTMFCN implemented by Karim et al. : https://github.com/titu1994/LSTM-FCN </li>
        </ul>
      </li> 
      <li> explanations.py: Contains code generating explanations: 
        <ul>
          <li> LIMEsegment: Our proposed adaptation of LIME to time series data  </li>
          <li> Neves: The proposed adapatation of LIME to time series data of Neves et al.: https://boa.unimib.it/retrieve/handle/10281/324847/492202/Manuscript.pdf </li>
          <li> Leftist: The proposed adaptation of LIME to time series data of Guilleme et al.: https://ieeexplore.ieee.org/abstract/document/8995349
        </ul>
        <li> Contains code for NNSegement, our proposed segmentation algorithm and is a building block of LIMESegment </li>
      <li>  perturbations.py: Contains code for the perturbation strategies evaluated:
        <ul>
          <li> RBP: Realistic Background Peturbation, a frequency based perturbation strategy proposed in the paper and is a building block of LIMESegment </li>
          <li> Zero, random noise and Gaussian blur perturbations </li> 
        </ul>
      </li>
      <li> metrics.py: Contains code for the Robustness and Faithfulness measures used to evaluate each explanation module </li>
      <li> constants.py: Contains details for loading and processing the UCR Time Series datasets used in experiments </li> 
    </ul>
  </li>
  <li> Experiments: Jupyter notebooks containing reproducable implementations of our paper experiments:
    <ul>
      <li>Segmentation: Evaluate NNSegment against state-of-the-art segmentation approaches</li>
      <li> Background Perturb: Evaluate RBP against other peturbation strategies </li>
      <li> Locality: Evaluate the use of DTW in LIMESegment against the Euclidean distance measure </li>
    </ul> 
      <li>RobustnessFaithfulness.py: Evalue LIMESegment against Neves and Leftist on overall explanations </li>
  </li>
  <li> Data: Contains UCR datasets used in experiments. Note that the apnea dataset used in the Segmentation Tests is not included in this repo but can be downloaded from: https://www.physionet.org/content/apnea-ecg/1.0.0/) </li>
</ul>


## Running LIMESegment
Run:
```python
  from Utils.explanations import LIMESegment
  explanations = LIMESegment(ts, model, model_type, distance, window_size, cp, f)
```
Returns segment importance vector as returned by LIMESegment
LIMESegment takes as input:
<ul> 
  <li> ts: TS array of shape T x 1 where T is length of time series </li>
  <li> model: Trained model on dataset array of shape n x T x 1 </li>
  <li> model_type: String indicating if classificaton model produces binary output "class" or probability output "proba", default     "class" </li>
  <li> distance: Distance metric to be used by LIMESegment default is 'dtw' </li>
  <li> window_size: Window size to be used by NNSegment default is T/5 </li>
  <li> cp: Number of change points to be determinded by NNSegment default is 3 </li>
  <li> f: Frequency parameter for RBP default is T/10 </li>
 </ul> 
