# A Brain-Inspired Model for Multi-Step Forecasting of Malignant Arrhythmias

# Abstract
Malignant arrhythmias (MA), stemming from abnormalities in the neuronal signaling of the cardiac muscle, necessitate sophisticated predictive models for effective clinical management. Traditional machine learning models primarily rely on single-step forecasting and fail to capture the complex temporal dynamics of underlying arrhythmogenic processes. This paper introduces the first multi-step forecasting framework for MA, leveraging a brain-inspired approach that emulates and captures the neuronal signal transmission patterns embedded in electrocardiogram (ECG) data. Our framework comprises three key modules: (i) input module, (ii) multi-path propagation module, and (iii) multi-step forecasting module. The multi-path propagation module incorporates short-term and long-term paths that reflect the different time scales of neural information processing. We further introduce novel brain-inspired information processing units within this module. First, the local and global synaptic plasticity units extract the local and global temporal patterns in the ECG using temporal convolution blocks and cosine-similarity based pattern matching. The processed information is transmitted to the subsequent unit, as well as the Hebb-based learning unit, designed to model the neuromodulation of spike- and feature-level activations and connection strength of the pre- and post-synaptic neurons. Evaluated on two benchmark datasets, our model outperforms existing state-of-the-art models and baseline multi-step models in both short-term and long-term forecasting tasks. The results not only demonstrate the potential of our model in providing a robust clinical tool for fine-grained arrhythmia intervention but also offer valuable insights for advancing multi-step forecasting in other applications.

# Requirements
keras==2.6.0 \
numpy==1.19.5 \
numpydoc==1.1.0 \
pandas==1.1.5 \
scikit-learn==0.24.1 \
scipy==1.5.4 \
sklearn==0.0.post1 \
tensorflow==2.6.2 \
tqdm==4.59.0 \
transformers==4.5.1 \
wfdb==3.3.0


# Dataset
1. MIT-BIH Malignant Ventricular Ectopy Database (MADB): https://physionet.org/content/vfdb/1.0.0/
2. Sudden Cardiac Death Holter Database (SCHDB): https://physionet.org/content/sddb/1.0.0/

To process dataset as mentioned above, run this command:
- readpath: original dataset path
- outpath: save path of processed dataset
  
python preprocessing_SCHDB.py --path "<readpath>" --outpath "<outpath>" \
python preprocessing_MADB.py --path "<readpath>" --outpath "<outpath>"


# Training and testing
To train and test the brain-inspiredMSMA in the paper, run this command:
