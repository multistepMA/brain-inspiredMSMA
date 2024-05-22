# A Brain-Inspired Model for Multi-Step Forecasting of Malignant Arrhythmias
This repository is the official implementation of "A Brain-Inspired Model for Multi-Step Forecasting of Malignant Arrhythmias, Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024)".

# Overview
Malignant arrhythmias (MA), stemming from abnormalities in the neuronal signaling of the cardiac muscle, necessitate sophisticated predictive models for effective clinical management. Traditional machine learning models primarily rely on single-step forecasting and fail to capture the complex temporal dynamics of underlying arrhythmogenic processes. 

<p align="center"><img src=https://github.com/multistepMA/brain-inspiredMSMA/assets/170433512/5f21acbd-865e-47d7-880d-28dbc218fb8a width="70%" height="50%"></p>

We introduces the first multi-step forecasting framework for MA, leveraging a brain-inspired approach that emulates and captures the neuronal signal transmission patterns embedded in electrocardiogram (ECG) data. Our framework comprises three key modules: (i) input module, (ii) multi-path propagation module, and (iii) multi-step forecasting module. The multi-path propagation module incorporates short-term and long-term paths that reflect the different time scales of neural information processing. 


<p align="center"><img src = https://github.com/multistepMA/brain-inspiredMSMA/assets/170433512/457fbdb5-3382-40e5-b207-a3d8bd2ea014 width="70%" height="50%"></p>

<p align="center"><img src = https://github.com/multistepMA/brain-inspiredMSMA/assets/170433512/28b06a5d-1e9d-4cf7-ab62-c6d646c77913 width="70%" height="50%"></p>




# Requirements
To install requirements:

```bash
pip install -r requirements.txt
```


# Dataset
1. MIT-BIH Malignant Ventricular Ectopy Database (MADB): https://physionet.org/content/vfdb/1.0.0/
2. Sudden Cardiac Death Holter Database (SCHDB): https://physionet.org/content/sddb/1.0.0/

To process dataset as mentioned above, run this command:
For MADB:
```bash
python preprocessing_MADB.py --path <path_to_data> --outpath <path_to_processed_data>
```

For SCHDB:
```bash
python preprocessing_SCHDB.py --path <path_to_data> --outpath <path_to_processed_data>
```

- path_to_data: original dataset path
- path_to_processed_data: save path of processed dataset
 

# Training
To train the model in the paper, run this command:
```bash
python train.py --path <path_to_data> --model_save_dir <drectory_saved_model> --outpath <path_to_processed_data> --model_name <model_name_saved>
```

# Evaluation
To evaluate the model in the paper, run this command:
```bash
python evaluate.py --path <path_to_data> --model_save_dir <drectory_saved_model> --outpath <path_to_processed_data> --model_name <model_name_saved>
```

# Results
Evaluated on two benchmark datasets, our model outperforms existing state-of-the-art models and baseline multi-step models in both short-term and long-term forecasting tasks. The results not only demonstrate the potential of our model in providing a robust clinical tool for fine-grained arrhythmia intervention but also offer valuable insights for advancing multi-step forecasting in other applications.
<p align="center"><img src = https://github.com/multistepMA/brain-inspiredMSMA/assets/170433512/b447c626-8e00-4713-8c1a-7742ed934312  width="70%" height="50%"></p>

<p align="center"><img src = https://github.com/multistepMA/brain-inspiredMSMA/assets/170433512/f1358678-3c57-476c-bff4-d73ed108c999  width="70%" height="50%"></p>

