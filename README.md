# Gait_DL_Diagnostic_Tool
A diagnostic tool that combines clinical gait analysis and deep learning - Paper: "Deep learning for clinical kinematic gait classification - a diagnostic tool"

This directory shows the code of the clinical gait-based diagnostic model used in paper "Deep learning for clinical kinematic gait classification - a diagnostic tool" and demonstrates its training and prediction process. [*Training_with_lrR_ET-TDvsCP.ipynb*](https://github.com/LANZhengyang/Gait_DL_Diagnostic_Tool/blob/main/Training_with_lrR_ET-TDvsCP.ipynb) shows a demo example of training ResNet using the dataset TDvsCP. To train a different model, you can modify the Net_name (options: ResNet, LSTM, InceptionTime). To read different datasets, you can change the dir_dataset in the load_dataset_v1 function to specify the database location and the d_file_list to specify the dataset file.

The implementation is base on Python 3.9.11. The dependencies can be installed by 'pip3 install -r requirements.txt'.


