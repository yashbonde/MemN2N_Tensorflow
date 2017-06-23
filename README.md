# MemN2N_Tensorflow
This is a repository that has my work on Memory Networks using various libraries.

The meomory networks are based on https://arxiv.org/abs/1410.3916

Though they are modified to be used in a more practical sense (i.e. weakly supervised, back-prop enabled), known as End-to-end memory networks, as shown in https://arxiv.org/abs/1503.08895

Check the individal files for more details.

#### Babi baseline model

Adding the baseline model for babi tasks.

## First Model

This is the first model and is used to run the bAbI tasks.

To run the first model use the files memn2n_simple.py, data_utils.py and model.py

##### model.py

This file has the End-to-End Memory Network. I will add further details of the model soon.

##### data_utils.py

This file has the necessary functions to load any bAbI task.

##### memn2n_simple.py

To run the model, execute in terminal python3 memn2n_simple.py . For further help run the command python3 memn2n_simple.py --help .
