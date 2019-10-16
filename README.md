This program uses a convolutional neural network to learn the
sequence-structure relationship for cyclic hexapeptides. This program 
has multiple different functions:

Grid searching for hyperparameters, finding the cross-validation accuracy for a
given set of hyperparameters, and finding well structured sequences. 

To switch between these different functions, the only current way to do so is
going into main.py and changing which functions are called. This main file can
then be run with simply:

python main.py

This main.py file will use a variety of other files for helper functions. These
other files have descriptions in their headers. This includes parsing raw data,
finding cyclic sequence equivalents, processing energy information,
creating a neural network in tensorflow, and searching for hyperparameters.
