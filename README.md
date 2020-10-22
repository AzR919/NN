# NN
Neural Net for 10701


Basic feed forward nn implementation With one hidden layer.

Designed to work on the mnist dataset.

To run:

$python main.py

optional arguments:
  -h, --help            show this help message and exit
  --train_root TRAIN_ROOT
                        path to the training data
  --test_root TEST_ROOT
                        path to the test data
  --params_root PARAMS_ROOT
                        path to the initial weights
  --num_epochs NUM_EPOCHS
                        number of epochs to train for
  --step_size STEP_SIZE
                        step size for SGD
  --hidden_dim HIDDEN_DIM
                        dimension for the hidden layer
  --seed SEED           seed for randomization
  --eps EPS             eps to prevent negative log