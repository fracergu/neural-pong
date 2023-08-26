# Pong game with Neural Network training

## Installation

1. Clone the repository

    ```bash
    git clone https://github.com/fracergu/neural-pong.git
    ```

2. Install the requirements

    ```bash
    cd neural-pong
    pip install -r requirements.txt
    ```

## Usage

There two modes of operation: training and playing.

### Training

To train the neural network, run the following command:

```bash
python game.py --mode train
```

The training will run until the user stops it. The neural network will be saved in the `pong_model.h5` file. About 100 rounds of training are required to get a decent result.

### Playing

To play the game, run the following command:

```bash
python game.py --mode play
```

The game will start and the user will be able to play with W and S keys to move the paddle. The neural network will be loaded from the `pong_model.h5` file.
