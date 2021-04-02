# COPE
This is chessbot written in almost-pure Python. The choice of language makes it quite slow at calculating, but it does 
mean that writing it was a COmfy Python Experience, hence the name.

### Install and run
Just clone the repo and install the package as usual, by going to the repo directory and executing

`python -m pip install .`

Then execute the package to play against the AI using

`python -m chessbot`

### TODO

- Experiment with using an NN heuristic evaluation function
- Try out tree search performance optimisations such as transposition tables and killer heuristic
- Try doing parameter optimisation via self-play
