from sobamchan.sobamchan_agcorpus import train
from model import MLP, CNN, FixedEmbedCNN, ResCNN, RNN

def main():
    train(RNN)

if __name__ == '__main__':
    main()
