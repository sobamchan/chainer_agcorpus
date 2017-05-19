from sobamchan.sobamchan_agcorpus import train
from model import MLP, CNN, FixedEmbedCNN, ResCNN

def main():
    train(ResCNN)

if __name__ == '__main__':
    main()
