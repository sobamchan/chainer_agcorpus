from sobamchan.sobamchan_agcorpus import train
from model import MLP, CNN, FixedEmbedCNN

def main():
    train(CNN)

if __name__ == '__main__':
    main()
