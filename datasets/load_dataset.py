import os
import torchvision

def build_dataset(args, is_train, trnsfrm=None, training_mode='finetune'):
    if args.data_set == 'MNIST':
        dataset = torchvision.datasets.MNIST(os.path.join(args.data_location, 'MNIST_dataset'), 
                                   train=is_train, transform=trnsfrm, download=True)

        nb_classes = 10        
    return dataset, nb_classes

