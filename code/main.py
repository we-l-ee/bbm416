from os import path
from models import *

import argparse
import gc

import sys
import matplotlib.pyplot as plt

# Volatile variables to be used in inference only. They get rid off other variables that is used in back propagation.
#
gc.enable()


def load(model_path, output_path, lname, sname, cuda, loss):
    print("Model loading...")

    loc = os.path.join(model_path, lname)
    with open(loc + ".info", 'rb') as f:
        _args  = f.readline().strip().split()
        type_, args = _args[0], _args[1:]
    func = {b"VGGModel": VGGModel.load}
    return func[type_](loc, model_path, output_path, sname, cuda, loss, args = args)


def init(type_, model_path, output_path, name, cuda, loss, batch_norm):
    print("Model initialization...")
    func = {"vgg16": VGGModel}
    return func[type_](model_path, output_path, name, cuda=cuda, loss=loss, batch_norm=batch_norm)


def plot_all(output_path, figure_folder, figname):
    outputs = np.load(output_path)
    train_outs = dict()
    test_outs = dict()
    for out in outputs:
        if out['type'] == 'train':
            try:
                train_outs['error1'].extend([err[0] for err in out['error']])
            except KeyError:
                train_outs['error1'] = [err[0] for err in out['error']]

            try:
                train_outs['error5'].extend([err[1] for err in out['error']])
            except KeyError:
                train_outs['error5'] = [err[1] for err in out['error']]

            try:
                train_outs['loss'].extend([l for loss in out['loss'] for l in loss])
            except KeyError:
                train_outs['loss'] = [l for loss in out['loss'] for l in loss]

            try:
                train_outs['config'].append(out['config'])
            except KeyError:
                train_outs['config'] = [out['config']]

        elif out['type'] == 'test':
            try:
                test_outs['error1'].append(out['error'][0])
            except KeyError:
                test_outs['error1'] = [out['error'][0]]

            try:
                test_outs['error5'].append(out['error'][1])
            except KeyError:
                test_outs['error5'] = [out['error'][1]]

    for k in train_outs:
        train_outs[k] = np.array(train_outs[k])
    for k in test_outs:
        test_outs[k] = np.array(test_outs[k])

    fig1 = plt.figure(1)
    plt.title("Train top 1 error percentages after each epoch")
    plt.plot(range(len(train_outs['error1'])), train_outs['error1'])
    plt.savefig(os.path.join(figure_folder, figname) + "-train-err1.png", format='png')

    fig2 = plt.figure(2)
    plt.title("Train top 5 error percentages after each epoch")
    plt.plot(range(len(train_outs['error5'])), train_outs['error5'])
    plt.savefig(os.path.join(figure_folder, figname) + "-train-err5.png", format='png')

    fig3 = plt.figure(3)
    plt.title("Test top 1 error percentages after each tranning")
    plt.plot(range(len(test_outs['error1'])), test_outs['error1'])
    plt.savefig(os.path.join(figure_folder, figname) + "-test-err1.png", format='png')

    fig4 = plt.figure(4)
    plt.title("Test top 5 error percentages after each tranning")
    plt.plot(range(len(test_outs['error5'])), test_outs['error5'])
    plt.savefig(os.path.join(figure_folder, figname) + "-test-err5.png", format='png')

    fig5 = plt.figure(5, figsize=(5, 3), dpi=500)
    plt.title("Loss of the training after each iteration")
    plt.scatter(y=train_outs['loss'], x=range(len(train_outs['loss'])), s=0.5)
    plt.savefig(os.path.join(figure_folder, figname) + "-train-loss.png", format='png')
    print("Plots are written under to './" + figure_folder + "/'!")


def main():
    parser = argparse.ArgumentParser(description='DNN Parser of BBM418 Assignment 3.')
    parser.add_argument("-ftrain", default='train.txt',
                        help="Text file which has the paths of all the train images."
                             " Default is 'train.txt'."
                        )
    parser.add_argument("-ftest", default='test.txt',
                        help="Text file which has the paths of all the test images."
                             " Default is 'test.txt'."
                        )
    parser.add_argument("-model_path", default='models',
                        help="Root folder of the stored models."
                             " Default is 'models'."
                        )
    parser.add_argument("-output_path", default='outputs',
                        help="Root folder of the any outputs of the models."
                             " Default is 'outputs'."
                        )
    parser.add_argument("-figure_path", default='figures',
                        help="Root folder of the any outputs of the plot figures."
                             " Default is 'figures'."
                        )

    parser.add_argument("-batch", type=int, default=16,
                        help="Batch size of the train data")
    parser.add_argument("-lr", type=float, default=0.01,
                        help="Learning rate of the training optimizer")
    parser.add_argument("-momentum", type=float, default=0.9,
                        help="Momentum value of the SGD optimizer")

    parser.add_argument("-save", action='store_true',
                        help="Save after the training. Model is named incrementally if there is already loaded model."
                             " Default is 'false', disabled."
                        )
    parser.add_argument("-load", default=None,
                        help="Load model before the training. This will be name for the loading."
                             " Default is 'None', disabled."
                        )
    parser.add_argument("-mname", default="model-1.0",
                        help="Model`s name that will be used in saving action."
                             " Default is 'model-1.0'.")

    parser.add_argument("-plot", action='store_true',
                        help="Plots the loss output by given -mname which is used with -output_path to determine which "
                             "loss output should be plotted.")

    parser.add_argument("-train", type=int, nargs='+', default=[0],
                        help="For every given integer it will run as that much epochs. After each run of epocs "
                             "if test is activated test will be applied to model. Default is single run of 50 epochs."
                             " Default is '0' which means disabled."
                        )

    parser.add_argument("-test", type=int, default=0,
                        help="For given value, test will be applied after that epochs. For example if the value is 1, "
                             "it will apply test after 1 epoch. Default is 0 which means disabled.")

    parser.add_argument("-freeze", action='store_true',
                        help="Enables freezing -clip param will change the freezed layers."
                             " Default is 'false', disabled.")
    parser.add_argument("-clip", type=int, nargs='+', default=[0, 26],
                        help="Freezes the indicated layers by given to int.[0,26] is the default between 0 and 25th "
                             "layers is freezed." " Default is '0 26'."
                        )

    parser.add_argument("-type", default='vgg16',
                        help="Initialized vgg16 model as default.")

    parser.add_argument("-cuda", action='store_true',
                        help="Use cuda.")

    parser.add_argument("-mkdir", action='store_true',
                        help="Activate it if giving directories are not made before hand.")

    parser.add_argument("-loss", default='mse',
                        help="Loss function. Default is mse.")

    parser.add_argument("-batch_norm", action='store_true',
                        help="Activates batch_normalization. It can only be used when new model is initialized")

    # print(sys.argv)
    args = parser.parse_args(sys.argv[1:])
    print(args)

    if args.mkdir:
        try:
            os.mkdir(args.model_path)
        except:
            pass
        try:
            os.mkdir(args.output_path)
        except:
            pass
        try:
            os.mkdir(args.figure_path)
        except:
            pass

    model = None
    if args.load is not None:
        model = load(args.model_path, args.output_path, args.load, args.mname, args.cuda, args.loss);
    elif (len(args.train) > 0 and args.train[0] > 0) or args.test:
        model = init(args.type, args.model_path, args.output_path, args.mname, args.cuda, args.loss, args.batch_norm);

    if args.freeze:
        model.freeze(args.clip)


    if args.test !=0:
        if args.test > args.train:
            raise Exception("Tests instances can`t be bigger than epochs.")
        _iter = int(args.train/args.test)
        epochs = [args.test for i in range(_iter)]
        if args.train%args.test != 0:
            epochs.append(args.train%args.test)
    else:
        epochs = [args.train]

    if len(args.train) > 0 and args.train[0] > 0:
        model.update_train_dataset(args.ftrain, args.batch)
        model.update_test_dataset(args.ftest, args.batch)

        for i, epoch in enumerate(epochs):
            print("Training of (", i + 1, "/", len(epochs), ") with epoch [", epoch, "] initializing...")
            model.train(epoch=epoch, lr=args.lr, momentum=args.momentum, write=True)
            print("Testing of (", i + 1, "/", len(epochs), ") initializing...")
            if args.test != 0:
                model.test(write=True)

    elif args.test:
        model.update_test_dataset(args.ftest, args.batch)
        model.test(write=True)

    if model is not None:
        model.save_info()
    if args.save:
        model.save()

    if args.plot:
        plot_all(path.join(args.output_path, args.mname) + '.npy', args.figure_path, args.mname)

    return model


# sys.argv.extend("-train 1 1 1 1 1 1 1 1 1 1 -load vgg-mse-10.32 -save -test -lr 0.01 -batch 32 -freeze -clip 0 26 -mname vgg-mse-10.32 -cuda".split())
# sys.argv.extend("-train 10 10 10 -save -test -lr 0.01 -batch 16 -mname vgg-mse-full-3.0 -cuda".split())
# sys.argv.extend("-test -load vgg-full-3.0 -cuda".split())
# sys.argv.extend("-plot -mname vgg-mse-5.0".split())
# sys.argv.extend("-plot -mname vgg-mse-full-2.0".split())

# sys.argv.extend("-train 1 1 -test -lr 0.01 -batch 32 -loss mse -freeze -clip 0 26 -cuda".split())

model = main()