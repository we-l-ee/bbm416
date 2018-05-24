from os import path
from models import *

import argparse
import gc

import sys
import matplotlib.pyplot as plt
import numpy as np
# Volatile variables to be used in inference only. They get rid off other variables that is used in back propagation.
#
gc.enable()


def load(root, name):
    print("Model loading...")

    loc = os.path.join(root, name)
    with open(loc + ".info", 'rb') as f:
        _args = f.readline().strip().split()
        type_, args = _args[0], _args[1:]
    func = {b"VGGModel": VGGModel,
            b"ResNetModel":  ResNetModel,
            b"DenseNetModel": DenseNetModel,
            b"GoogLeNetModel": GoogLeNetModel,
            b"AlexNetModel": AlexNetModel,
            b"SqueezeNetModel": SqueezeNetModel}
    model_type = func[type_]
    return model_type.load(args, model_type, loc)


def init(_type, _batch_norm):
    print("Model initialization...")
    func = {"vgg11": VGGModel.init_11,
            "vgg13": VGGModel.init_13,
            "vgg16": VGGModel.init_16,
            "vgg19": VGGModel.init_19,
            "resnet18": ResNetModel.init_18,
            "resnet34": ResNetModel.init_34,
            "resnet50": ResNetModel.init_50,
            "resnet101": ResNetModel.init_101,
            "resnet152": ResNetModel.init_152,
            "densenet121": DenseNetModel.init_121,
            "densenet161": DenseNetModel.init_161,
            "densenet169": DenseNetModel.init_169,
            "densenet201": DenseNetModel.init_201,
            "googlenet": GoogLeNetModel.init_v3,
            "googlenetv3": GoogLeNetModel.init_v3,
            "alexnet": AlexNetModel.init,
            "squeezenet": SqueezeNetModel.init
            }
    if _type.startswith("vgg"):
        return func[_type](batch_norm=_batch_norm)
    return func[_type]()


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
    parser.add_argument("-ftrain", default='../dataset/train',
                        help="Folder which has training data."
                             " Default is '../dataset/train'."
                        )
    parser.add_argument("-ftest", default='../dataset/test',
                        help="Folder which has test data."
                             " Default is '../dataset/test'."
                        )

    # parser.add_argument("-fval", default='../dataset/validation',
    #                     help="Folder which has validation data."
    #                          " Default is '../dataset/validation'."
    #                     )

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
    parser.add_argument("-valratio", type=float, default=0.0,
                        help="Ratio for Train Set / Validation Set split")
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

    parser.add_argument("-train", type=int, default=0,
                        help="For every given integer it will run as that much epochs. After each run of epocs "
                             "if test is activated test will be applied to model. Default is single run of 50 epochs."
                             " Default is '0' which means disabled."
                        )
    parser.add_argument("-test", action='store_true',
                        help="Evaluate test data set."
                        )

    parser.add_argument("-validation", type=int, default=0,
                        help="For given value, validation will be applied after that epochs. For example if the value "
                             "is 1, it will apply test after 1 epoch. Default is 0 which means disabled.")

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

    operator = None
    model = None
    if args.load is not None:
        model = load(args.model_path, args.name)

    elif args.train > 0 or args.test:
        model = init(args.type, args.batch_norm)

    if model is not None:
        operator = ModelOperator(model, args.model_path, args.output_path, args.mname, args.loss, args.cuda)

    if args.freeze:
        operator.freeze(args.clip)

    if args.validation != 0:
        if args.validation > args.train:
            raise Exception("Tests instances can`t be bigger than epochs.")
        _iter = int(args.train/args.validation)
        epochs = [args.validation for _ in range(_iter)]
        if args.train % args.validation != 0:
            epochs.append(args.train % args.validation)
    else:
        epochs = [args.train]

    if args.train > 0:
        operator.update_train_dataset(args.ftrain, args.batch, args.valratio)
        if args.validation != 0:
            operator.update_val_dataset(args.fval, args.batch)

        for i, epoch in enumerate(epochs):
            print("Training of (", i + 1, "/", len(epochs), ") with epoch [", epoch, "] initializing...")
            operator.train(epoch=epoch, lr=args.lr, momentum=args.momentum, write=True)
            if args.validation != 0:
                print("Validation of (", i + 1, "/", len(epochs), ") initializing...")
                operator.validate(write=True)

    if args.test:
        operator.update_test_dataset(args.ftest, args.batch)
        preds, ids = operator.test(write=True)

    if operator is not None:
        operator.save_info()
    if args.save:
        operator.save()

    if args.plot:
        plot_all(path.join(args.output_path, args.mname) + '.npy', args.figure_path, args.mname)

    return model

# sys.argv.extend("-train 1 -save -lr 0.01 -batch 32 -freeze -clip 0 26 -mname vgg-mse-10.32 -cuda".split())
# sys.argv.extend("-train 10 10 10 -save -test -lr 0.01 -batch 16 -mname vgg-mse-full-3.0 -cuda".split())
# sys.argv.extend("-test -load vgg-full-3.0 -cuda".split())
# sys.argv.extend("-plot -mname vgg-mse-5.0".split())
# sys.argv.extend("-plot -mname vgg-mse-full-2.0".split())

# sys.argv.extend("-train 1 1 -test -lr 0.01 -batch 32 -loss mse -freeze -clip 0 26 -cuda".split())

model = main()
