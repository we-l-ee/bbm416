python main.py -test -load alexnet-lazy-adam-1.0 -mname alexnet-lazy-adam-1.0 -ftest C:/dataset/test -testdtype lazy -cuda -batch 400
python main.py -test -load vgg16-full-lazy-1 -mname vgg16-full-lazy-1 -ftest C:/dataset/test -testdtype lazy -cuda -batch 12
python main.py -test -load resnet18-lazy-1.0 -mname resnet18-lazy-1.0 -ftest C:/dataset/test -testdtype lazy -batch 12