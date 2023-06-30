import argparse
import torch
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar10, Cifar100
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
import os
import sys; sys.path.append("..")
from optimizers import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epoch", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--annealing", default=True, type=bool, help="whether to use learning rate decay")
    parser.add_argument("--alg", default='adam3', type=str, help="optimization algorithm")
    parser.add_argument("--model", default='res-34', type=str, help="network architecture")
    parser.add_argument("--dataset", default='cifar10', type=str, help="dataset")

    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--final_lr', default=0.1, type=float,
                        help='final learning rate of AdaBound')
    parser.add_argument('--gamma', default=1e-3, type=float,
                        help='convergence speed term of AdaBound')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
    parser.add_argument('--eps', type=float, default=1e-8, help='very small term epsilon in adaptive gradient methods')
    parser.add_argument('--seed', type=int, default=41, help='random seed')


    args = parser.parse_args()

    initialize(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'cifar10':
        dataset = Cifar10(args.batch_size, args.threads)
    elif args.dataset == 'cifar100':
        dataset = Cifar100(args.batch_size, args.threads)
    else:
        raise NotImplementedError

    log = Log(log_each=10)

    if args.model == 'res-34':
        from model.resnet import ResNet34
        model = ResNet34().to(device)
    elif args.model == 'dense-121':
        from model.densenet import densenet_cifar
        model = densenet_cifar().to(device)
    elif args.model == 'vgg-16':
        from model.vgg import vgg16_bn
        model = vgg16_bn().to(device)
    else:
        raise NotImplementedError
        


    if args.alg == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.alg == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)
    elif args.alg =='adabound':
        from adabound import AdaBound
        optimizer = AdaBound(model.parameters(), args.lr, final_lr=args.final_lr, gamma=args.gamma, weight_decay=args.weight_decay, eps=args.eps)
    elif args.alg == 'adabelief':
        optimizer = AdaBelief(model.parameters(), args.lr, weight_decay=args.weight_decay, weight_decouple=False,rectify=False, eps=args.eps)
    elif args.alg == 'radam':
        optimizer = RAdam(model.parameters(), args.lr, weight_decay=args.weight_decay, eps=args.eps)
    elif args.alg == 'adamw':
        optimizer = AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay, eps=args.eps)
    elif args.alg == 'adam3':
        optimizer = AdaM3(model.parameters(), args.lr, weight_decay=args.weight_decay, weight_decouple=False, rectify=False, eps=args.eps)
    elif args.alg == 'yogi':
        optimizer = Yogi(model.parameters(), args.lr, weight_decay=args.weight_decay, eps=args.eps)


    if args.annealing:
        scheduler = StepLR(optimizer, args.lr, args.epoch)

    train_accuracies = []
    test_accuracies = []

    for epoch in range(args.epoch):
        model.train()
        log.train(len_dataset=len(dataset.train))
        train_accuracy = 0
        train_total = 0

        test_accuracy = 0
        test_total = 0

        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets)
            torch.autograd.set_detect_anomaly(True)
            loss.mean().backward()
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                train_accuracy += correct.sum().item()
                train_total += targets.size(0)
                log(model, loss.cpu(), correct.cpu(), args.lr)
                if args.annealing:
                    scheduler(epoch)

        train_accuracy = 100. * train_accuracy / train_total
        train_accuracies.append(train_accuracy)
        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                test_accuracy += correct.sum().item()
                test_total += targets.size(0)
                log(model, loss.cpu(), correct.cpu(), args.lr)

        test_accuracy = 100. * test_accuracy / test_total
        test_accuracies.append(test_accuracy)

    if args.dataset == 'cifar10':
        if not os.path.isdir('curve_cifar10'):
            os.mkdir('curve_cifar10')
        if not os.path.isdir(os.path.join('curve_cifar10', args.model)):
            os.mkdir(os.path.join('curve_cifar10', args.model))
        torch.save({'train_acc': train_accuracies, 'test_acc': test_accuracies},
                   os.path.join('curve_cifar10/{}'.format(args.model), '{}_seed{}'.format(args.alg, args.seed)))
        log.flush()
    elif args.dataset == 'cifar100':
        if not os.path.isdir('curve_cifar100'):
            os.mkdir('curve_cifar100')
        if not os.path.isdir(os.path.join('curve_cifar100', args.model)):
            os.mkdir(os.path.join('curve_cifar100', args.model))
        torch.save({'train_acc': train_accuracies, 'test_acc': test_accuracies},
                   os.path.join('curve_cifar100/{}'.format(args.model), '{}_seed{}'.format(args.alg, args.seed)))
        log.flush()
    else:
        raise NotImplementedError
