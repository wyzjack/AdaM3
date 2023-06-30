echo "=== DenseNet-121 on CIFAR-10 ==="

export CUDA_VISIBLE_DEVICES=2

echo "optimizing using AdaM3"
PYTHONIOENCODING=utf-8 python main.py --alg adam3 --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 0
PYTHONIOENCODING=utf-8 python main.py --alg adam3 --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 1
PYTHONIOENCODING=utf-8 python main.py --alg adam3 --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 2
PYTHONIOENCODING=utf-8 python main.py --alg adam3 --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 3
PYTHONIOENCODING=utf-8 python main.py --alg adam3 --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 4

echo "optimizing using SGD"
PYTHONIOENCODING=utf-8 python main.py --alg sgd --dataset cifar10 --model dense-121 --lr 0.1 --seed 0
PYTHONIOENCODING=utf-8 python main.py --alg sgd --dataset cifar10 --model dense-121 --lr 0.1 --seed 1
PYTHONIOENCODING=utf-8 python main.py --alg sgd --dataset cifar10 --model dense-121 --lr 0.1 --seed 2
PYTHONIOENCODING=utf-8 python main.py --alg sgd --dataset cifar10 --model dense-121 --lr 0.1 --seed 3
PYTHONIOENCODING=utf-8 python main.py --alg sgd --dataset cifar10 --model dense-121 --lr 0.1 --seed 4

echo "optimizing using Adam"
PYTHONIOENCODING=utf-8 python main.py --alg adam --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 0
PYTHONIOENCODING=utf-8 python main.py --alg adam --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 1
PYTHONIOENCODING=utf-8 python main.py --alg adam --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 2
PYTHONIOENCODING=utf-8 python main.py --alg adam --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 3
PYTHONIOENCODING=utf-8 python main.py --alg adam --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 4

echo "optimizing using AdamW"
PYTHONIOENCODING=utf-8 python main.py --alg adamw --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 0
PYTHONIOENCODING=utf-8 python main.py --alg adamw --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 1
PYTHONIOENCODING=utf-8 python main.py --alg adamw --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 2
PYTHONIOENCODING=utf-8 python main.py --alg adamw --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 3
PYTHONIOENCODING=utf-8 python main.py --alg adamw --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 4

echo "optimizing using Yogi"
PYTHONIOENCODING=utf-8 python main.py --alg yogi --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 0
PYTHONIOENCODING=utf-8 python main.py --alg yogi --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 1
PYTHONIOENCODING=utf-8 python main.py --alg yogi --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 2
PYTHONIOENCODING=utf-8 python main.py --alg yogi --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 3
PYTHONIOENCODING=utf-8 python main.py --alg yogi --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 4

echo "optimizing using AdaBound"
PYTHONIOENCODING=utf-8 python main.py --alg adabound --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 0
PYTHONIOENCODING=utf-8 python main.py --alg adabound --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 1
PYTHONIOENCODING=utf-8 python main.py --alg adabound --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 2
PYTHONIOENCODING=utf-8 python main.py --alg adabound --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 3
PYTHONIOENCODING=utf-8 python main.py --alg adabound --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 4

echo "optimizing using RAdam"
PYTHONIOENCODING=utf-8 python main.py --alg radam --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 0
PYTHONIOENCODING=utf-8 python main.py --alg radam --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 1
PYTHONIOENCODING=utf-8 python main.py --alg radam --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 2
PYTHONIOENCODING=utf-8 python main.py --alg radam --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 3
PYTHONIOENCODING=utf-8 python main.py --alg radam --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 4

echo "optimizing using AdaBelief"
PYTHONIOENCODING=utf-8 python main.py --alg adabelief --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 0
PYTHONIOENCODING=utf-8 python main.py --alg adabelief --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 1
PYTHONIOENCODING=utf-8 python main.py --alg adabelief --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 2
PYTHONIOENCODING=utf-8 python main.py --alg adabelief --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 3
PYTHONIOENCODING=utf-8 python main.py --alg adabelief --dataset cifar10 --model dense-121 --lr 0.001 --eps 1e-8 --seed 4









