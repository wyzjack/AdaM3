echo "=== dense-121 on CIFAR-10 ==="


echo "seed 0"
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python main.py --alg adam3 --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 0 & CUDA_VISIBLE_DEVICES=1 PYTHONIOENCODING=utf-8 python main.py --alg sgd --dataset cifar100 --model dense-121 --lr 0.1 --seed 0 & CUDA_VISIBLE_DEVICES=2 PYTHONIOENCODING=utf-8 python main.py --alg adamw --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 0 & CUDA_VISIBLE_DEVICES=3 PYTHONIOENCODING=utf-8 python main.py --alg adam --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 0 & CUDA_VISIBLE_DEVICES=4 PYTHONIOENCODING=utf-8 python main.py --alg radam --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 0& CUDA_VISIBLE_DEVICES=5 PYTHONIOENCODING=utf-8 python main.py --alg adabelief --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 0 & CUDA_VISIBLE_DEVICES=6 PYTHONIOENCODING=utf-8 python main.py --alg yogi --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 0 & CUDA_VISIBLE_DEVICES=7 PYTHONIOENCODING=utf-8 python main.py --alg adabound --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 0

echo "seed 1"
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python main.py --alg adam3 --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 1 & CUDA_VISIBLE_DEVICES=1 PYTHONIOENCODING=utf-8 python main.py --alg sgd --dataset cifar100 --model dense-121 --lr 0.1 --seed 1 & CUDA_VISIBLE_DEVICES=2 PYTHONIOENCODING=utf-8 python main.py --alg adamw --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 1 & CUDA_VISIBLE_DEVICES=3 PYTHONIOENCODING=utf-8 python main.py --alg adam --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 1 & CUDA_VISIBLE_DEVICES=4 PYTHONIOENCODING=utf-8 python main.py --alg radam --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 1& CUDA_VISIBLE_DEVICES=5 PYTHONIOENCODING=utf-8 python main.py --alg adabelief --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 1 & CUDA_VISIBLE_DEVICES=6 PYTHONIOENCODING=utf-8 python main.py --alg yogi --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 1 & CUDA_VISIBLE_DEVICES=7 PYTHONIOENCODING=utf-8 python main.py --alg adabound --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 1

echo "seed 2"
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python main.py --alg adam3 --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 2 & CUDA_VISIBLE_DEVICES=1 PYTHONIOENCODING=utf-8 python main.py --alg sgd --dataset cifar100 --model dense-121 --lr 0.1 --seed 2 & CUDA_VISIBLE_DEVICES=2 PYTHONIOENCODING=utf-8 python main.py --alg adamw --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 2 & CUDA_VISIBLE_DEVICES=3 PYTHONIOENCODING=utf-8 python main.py --alg adam --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 2 & CUDA_VISIBLE_DEVICES=4 PYTHONIOENCODING=utf-8 python main.py --alg radam --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 2& CUDA_VISIBLE_DEVICES=5 PYTHONIOENCODING=utf-8 python main.py --alg adabelief --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 2 & CUDA_VISIBLE_DEVICES=6 PYTHONIOENCODING=utf-8 python main.py --alg yogi --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 2 & CUDA_VISIBLE_DEVICES=7 PYTHONIOENCODING=utf-8 python main.py --alg adabound --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 2

echo "seed 3"
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python main.py --alg adam3 --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 3 & CUDA_VISIBLE_DEVICES=1 PYTHONIOENCODING=utf-8 python main.py --alg sgd --dataset cifar100 --model dense-121 --lr 0.1 --seed 3 & CUDA_VISIBLE_DEVICES=2 PYTHONIOENCODING=utf-8 python main.py --alg adamw --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 3 & CUDA_VISIBLE_DEVICES=3 PYTHONIOENCODING=utf-8 python main.py --alg adam --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 3 & CUDA_VISIBLE_DEVICES=4 PYTHONIOENCODING=utf-8 python main.py --alg radam --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 3& CUDA_VISIBLE_DEVICES=5 PYTHONIOENCODING=utf-8 python main.py --alg adabelief --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 3 & CUDA_VISIBLE_DEVICES=6 PYTHONIOENCODING=utf-8 python main.py --alg yogi --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 3 & CUDA_VISIBLE_DEVICES=7 PYTHONIOENCODING=utf-8 python main.py --alg adabound --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 3

echo "seed 4"
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python main.py --alg adam3 --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 4 & CUDA_VISIBLE_DEVICES=1 PYTHONIOENCODING=utf-8 python main.py --alg sgd --dataset cifar100 --model dense-121 --lr 0.1 --seed 4 & CUDA_VISIBLE_DEVICES=2 PYTHONIOENCODING=utf-8 python main.py --alg adamw --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 4 & CUDA_VISIBLE_DEVICES=3 PYTHONIOENCODING=utf-8 python main.py --alg adam --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 4 & CUDA_VISIBLE_DEVICES=4 PYTHONIOENCODING=utf-8 python main.py --alg radam --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 4& CUDA_VISIBLE_DEVICES=5 PYTHONIOENCODING=utf-8 python main.py --alg adabelief --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 4 & CUDA_VISIBLE_DEVICES=6 PYTHONIOENCODING=utf-8 python main.py --alg yogi --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 4 & CUDA_VISIBLE_DEVICES=7 PYTHONIOENCODING=utf-8 python main.py --alg adabound --dataset cifar100 --model dense-121 --lr 0.001 --eps 1e-8 --seed 4












