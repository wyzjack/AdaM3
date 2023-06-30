echo "=== Three layer LSTM ==="

echo "optimizing using AdaM3"
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer adam3 --lr 0.001 --eps 1e-16 --nlayer 3 --seed 0 --run 0
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer adam3 --lr 0.001 --eps 1e-16 --nlayer 3 --seed 1 --run 1
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer adam3 --lr 0.001 --eps 1e-16 --nlayer 3 --seed 2 --run 2
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer adam3 --lr 0.001 --eps 1e-16 --nlayer 3 --seed 3 --run 3
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer adam3 --lr 0.001 --eps 1e-16 --nlayer 3 --seed 4 --run 4

echo "optimizing using AdaBelief"
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer adabelief --lr 0.01 --eps 1e-12 --nlayer 3 --seed 0 --run 0
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer adabelief --lr 0.01 --eps 1e-12 --nlayer 3 --seed 1 --run 1
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer adabelief --lr 0.01 --eps 1e-12 --nlayer 3 --seed 2 --run 2
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer adabelief --lr 0.01 --eps 1e-12 --nlayer 3 --seed 3 --run 3
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer adabelief --lr 0.01 --eps 1e-12 --nlayer 3 --seed 4 --run 4

echo "optimizing using SGD"
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer sgd --lr 30 --nlayer 3 --seed 0 --run 0
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer sgd --lr 30 --nlayer 3 --seed 1 --run 1
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer sgd --lr 30 --nlayer 3 --seed 2 --run 2
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer sgd --lr 30 --nlayer 3 --seed 3 --run 3
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer sgd --lr 30 --nlayer 3 --seed 4 --run 4

echo "optimizing using AdaBound"
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer adabound --lr 0.01 --eps 1e-8 --nlayer 3 --seed 0 --run 0
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer adabound --lr 0.01 --eps 1e-8 --nlayer 3 --seed 1 --run 1
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer adabound --lr 0.01 --eps 1e-8 --nlayer 3 --seed 2 --run 2
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer adabound --lr 0.01 --eps 1e-8 --nlayer 3 --seed 3 --run 3
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer adabound --lr 0.01 --eps 1e-8 --nlayer 3 --seed 4 --run 4

echo "optimizing using Yogi"
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer yogi --lr 0.01 --eps 1e-3 --nlayer 3 --seed 0 --run 0
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer yogi --lr 0.01 --eps 1e-3 --nlayer 3 --seed 1 --run 1
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer yogi --lr 0.01 --eps 1e-3 --nlayer 3 --seed 2 --run 2
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer yogi --lr 0.01 --eps 1e-3 --nlayer 3 --seed 3 --run 3
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer yogi --lr 0.01 --eps 1e-3 --nlayer 3 --seed 4 --run 4

echo "optimizing using Adam"
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer adam --lr 0.01 --eps 1e-8 --nlayer 3 --seed 0 --run 0
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer adam --lr 0.01 --eps 1e-8 --nlayer 3 --seed 1 --run 1
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer adam --lr 0.01 --eps 1e-8 --nlayer 3 --seed 2 --run 2
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer adam --lr 0.01 --eps 1e-8 --nlayer 3 --seed 3 --run 3
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer adam --lr 0.01 --eps 1e-8 --nlayer 3 --seed 4 --run 4

echo "optimizing using RAdam"
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer radam --lr 0.001 --eps 1e-8 --nlayer 3 --seed 0 --run 0
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer radam --lr 0.001 --eps 1e-8 --nlayer 3 --seed 1 --run 1
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer radam --lr 0.001 --eps 1e-8 --nlayer 3 --seed 2 --run 2
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer radam --lr 0.001 --eps 1e-8 --nlayer 3 --seed 3 --run 3
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer radam --lr 0.001 --eps 1e-8 --nlayer 3 --seed 4 --run 4

echo "optimizing using AdamW"
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer adamw --lr 0.001 --eps 1e-8 --nlayer 3 --seed 0 --run 0
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer adamw --lr 0.001 --eps 1e-8 --nlayer 3 --seed 1 --run 1
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer adamw --lr 0.001 --eps 1e-8 --nlayer 3 --seed 2 --run 2
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer adamw --lr 0.001 --eps 1e-8 --nlayer 3 --seed 3 --run 3
python main.py --epoch 200 --save PTB.pt --when 100 145 --beta1 0.9 --beta2 0.999 --optimizer adamw --lr 0.001 --eps 1e-8 --nlayer 3 --seed 4 --run 4