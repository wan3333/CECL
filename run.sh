# Please carefully check the code if you would like to use multiple GPUs

CUDA_VISIBLE_DEVICES=0 python train.py  -c config/cifar100.cfg --dist-url 'tcp://localhost:10002' \
--multiprocessing-distributed --world-size 1 --rank 0 --cosine --sigma 0.2 


