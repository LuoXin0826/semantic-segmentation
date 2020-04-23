sudo docker run -it --rm -p 8888:8888 -v ~/workspace/semantic-segmentation:/home/workspace nvidia-segmentation:latest bash

sudo docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 --ipc=host -it --rm -p 8888:8888 -v ~/workspace/semantic-segmentation:/home/workspace -v /media/youngji/StorageDevice/data/nvidia-segmentation/data_semantics_small:/home/dataset  nvidia-segmentation:latest bash

sudo docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=2,3 --ipc=host -it --rm -p 8888:8888 -v ~/workspace/semantic-segmentation:/home/workspace -v /media/youngji/StorageDevice/nvidia-segmentation/data_semantics:/home/dataset  nvidia-segmentation:latest bash

sudo docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 --ipc=host -it --rm -p 8888:8888 -v ~/workspace/semantic-segmentation:/home/workspace -v /media/youngji/StorageDevice/data/kitti_odometry/dataset/sequences/15/image_2:/home/dataset  nvidia-segmentation:latest bash

python demo_folder_trav.py --demo-folder /home/dataset/ --snapshot ./ckpts/last_epoch_89_mean-iu_0.42465.pth --save-dir ./out4/

python demo_folder_trav.py --demo-folder /home/dataset/ --snapshot ./ckpts/last_epoch_89_mean-iu_0.08543.pth --save-dir ./out/

python demo_folder.py --demo-folder /home/dataset/ --snapshot ./pretrained_models/kitti_best.pth --save-dir ./out3/

sudo docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1,2 --ipc=host -it --rm -p 8888:8888 -v ~/youngji/semantic-segmentation:/home/workspace -v ~/youngji/data_semantics_small:/home/dataset  nvidia-segmentation:latest bash

sudo docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 --ipc=host -it --rm -p 8888:8888 -v ~/workspace/semantic-segmentation:/home/workspace -v /media/youngji/StorageDevice/data/nvidia-segmentation/data_trav_test:/home/dataset  nvidia-segmentation:latest bash

./scripts/eval_kitti_WideResNet38.sh ./ckpts/decoder_part.pth ./eval/

pip install torchviz

