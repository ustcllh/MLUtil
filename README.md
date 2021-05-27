# Machine Learning Utilities For ACCRE

## PyTorch Docker Image Setup (CPU Backend)

Create a new docker container with Ubuntu (20.04.2 LTS)
```
docker run -it ubuntu
```

Setup Python
```
apt-get update
apt-get install python3
apt-get install python3-pip
```

Setup Pytorch
[Pytorch: Get Started](https://pytorch.org/get-started/locally/)
```
pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

Optional: Push to Docker Hub

```
docker commit <container id> <repo/name>:<tag>
docker push <repo/name>:<tag>
```

Optional: Pull from Docker Hub

```
docker pull <repo/name>:<tag>
```

## Digit Recognition
Clone this repo
```
git clone https://github.com/ustcllh/MLUtility.git
```

Digit Recognizer with CNN
```
python3 digit_recognition_conv.py
```



