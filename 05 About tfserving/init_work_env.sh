sudo passwd
sudo snap install pycharm-community --classic
sudo apt-get install net-tools

sudo apt install python3
sudo apt install ipython3
sudo apt install python3-pip
sudp apt install jupyter-notebook
pip3 install jupyter
pip3 install -r requirements.txt


sudo apt-get install docker
sudo apt-get install docker.io
sudo gpasswd -a ${USER} docker
newgrp - docker
sudo service docker restart
sudo echo '\
{ \
"registry-mirrors": ["https://alzgoonw.mirror.aliyuncs.com"] \
}\
' >>  /etc/docker/daemon.json

systemctl daemon-reload
systemctl restart docker
sudo docker run hello-world