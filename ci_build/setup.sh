# exit when any command fails
set -e

apt update
apt -y install python3-pip
apt install rustc
pip3 install virtualenv