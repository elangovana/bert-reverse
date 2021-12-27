# exit when any command fails
set -e

apt-get -y update
apt-get -y install python3-pip
#apt-get -y install llvm7.0
apt-get -y install rustc
pip3 install virtualenv