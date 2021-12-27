# exit when any command fails
set -e

yum -y update
yum -y install python3-pip
yum -y install llvm7.0
yum -y rustc
make
pip3 install virtualenv