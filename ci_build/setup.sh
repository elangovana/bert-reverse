# exit when any command fails
set -e

yum -y update
yum -y install python3-pip
yum install llvm7.0
./configure --enable-rust
make
pip3 install virtualenv