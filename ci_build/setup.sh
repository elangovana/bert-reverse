# exit when any command fails
set -e

yum update
yum -y install python3-pip
yum install rustc
pip3 install virtualenv