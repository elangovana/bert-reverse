#!/usr/bin/env bash
 set -e

 echo "$@"

 VIRTUAL_ENV=$1
 TEST_REPORT_PATH=$2

 # Set up virtual env
 virtualenv -p python3 $VIRTUAL_ENV
 . $VIRTUAL_ENV/bin/activate

 #Install requirements
 pip install pyflakes==2.3.0
 # Work around for tokensior as wheel doesnt seem to work
 git clone https://github.com/huggingface/tokenizers
 git checkout tags/python-v0.10.3
 cd tokenizers/bindings/python
 pip install setuptools_rust
 python setup.py install
  cd ../../..
 # End of workaround

 pip install -r tests/requirements.txt

 #Run tests
 export PYTHONPATH=./src
 pytest tests --tb=short --junitxml=$TEST_REPORT_PATH

 #Run pyflakes to detect any import / syntax issues
 pyflakes ./**/*.py

 # Deactivate virtual envs
 deactivate