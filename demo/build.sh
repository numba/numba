#export CFLAGS='-O0 -g'
touch provider.pyx
touch consumer.pyx
python setup.py build_ext -i
