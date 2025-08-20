cd src
make clean
make
cd ../
taskset -c 0 ./bin/orchastractor
