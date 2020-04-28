
all:
	cython ../common/nms.pyx
	gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python3.6m -o ./lib/nms.so ../common/nms.c
	rm -rf ../common/nms.c