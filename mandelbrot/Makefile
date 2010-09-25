ifdef ATI
CLINCLUDE=-I${ATISTREAMSDKROOT}/include/CL \
	-I${ATISTREAMSDKROOT}/include \
	-L${ATISTREAMSDKROOT}/lib/x86
else
CLINCLUDE=-I/usr/include/nvidia-current \
	-I/usr/include/nvidia-current/CL
endif

CXXFLAGS=-g3 -Wall -Wextra \
	${CLINCLUDE} \
	-I/opt/MATLAB2008B/extern/include
LDFLAGS=-L${ATISTREAMSDKROOT}/lib/x86

all:
	${CXX} ${CXXFLAGS} -c mandelbrot.cpp
	mex -cxx ${LDFLAGS} -lOpenCL mandelbrot.o
