ifdef ATI
CLINCLUDE=-I${ATISTREAMSDKROOT}/include/CL \
	-I${ATISTREAMSDKROOT}/include \
	-L${ATISTREAMSDKROOT}/lib/x86
LDFLAGS=-L${ATISTREAMSDKROOT}/lib/x86
else
CLINCLUDE=-I/usr/include/nvidia-current \
	-I/usr/include/nvidia-current/CL
LDFLAGS=-L/usr/lib/nvidia-current
endif
MATLABDIR=/opt/matlab2009b

CXXFLAGS=-g3 -Wall -Wextra -fPIC \
	${CLINCLUDE} \
	-I${MATLABDIR}/extern/include

all:
	${CXX} ${CXXFLAGS} -c mandelbrot.cpp
	${MATLABDIR}/bin/mex -cxx ${LDFLAGS} -lOpenCL mandelbrot.o
