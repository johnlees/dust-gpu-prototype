
CXXFLAGS+=-Wall -Wextra -std=c++14 -fopenmp -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS
CUDAFLAGS = --std c++14
ifdef DEBUG
  CXXFLAGS+= -O0 -g
  CUDAFLAGS = -g -G
else ifdef PROFILE
  CUDAFLAGS = -pg --generate-line-info
  CXXFLAGS+= -O2 -g -flto -fno-fat-lto-objects -fPIC -fvisibility=hidden
else
  CXXFLAGS+= -O3 -flto -fno-fat-lto-objects -fPIC -fvisibility=hidden
endif

UNAME_S := $(shell uname -s)
LDLIBS = -lgomp
ifeq ($(UNAME_S),Linux)
	CXXFLAGS+= -m64
	ifdef PROFILE
		CXXFLAGS+= -Wl,--compress-debug-sections=none
	endif
	LDLIBS+= -lpthread -lm -ldl -lrt
	LDFLAGS=-Wl,-as-needed
endif
ifeq ($(UNAME_S),Darwin)
	LDLIBS+= -pthread
endif

CUDA_LDLIBS=-lcudadevrt -lcudart_static $(LDLIBS)

CUDA_LDFLAGS = -L${CUDA_HOME}/targets/x86_64-linux/lib/stubs -L${CUDA_HOME}/targets/x86_64-linux/lib
CUDAFLAGS += -Xcompiler -fPIC -Xptxas -dlcm=ca -Xcompiler -fopenmp --cudart static --relocatable-device-code=true

PROGRAMS=dust_test

DUST_OBJS=sireinfect.cu.o

all: $(PROGRAMS)

clean:
	$(RM) *.o *.so *.dylib ~* $(PROGRAMS)

install: all
	install -d $(BINDIR)
	install $(PROGRAMS) $(BINDIR)

dust_test: $(DUST_OBJS)
	nvcc $(CUDAFLAGS) $(CUDA_LDFLAGS) -Wno-deprecated-gpu-targets $^ -o $@ -Xnvlink $(CUDA_LDLIBS)

sireinfect.cu.o:
	nvcc $(CUDAFLAGS) $(CPPFLAGS) -DGPU_AVAILABLE -x cu -c sireinfect.cu -o $@

.PHONY: all clean install
