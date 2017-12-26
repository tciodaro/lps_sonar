#Makefile

#author: natmourajr@gmail.com
CC=g++
LD=g++
SWG= swig
VERSION=2.0.0

SWGFLAGS= -c++ -python
CCFLAGS= -Wall $(shell python-config --includes) -fPIC -O3
LDFLAGS= -Wall $(shell python-config --ldflags)  -fPIC -O3
               
               

SWIGFILES = python/neuralnet_wrap.o

SRCS= nets/src/neuralnet.cxx \
      nets/src/backpropagation.cxx \
      nets/src/rprop.cxx \
      nets/src/similarities.cxx \
      nets/src/artnet.cxx \
      train/src/trninfo.cxx \
      train/src/trainbp.cxx \
      train/src/trainart.cxx \
      io/src/iomgr.cxx \
            
OBJ=$(SRCS:.cxx=.o)

LIB=lib/_neuralnet.so

python/%_wrap.cxx: python/%.i
	@echo SWIGing $@ 
	$(SWG) $(SWGFLAGS) $<


%.o: %.cxx
	@echo Compiling $@ 
	$(CC) -o $@ -c $(CCFLAGS) $<

all: $(LIB) $(OBJ)
	@echo "Everthing DONE"

clean:
	@echo Cleaning all objects and executables
	@rm -f ./*/src/*.o ./*/src/*~ ./*/inc/*~ ./python/*_wrap.* ./python/*.py
	@rm -f $(LIB)

$(LIB): $(SWIGFILES) $(OBJ)
	@echo "**"
	@echo "** Linking library $@"
	@echo "**"
	$(LD) $(LDFLAGS) -shared -o $@ $(SWIGFILES) $(OBJ)
	@mv python/*.py lib/



