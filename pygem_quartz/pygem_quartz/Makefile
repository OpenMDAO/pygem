#	Makefile for pygem

ifeq ($(GEM_ARCH),DARWIN64)
PYARCH = -arch x86_64
else
PYARCH = -arch i386
endif

#
#	targets for make
#
all	: pygem.c
	@echo "Making test/gem.so..."
	-/bin/rm -rf build test/gem.so
	-python setup.py build
ifeq ($(GEM_TYPE),quartz)
ifeq ($(MACOSX_DEPLOYMENT_TARGET),10.5)
ifeq ($(GEM_GRAPHICS),gv)
	@echo "special relink for quartz gv on MACOSX_DEPLOYMENT_TARGET=10.5..."
	gcc-4.2 -Wl,-F. -bundle -undefined dynamic_lookup $(PYARCH) build/temp.macosx-10.5-*-2.5/pygem.o -L$(GEM_BLOC)/lib -L$(CAPRILIB) -L/usr/X11/lib -lgem -lquartz -lgem -lquartz -lcapriDyn -ldcapri -lgv -lGLU -lGL -lX11 -lXext -lpthread -o test/gem.so -u _gixCADLoad -u _gibFillCoord -u _gibFillDNodes -u _gibFillQMesh -u _gibFillQuads -u _gibFillSpecial -u _gibFillTris -u _giiFillAttach -u _giuDefineApp -u _giuProgress -u _giuRegisterApp -u _giuSetEdgeTs -u _giuWriteApp -framework CoreFoundation -framework IOKit -dylib_file /System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGL.dylib:/System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGL.dylib
else
	@echo "special relink for quartz MACOSX_DEPLOYMENT_TARGET=10.5..."
	gcc-4.2 -Wl,-F. -bundle -undefined dynamic_lookup $(PYARCH) build/temp.macosx-10.5-*-2.5/pygem.o -L$(GEM_BLOC)/lib -L$(CAPRILIB) -L/usr/X11/lib -lgem -lquartz -lgem -lquartz -lcapriDyn -ldcapri -lX11 -o test/gem.so -u _gixCADLoad -u _gibFillCoord -u _gibFillDNodes -u _gibFillQMesh -u _gibFillQuads -u _gibFillSpecial -u _gibFillTris -u _giiFillAttach -u _giuDefineApp -u _giuProgress -u _giuRegisterApp -u _giuSetEdgeTs -u _giuWriteApp -framework CoreFoundation -framework IOKit
endif
endif
ifeq ($(MACOSX_DEPLOYMENT_TARGET),10.6)
ifeq ($(GEM_GRAPHICS),gv)
	@echo "special relink for quartz gv on MACOSX_DEPLOYMENT_TARGET=10.6..."
	gcc-4.2 -Wl,-F. -bundle -undefined dynamic_lookup $(PYARCH) build/temp.macosx-10.6-*-2.6/pygem.o -L$(GEM_BLOC)/lib -L$(CAPRILIB) -L/usr/X11/lib -lgem -lquartz -lgem -lquartz -lcapriDyn -ldcapri -lgv -lGLU -lGL -lX11 -lXext -lpthread -o test/gem.so -u _gixCADLoad -u _gibFillCoord -u _gibFillDNodes -u _gibFillQMesh -u _gibFillQuads -u _gibFillSpecial -u _gibFillTris -u _giiFillAttach -u _giuDefineApp -u _giuProgress -u _giuRegisterApp -u _giuSetEdgeTs -u _giuWriteApp -framework CoreFoundation -framework IOKit
else
	@echo "special relink for quartz MACOSX_DEPLOYMENT_TARGET=10.6..."
	gcc-4.2 -Wl,-F. -bundle -undefined dynamic_lookup $(PYARCH) build/temp.macosx-10.6-*-2.6/pygem.o -L$(GEM_BLOC)/lib -L$(CAPRILIB) -L/usr/X11/lib -lgem -lquartz -lgem -lquartz -lcapriDyn -ldcapri -lX11 -o test/gem.so -u _gixCADLoad -u _gibFillCoord -u _gibFillDNodes -u _gibFillQMesh -u _gibFillQuads -u _gibFillSpecial -u _gibFillTris -u _giiFillAttach -u _giuDefineApp -u _giuProgress -u _giuRegisterApp -u _giuSetEdgeTs -u _giuWriteApp -framework CoreFoundation -framework IOKit
endif
endif
ifeq ($(MACOSX_DEPLOYMENT_TARGET),10.7)
ifeq ($(GEM_GRAPHICS),gv)
	@echo "special relink for quartz gv on MACOSX_DEPLOYMENT_TARGET=10.7..."
	llvm-gcc-4.2 -Wl,-F. -bundle -undefined dynamic_lookup $(PYARCH) build/temp.macosx-10.7-*-2.7/pygem.o -L$(GEM_BLOC)/lib -L$(CAPRILIB) -L/usr/X11/lib -lgem -lquartz -lgem -lquartz -lcapriDyn -ldcapri -lgv -lGLU -lGL -lX11 -lXext -lpthread -o test/gem.so -u _gixCADLoad -u _gibFillCoord -u _gibFillDNodes -u _gibFillQMesh -u _gibFillQuads -u _gibFillSpecial -u _gibFillTris -u _giiFillAttach -u _giuDefineApp -u _giuProgress -u _giuRegisterApp -u _giuSetEdgeTs -u _giuWriteApp -framework CoreFoundation -framework IOKit
else
	@echo "special relink for quartz MACOSX_DEPLOYMENT_TARGET=10.7..."
	llvm-gcc-4.2 -Wl,-F. -bundle -undefined dynamic_lookup $(PYARCH) build/temp.macosx-10.7-*-2.7/pygem.o -L$(GEM_BLOC)/lib -L$(CAPRILIB) -L/usr/X11/lib -lgem -lquartz -lgem -lquartz -lcapriDyn -ldcapri -lX11 -o test/gem.so -u _gixCADLoad -u _gibFillCoord -u _gibFillDNodes -u _gibFillQMesh -u _gibFillQuads -u _gibFillSpecial -u _gibFillTris -u _giiFillAttach -u _giuDefineApp -u _giuProgress -u _giuRegisterApp -u _giuSetEdgeTs -u _giuWriteApp -framework CoreFoundation -framework IOKit
endif
endif
else
ifeq ($(MACOSX_DEPLOYMENT_TARGET),10.5)
ifeq ($(GEM_GRAPHICS),gv)
	@echo "special relink for diamond gv on MACOSX_DEPLOYMENT_TARGET=10.5..."
	gcc -Wl,-F. -bundle -undefined dynamic_lookup $(PYARCH) build/temp.macosx-10.5-*-2.5/pygem.o -L$(GEM_BLOC)/lib -L$(EGADSLIB) -L/usr/X11/lib -lgem -ldiamond -legads -lgv -lGLU -lGL -lX11 -lXext -lpthread -o test/gem.so -framework IOKit -framework CoreFoundation -dylib_file /System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGL.dylib:/System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGL.dylib
endif
endif
endif
	-/bin/rm -rf build

clean   :
	@echo "Removing build tree..."
	-/bin/rm -rf build test/gem.so

help    :
	@echo "'make'           builds the programs"
	@echo "'make clean'     clean up"
	@echo "'make help'      print this listing"

