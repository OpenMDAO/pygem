
import os
import sys
import subprocess
from os.path import join, dirname, basename, abspath, expanduser, isdir, isfile, expandvars
import platform
import shutil
import fnmatch
import struct


def get_osx_distutils_files(startdir='.'):
    """Return a tuple of (objfile, libfile), the object and lib files that
    distutils builds when running 'python setup.py build' for pygem_quartz.
    Note that this applies only to the pygem_quartz distro on OSX and is not a 
    general routine.
    """
    objfile = libfile = None
    startdir = expand_path(startdir)
    for path, dirlist, filelist in os.walk(startdir):
        for name in filelist:
            fpath = os.path.join(path, name)
            if fpath.endswith('.so'):
                libfile = fpath
            elif fpath.endswith('.o'):
                objfile = fpath

    return (objfile, libfile)


def osx_hack(options, env, arch, srcdir, gv=None):
    dct = env.copy()
    mac_ver = '.'.join(platform.mac_ver()[0].split('.')[:2])
    objfile, libfile = get_osx_distutils_files(srcdir)
    dct['PYARCH'] = arch
    dct['OBJFNAME'] = objfile
    dct['LIBFNAME'] = libfile

    cmd = None
    if options.gem_type == 'quartz':
        print 'special relink for quartz on OSX %s...' % mac_ver
        xtras = ' -u _gixCADLoad -u _gibFillCoord -u _gibFillDNodes -u _gibFillQMesh -u _gibFillQuads -u _gibFillSpecial -u _gibFillTris -u _giiFillAttach -u _giuDefineApp -u _giuProgress -u _giuRegisterApp -u _giuSetEdgeTs -u _giuWriteApp -framework CoreFoundation -framework IOKit'
        cmd = "gcc-4.2 -Wl,-F. -bundle -undefined dynamic_lookup %(PYARCH)s %(OBJFNAME)s -L%(GEM_BLOC)s/lib -L%(CAPRILIB)s -L/usr/X11/lib -lgem -lquartz -lgem -lquartz -lcapriDyn -ldcapri" % dct
        if gv:
            cmd = cmd + " -lgv -lGLU -lGL -lX11 -lXext -lpthread -o %(LIBFNAME)s " % dct + xtras
            if mac_ver == '10.5':
                cmd = cmd + " -dylib_file /System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGL.dylib:/System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGL.dylib"
        else:
            cmd = cmd + " -lX11 -o %(LIBFNAME)s " % dct + xtras
    elif options.gem_type == 'diamond':
        if gv and mac_ver == '10.5':
            print "special relink for diamond gv on OSX 10.5..."
            cmd = "gcc -Wl,-F. -bundle -undefined dynamic_lookup %(PYARCH)s %(OBJFNAME)s -L%(GEM_BLOC)s/lib -L%(EGADSLIB)s -L/usr/X11/lib -lgem -ldiamond -legads -lgv -lGLU -lGL -lX11 -lXext -lpthread -o %(LIBFNAME)s -framework IOKit -framework CoreFoundation -dylib_file /System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGL.dylib:/System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGL.dylib" % dct

    if cmd:
        return subprocess.call(cmd, shell=True, env=os.environ,  
                               cwd=srcdir)


def copy(src, dst):
    """copy symlinks if present"""
    if (not sys.platform.startswith('win')) and os.path.islink(src):
        linkto = os.readlink(src)
        os.symlink(linkto, dst)
    else:
        shutil.copy(src, dst)

def expand_path(path):
    if path is not None:
        return abspath(expandvars(expanduser(path)))

def _get_dlibpath(libs):
    _lib_path_dct = {
        'darwin': 'DYLD_LIBRARY_PATH',
        'win32': 'PATH',
    }
    pname = _lib_path_dct.get(sys.platform, 'LD_LIBRARY_PATH')
    path = os.environ.get(pname)
    if path is None:
        path = ''
    parts = path.split(os.pathsep)
    for lib in libs:
        if lib not in parts:
            parts = [lib]+parts
    return (pname, os.pathsep.join([p for p in parts if os.path.isdir(p)]))

def _get_arch():
    """Get the architecture string (DARWIN, DARWIN64, LINUX, LINUX64, WIN32,
    WIN64)
    """
    arch_dct = {
        'darwin': 'DARWIN',
        'linux2': 'LINUX',
        'win32': 'WIN',
        }
    bits = struct.calcsize("P") * 8
    if bits == 32:
        if sys.platform == 'win32':
            return 'WIN32'
        else:
            return arch_dct[sys.platform]
    else: # assume 64 bit
        return arch_dct[sys.platform]+'64'
    
    
def _get_cas_rev(cas_root):
    for path, dirlist, filelist in os.walk(cas_root):
        for name in filelist:
            if name == 'Standard_Version.hxx':
                with open(join(path, name), 'r') as f:
                    for line in f:
                        parts = line.split()
                        if len(parts)>1 and parts[0] == '#define' and parts[1] == 'OCC_VERSION':
                            return parts[2]

def _get_occ_libs(rootpath, libpath, version_tup):
    extras = []
    if sys.platform.startswith('linux'):
        libs = fnmatch.filter(os.listdir(libpath), "*.so")
        libs.extend(fnmatch.filter(os.listdir(libpath), "*.so.*"))
    elif sys.platform == 'darwin':
        libs = fnmatch.filter(os.listdir(libpath), "*.dylib")
    elif sys.platform.startswith("win"):
        libpath = join(dirname(libpath), 'bin')
        libs = fnmatch.filter(os.listdir(libpath), "*.dll")
        if version_tup < ('6','6'):
            extras = [join(dirname(rootpath), '3rdparty', 'win32', 
                                    'tbb', 'bin', 'tbbmalloc.dll')]
        else:
            extras = [join(dirname(rootpath), '3rdparty', 'tbb30_018oss',
                               'bin', 'ia32', 'vc9', 'tbbmalloc.dll')]
    return [join(libpath, lib) for lib in libs]+extras

def _get_capri_libs(libpath):
    libpath = expand_path(libpath)
    if sys.platform.startswith('darwin'):
        libs = fnmatch.filter(os.listdir(libpath), "*.SO")
        libs.extend(fnmatch.filter(os.listdir(libpath), "*.dylib"))
    elif sys.platform.startswith('win'):
        libs = fnmatch.filter(os.listdir(libpath), "*.dll")
        for rem in ['capriCS.dll', 'capriSCS.dll']:
            if rem in libs: 
                libs.remove(rem)
    elif sys.platform.startswith('linux'):
        libs = fnmatch.filter(os.listdir(libpath), "*.so")        
    else:
        raise NotImplementedError("current platform not supported")
    return [join(libpath, lib) for lib in libs]

def _get_capri_key(libpath):
    keyfile = join(dirname(libpath), 'KEYgen.txt')
    with open(keyfile, 'r') as f:
        for line in f:
            if 'CAPRIkey' in line:
                line = line.strip()
                return line.split('=')[1].strip()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--esp", action="store", type=str,
                        dest='esp_dir', help="Engineering Sketchpad directory")
    parser.add_argument("--casroot", action="store", type=str,
                        dest='casroot', help="OpenCASCADE root directory")
    parser.add_argument("--gemroot", action="store", type=str,
                        dest='gemroot', help="GEM root directory")
    parser.add_argument("--caprilib", action="store", type=str,
                        dest='caprilib', help="Capri lib directory (used with gemtype=quartz)")
    parser.add_argument("--capriinc", action="store", type=str,
                        dest='capriinc', help="Capri include directory (used with gemtype=quartz)")
    parser.add_argument("--casrev", action="store", type=str,
                        dest='casrev', help="OpenCASCADE revision number")
    parser.add_argument("-c", "--clean", action="store_true", dest="clean",
                        help="do a make clean before building")
    parser.add_argument("--bdist_egg", action="store_true", dest="bdist_egg",
                        help="build a binary egg for pygem")
    parser.add_argument("--develop", action="store_true", dest="develop",
                        help="build a 'develop' dist for pygem")
    parser.add_argument("--inplace", action="store_true", dest="inplace",
                        help="build an in-place dist for pygem")
    parser.add_argument("--sdist", action="store_true", dest="sdist",
                        help="build a source distribution for pygem")
    parser.add_argument("--gv", action="store_true", dest="gv",
                        help="link in the gv libraries")
    parser.add_argument("--gemtype", action="store", type=str,
                        dest='gem_type', help="GEM type (diamond or quartz)")
    parser.add_argument("--debug", action="store_true", dest="debug",
                        help="add debug symbols")
  
    options = parser.parse_args()
    
    if not options.gem_type:
        print '\nYou must specify a GEM type (diamond or quartz)\n'
        parser.print_help()
        sys.exit(-1)

    distroot = join(dirname(abspath(__file__)), 'pygem_'+options.gem_type)
    pkg_name = 'pygem_'+options.gem_type
    
    capri = options.caprilib or options.capriinc
    
    if options.esp_dir:
        esp_dir = expand_path(options.esp_dir)
        esp_src = join(esp_dir,'src')
        esp_libs = join(esp_dir, 'lib')
        egads_lib = join(esp_dir, 'lib')
    
        
    if options.casroot:
        cas_rev = options.casrev
        cas_root = expand_path(options.casroot)
        if sys.platform.startswith('win'):
            cas_root = join(cas_root, 'ros')
        if not isdir(cas_root):
            print "OpenCASCADE directory %s doesn't exist\n" % cas_root
            sys.exit(-1)
              
        if cas_rev is None:
            cas_rev = _get_cas_rev(cas_root)

        if cas_rev is None:
            print "Can't determine OpenCASCADE revision\n"
            sys.exit(-1)

        tup = tuple(cas_rev.split('.'))

        if tup < ('6','6'):
            vc_rev = 'vc8'
        else:
            vc_rev = 'vc9'

        if sys.platform.startswith('win'):
            # TODO: make the determination of cas_lib on windows more robust
            cas_lib = join(cas_root, 'win32', vc_rev, 'lib')
        else:
            cas_lib = join(cas_root, 'lib')

    if options.gem_type == 'diamond':
        if options.casroot is None:
            print "OpenCASCADE directory must be supplied\n"
            parser.print_help()
            sys.exit(-1)
        if options.esp_dir is None:
            print "Engineering Sketchpad directory must be supplied\n"
            parser.print_help()
            sys.exit(-1)
            
        libs = [egads_lib, cas_lib]
    elif options.gem_type == 'quartz':
        libs = [options.caprilib]
     
    if options.esp_dir and not isdir(esp_dir):
        print "Engineering Sketchpad directory %s doesn't exist\n" % esp_dir
        sys.exit(-1)
            
    # make a lib dir inside of our package where we can put all of
    # the libraries that we'll include in the binary distribution
    pygem_libdir = join(dirname(abspath(__file__)), pkg_name, pkg_name, 'lib')

    if options.develop:
        libs.append(pygem_libdir)

    lib_path_tup = _get_dlibpath(libs)
    arch = _get_arch()

    env = {
        'GEM_ARCH': arch,
        'GEM_TYPE': options.gem_type,
        'GEM_BLOC': expand_path(options.gemroot),
        lib_path_tup[0]: lib_path_tup[1],
    }
    
    if arch.endswith('64'):
        arch = arch[:-2]

    if options.esp_dir:
        env['GEM_ROOT'] = esp_dir

    if sys.platform == 'darwin':
        env['MACOSX'] = '.'.join(platform.mac_ver()[0].split('.')[0:2])
        
    if options.gem_type == 'diamond':
        env.update({
            'OCSM_SRC': join(esp_dir, 'src', 'OpenCSM'),
            'EGADSINC': join(esp_dir, 'src', 'EGADS', 'include'),
            'EGADSLIB': egads_lib,
            'CASROOT': cas_root,
            'CASREV': cas_rev,
            'CASARCH': arch[0]+arch[1:].lower(),
            'LIBPATH': os.pathsep.join(libs),
        })
        if sys.platform.startswith('win'):
            env['CASARCH'] = env['CASARCH']+'\\'+vc_rev
    elif options.gem_type == 'quartz':
        env['CAPRILIB'] = expand_path(options.caprilib)
        env['CAPRIINC'] = expand_path(options.capriinc)
        env['CAPRIkey'] = _get_capri_key(env['CAPRILIB'])
        if sys.platform.startswith('linux'):
            #env['LDSHARED'] = '"gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro"'
            env['LDSHARED'] = '"gcc -pthread -shared -fPIC -Wl,-O1 -Wl,-z,relro"'

    if options.gv:
        env['GEM_GRAPHICS'] = 'gv'

    # generate some shell scripts here to set up the environment for those
    # that want to do things by running make directly
    shfile = open('gemEnv.sh', 'w')
    cshfile = open('gemEnv.csh', 'w')
    try:
        for name, val in env.items():
            shfile.write('export %s=%s\n' % (name, val))
            cshfile.write('setenv %s %s\n' % (name, val))
    finally:
        shfile.close()
        cshfile.close()

    if 'LDSHARED' in env:
        env['LDSHARED'] = env['LDSHARED'].strip('"')

    # update the current environment
    os.environ.update(env)

    # we'll do a make in these directories
    srcdirs = [join(env['GEM_BLOC'], 'src'),
               join(env['GEM_BLOC'], options.gem_type)]

    if options.esp_dir:
        srcdirs = [esp_src]+srcdirs
 
    if options.clean:
        for srcdir in srcdirs:
            ret = subprocess.call('make clean', shell=True, env=os.environ, 
                                  cwd=srcdir)
        shutil.rmtree(os.path.join(pkg_name, 'build'), ignore_errors=True)
        shutil.rmtree(os.path.join(pkg_name, pkg_name, 'lib'), ignore_errors=True)
        shutil.rmtree(os.path.join(pkg_name, pkg_name+'.egg-info'), ignore_errors=True)
        for ext in ['.c', '.so', '.dylib', '.pyd', '.dll']:
            name = os.path.join(pkg_name, pkg_name, 'gem'+ext)
            try:
                os.remove(name)
            except:
                print name,'not found'

    for srcdir in srcdirs:
        print 'calling make in %s' % srcdir
        ret = subprocess.call('make', shell=True, env=os.environ, 
                              cwd=srcdir)
        if ret != 0:
            print 'return from make was %s. aborting...' % ret
            sys.exit(ret)
            
    if isdir(pygem_libdir):
        shutil.rmtree(pygem_libdir)
    os.mkdir(pygem_libdir)
    
    if options.esp_dir:
        # collect EngSketchPad libs (egads, opencsm, ...)
        print 'Copying Engineering Sketchpad libs...'
        for name in os.listdir(esp_libs):
            lname = join(esp_libs, name)
            print lname
            copy(lname, join(pygem_libdir, name))
    
    # collect opencascade libs
    if options.casroot:
        print 'Copying OpenCASCADE libs'
        for libpath in _get_occ_libs(cas_root, cas_lib, tup):
            print libpath
            copy(libpath, join(pygem_libdir, basename(libpath)))

        # OCC adds a dependency on IEshims.dll on Windows
        if sys.platform.startswith('win'):
            shims = join(os.environ.get('ProgramFiles',''), 
                           'Internet Explorer', 'IEShims.dll')
            if isfile(shims):
                print shims
                copy(shims, join(pygem_libdir, basename(shims)))

    if options.caprilib:
        print 'Copying CAPRI libs'
        for libpath in _get_capri_libs(options.caprilib):
            print libpath
            copy(libpath, join(pygem_libdir, basename(libpath)))
    
    pkgdir = os.path.join(dirname(abspath(__file__)), pkg_name)

    print 'Compiling cython files'
    sys.stdout.flush()
    dbg = '--gdb' if options.debug else ''
    ret = subprocess.call("cython %s -v --fast-fail gem.pyx" % dbg,
                           shell=True, env=os.environ, 
                           cwd=os.path.join(pkgdir,pkg_name))
    if ret != 0:
        sys.exit(ret)

    print 'Calling setup in %s' % pkgdir
    if options.debug:
        arg = '--pyrex-gdb'
        interp = 'python-dbg'
    else:
        arg = ''
        interp = 'python'
    cmd = "%s setup.py build_ext %s -f" % (interp, arg)
    if options.inplace:
        cmd += " --inplace"
    sys.stdout.flush()
    ret = subprocess.call(cmd, shell=True, env=os.environ, 
                          cwd=pkgdir)

    if ret != 0:
        sys.exit(ret)

    if sys.platform == 'darwin':
        if arch == 'DARWIN64':
            arch = '-arch x86_64'
        else:
            arch = '-arch i386'
        ret = osx_hack(options, env, arch, distroot, gv=options.gv)
        if ret:
            print "OSX hack failed"
            sys.exit(-1)

    # build a binary egg distribution
    if options.bdist_egg:
        sys.stdout.flush()
        ret = subprocess.call("python setup.py bdist_egg",
                              shell=True, env=os.environ,
                              cwd=pkgdir)
        if ret != 0:
            sys.exit(ret)

        
    if options.develop:
        sys.stdout.flush()
        ret = subprocess.call("%s setup.py develop" % interp,
                              shell=True, env=os.environ,
                              cwd=pkgdir)
        if ret != 0:
            sys.exit(ret)
        
    # build a source distribution
    if options.sdist:
        sys.stdout.flush()
        ret = subprocess.call("python setup.py sdist",
                              shell=True, env=os.environ,
                              cwd=pkgdir)


