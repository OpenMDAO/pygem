from setuptools import setup, Extension
import os
import sys

try:
    import numpy
    numpy_include = os.path.join(os.path.dirname(numpy.__file__), 
                                 'core', 'include')
except ImportError:
    print 'numpy was not found.  Aborting build'
    sys.exit(-1)

# GEM_ARCH must be one of "DARWIN", "DARWIN64", LINUX64", "WIN32", or "WIN64"
gem_arch = os.environ['GEM_ARCH']

gem_type = 'quartz'
pkg_name = 'pygem_' + gem_type

# These environment variables are usually set for GEM builds:
gemlib = os.path.join(os.environ['GEM_BLOC'], 'lib')

caprilib = os.environ['CAPRILIB']

print '\nMaking "gem.so" for "%s" (on %s)\n' % (gem_type, gem_arch)

gem_include_dirs       = ['../include', numpy_include]

if gem_arch.startswith('DARWIN'):
    lib_stuff = ["lib/*.dylib", "lib/*.SO"]
    if gem_arch == "DARWIN64":
        os.environ['ARCHFLAGS'] = '-arch x86_64'
    else:
        os.environ['ARCHFLAGS'] = '-arch i386'
    gem_extra_compile_args = []
    gem_library_dirs       = [gemlib, caprilib, '/usr/X11/lib']
    gem_libraries          = ['gem', 'quartz', 'gem', 'quartz', 'capriDyn', 'dcapri', 'X11']
    gem_extra_link_args    = ['-framework CoreFoundation -framework IOKit']
elif gem_arch.startswith('LINUX'):
    lib_stuff = ["lib/*.so", "lib/*.so.*"]
    gem_extra_compile_args = []
    gem_library_dirs       = [gemlib, caprilib, '/usr/X11R6/lib', '/usr/lib/x86_64-linux-gnu/libX11']
    gem_libraries          = ['gem', 'quartz', 'gem', 'quartz', 'capriDyn', 'dcapri', 'X11']
    gem_extra_link_args    = ['-rdynamic']
elif gem_arch == 'WIN32':
    lib_stuff = ["lib/*.dll"]
    gem_extra_compile_args = []
    gem_extra_link_args    = []
    gem_library_dirs       = [gemlib, caprilib]
    gem_libraries          = ['gem', 'quartz', 'capriDyn', 'dcapri']
elif gem_arch == 'WIN64':
    lib_stuff = ["lib/*.dll"]
    gem_extra_compile_args = ['-DLONGLONG']
    gem_extra_link_args    = []
    gem_library_dirs       = [gemlib, caprilib]
    gem_libraries          = ['gem', 'quartz', 'capriDyn', 'dcapri']

if (os.environ.get('GEM_GRAPHICS') == "gv"):
    print "...gv graphics is enabled\n"
    gem_include_dirs.append(egadsinc)
    gem_extra_compile_args.append('-DGEM_GRAPHICS=gv')

    if gem_arch.startswith('WIN'):
        gem_libraries.extend(['gvIMD','GLU32','OpenGL32','User32','GDI32'])
    elif gem_arch.startswith('LINUX'):
        gem_libraries.extend(['gv','GLU','GL','X11','Xext','pthread'])
    elif gem_arch.startswith("DARWIN"):
        gem_libraries.extend(['gv','GLU','GL','X11','Xext','pthread'])
        gem_extra_link_args.append('-framework IOKit -framework CoreFoundation')

# Legal keyword arguments for the Extension constructor
#    'name', 'sources', 'include_dirs',
#    'define_macros', 'undef_macros',
#    'library_dirs', 'libraries', 'runtime_library_dirs',
#    'extra_objects', 'extra_compile_args', 'extra_link_args',
#    'swig_opts', 'export_symbols', 'depends', 'language'

module1 = Extension(pkg_name+'.gem',
                    include_dirs       = gem_include_dirs,
                    extra_compile_args = gem_extra_compile_args,
                    library_dirs       = gem_library_dirs,
                    libraries          = gem_libraries,
                    extra_link_args    = gem_extra_link_args,
                    language           = 'c',
                    sources            = ['pygem_quartz/pygem.c'])

# Legal keyword arguments for the setup() function
#    'distclass', 'script_name', 'script_args', 'options',
#    'name', 'version', 'author', 'author_email',
#    'maintainer', 'maintainer_email', 'url', 'license',
#    'description', 'long_description', 'keywords',
#    'platforms', 'classifiers', 'download_url',
#    'requires', 'provides', 'obsoletes'

setup (name = pkg_name,
       version = '0.9.1',
       description = 'Python interface to GEM using OpenCSM and Capri',
       zip_safe = False,
       ext_modules = [module1],
       packages = [pkg_name],
       package_data = { pkg_name: ['test/*.py', 'test/*.BRep', 'test/*.col']+
                        lib_stuff
       },
      ) 




