
from setuptools import setup, Extension
#	from distutils.core import setup
#from distutils.extension import Extension

import os
import sys

try:
    import numpy
    numpy_include = os.path.join(os.path.dirname(numpy.__file__),
                                 'core', 'include')
except ImportError:
    print 'numpy was not found.  Aborting build'
    sys.exit(-1)

# GEM_ARCH must be one of "DARWIN", "DARWIN64", "LINUX", "LINUX64", "WIN32", or "WIN64"
gem_arch = os.environ['GEM_ARCH']
gem_type = 'diamond'
pkg_name = 'pygem_diamond'

# These environment variables are usually set for GEM builds:
gemlib = os.path.abspath(os.path.join(os.environ['GEM_BLOC'], 'lib'))

egadsinc = os.environ['EGADSINC']
egadslib = os.environ['EGADSLIB']
caslib = os.path.join(os.environ['CASROOT'], 'lib')

print '\nMaking "gem.so" for "%s" (on %s)\n' % (gem_type, gem_arch)

gem_include_dirs = [os.path.join(os.path.dirname(gemlib),'include'), 
                    numpy_include, egadsinc]
gem_extra_compile_args = []
gem_extra_link_args = []
gem_libraries = ['gem', 'diamond', 'egads']
gem_library_dirs = [gemlib, egadslib, caslib]

if gem_arch.startswith('DARWIN'):
    lib_stuff = ["lib/*.dylib", "lib/*.so"]
    if gem_arch == "DARWIN64":
        os.environ['ARCHFLAGS'] = '-arch x86_64'
    else:
        os.environ['ARCHFLAGS'] = '-arch i386'
    gem_library_dirs.append('/usr/X11/lib')
elif gem_arch.startswith('LINUX'):
    lib_stuff = ["lib/*.so", "lib/*.so.*"]
elif gem_arch == 'WIN32':
    lib_stuff = ["lib/*.dll", "lib/*.manifest"]
elif gem_arch == 'WIN64':
    lib_stuff = ["lib/*.dll", "lib/*.manifest"]
    gem_extra_compile_args = ['-DLONGLONG']


module1 = Extension(pkg_name + '.gem',
                    include_dirs=gem_include_dirs,
                    extra_compile_args=gem_extra_compile_args,
                    library_dirs=gem_library_dirs,
                    libraries=gem_libraries,
                    extra_link_args=gem_extra_link_args,
                    sources=["pygem_diamond/gem.c"])

setup(
    name=pkg_name,
    version='0.9.10',
    description='Python interface to GEM using OpenCSM and EGADS',
    zip_safe=False,
    ext_modules = [module1],
    packages=[pkg_name],
    package_dir={'': '.'},
    include_package_data=True,
    install_requires=['pyV3D', 'openmdao.lib'],
    package_data={
        pkg_name: ['test/*.py', 'test/*.csm', 'test/*.col'] +
                    lib_stuff
    },
    entry_points = {
        "openmdao.parametric_geometry": [
            'pygem_diamond.geometry.GEMParametricGeometry = pygem_diamond.geometry:GEMParametricGeometry'
        ],
        "openmdao.binpub": [
            'pygem_diamond.geometry.GEM_Sender = pygem_diamond.geometry:GEM_Sender'
        ]
    }
)
