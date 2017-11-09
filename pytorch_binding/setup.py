# build.py
import os
import platform
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


extra_compile_args = ['-std=c++11', '-fPIC']

enable_gpu = ('CUDA_HOME' in os.environ)
if not enable_gpu:
    print("CUDA_HOME not found in the environment so building "
          "without GPU support. To build with GPU support "
          "please define the CUDA_HOME environment variable. "
          "This should be a path which contains include/cuda.h")

if platform.system() == 'Darwin':
    lib_ext = ".dylib"
    extra_compile_args.extend(['-DAPPLE', '-stdlib=libc++', '-mmacosx-version-min=10.8'])
else:
    lib_ext = ".so"
    extra_compile_args.append('-fopenmp')

if enable_gpu:
    extra_compile_args += ['-DWARPCTC_ENABLE_GPU']


ext_modules = [
    Extension(
        'warpctc_pytorch._warp_ctc',
        sources=[
            'src/binding.cpp',
            os.path.realpath('../src/ctc_entrypoint.cpp')
        ],
        extra_compile_args=extra_compile_args,
        language='c++'
    )
]


class BuildExt(build_ext):
    def build_extensions(self):
        import torch

        torch_dir = os.path.dirname(torch.__file__)
        include_dirs = [
            os.path.realpath('../include'),
            os.path.join(torch_dir, 'lib/include'),
            os.path.join(torch_dir, 'lib/include/TH'),
            os.path.join(torch_dir, 'lib/include/THC'),
        ]

        if enable_gpu:
            include_dirs.append(os.path.join(os.environ['CUDA_HOME'], 'include'))

        for ext in self.extensions:
            ext.include_dirs = include_dirs

        build_ext.build_extensions(self)


setup(
    name="warpctc_pytorch",
    version="0.1",
    description="PyTorch wrapper for warp-ctc",
    url="https://github.com/baidu-research/warp-ctc",
    author="Jared Casper, Sean Naren",
    author_email="jared.casper@baidu.com, sean.narenthiran@digitalreasoning.com",
    license="Apache",
    packages=["warpctc_pytorch"],
    setup_requires=['pybind11>=2.2.1', 'torch'],
    install_requires=['pybind11>=2.2.1', 'torch'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    test_suite='tests',
)
