from distutils.core import setup

# Read the version number
with open("ImageContour/_version.py") as f:
    exec(f.read())

setup(
    name='ImageContour',
    version=__version__, # use the same version that's in _version.py
    author='David N. Mashburn',
    author_email='david.n.mashburn@gmail.com',
    packages=['ImageContour'],
    scripts=[],
    url='http://pypi.python.org/pypi/ImageContour/',
    license='LICENSE.txt',
    description='construct and use linear paths around simply-connected binary and greyscale images',
    long_description=open('README.rst').read(),
    install_requires=[
                      'numpy>=1.0',
                      'matplotlib>=1.0',
                      'np_utils>=0.3.1.1'
                     ],
)
