from distutils.core import setup

setup(
    name='ImageContour',
    version='0.1.2',
    author='David N. Mashburn',
    author_email='david.n.mashburn@gmail.com',
    packages=['ImageContour'],
    scripts=[],
    url='http://pypi.python.org/pypi/ImageContour/',
    license='LICENSE.txt',
    description='',
    long_description=open('README.rst').read(),
    install_requires=[
                      'numpy>=1.0',
                      'matplotlib>=1.0',
                     ],
)
