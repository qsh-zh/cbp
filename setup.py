import re
from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the version string
with open(path.join(here, 'cbp', '__init__.py')) as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)

setup(
    name='pycbp',
    version=version,
    description='A Library for Constrained Belief Propagation',
    long_description=open('README.md', encoding='utf8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/zqsh/cbp',
    author='Qinsheng',
    author_email='qsh.zh27@gmail.com',
    license='MIT',
    python_requires='>=3.7',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='belief propagation bayesian inference',
    packages=find_packages(exclude=['test', 'test.*',
                                    'examples', 'examples.*',
                                    'docs', 'docs.*']),
    install_requires=[
        'numpy',
        'tqdm',
        'scipy',
        'matplotlib',
        'seaborn',
        'pygraphviz',
        'numba',
        'hmmlearn'
    ],
    extras_require={
        'dev': [
            'Sphinx',
            'sphinx_rtd_theme',
            'nbsphinx',
            'sphinxcontrib-bibtex',
            'pylint',
            'pytest',
            'pytest-cov',
        ]
    },
)
