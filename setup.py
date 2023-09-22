from setuptools import setup, find_packages

INSTALL_DEPS = ['numpy',
                'scipy',
                'cvxpy'
               ]

TEST_DEPS = ['pytest']
DEV_DEPS = []

setup(name='optimizn',
      version='0.0.8',
      author='Rohit Pandey, Akshay Sathiya',
      author_email='rohitpandey576@gmail.com, akshay.sathiya@gmail.com',
      description='Optimization problems.',
      packages=find_packages(exclude=['tests', 'Images']),
      long_description="Optimal settings such as optimal timeouts, retry thresholds, stripings required for desired results, etc.",
      zip_safe=False,
      install_requires=INSTALL_DEPS,
      include_package_data=True,
      python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, <4',
      # List additional groups of dependencies here (e.g. development
      # dependencies). You can install these using the following syntax,
      # for example:
      # $ pip install -e .[dev,test]
      extras_require={
          'dev': DEV_DEPS,
          'test': TEST_DEPS,
      },
     )
