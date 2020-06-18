from setuptools import setup


def readme():
    with open('README.MD') as f:
        return f.read()


setup(name='pyKneeSPM',
      version='0.0.1',
      description='SPM style analysis for bone meshes',
      long_description=readme(),
      url='',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        # 'Programming Language :: Python :: 3.4',
        # 'Programming Language :: Python :: 3.5',
        # 'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
      ],
      keywords='spm, mesh, statistics, polydata',
      author='Anthony Gatti',
      author_email='anthony@neuralseg.com',
      license='MIT',
      packages=['pyKneeSPM'],
      package_data={'pyKneeSPM': ['data/*.vtk']},
      install_requires=['vtk', 'numpy', 'scipy'],
      zip_safe=False)
