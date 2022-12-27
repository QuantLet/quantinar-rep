from setuptools import setup, find_packages

setup(
    name='quantinar_rep',
    version='0.0.0',
    url='https://github.com/Quantinar/quantinar-rep',
    author='Bruno Spilak',
    author_email='bruno.spilak@gmail.com',
    dependency_links=[],
    python_requires='~=3.9',
    install_requires=[
        "numpy>=1.22.3",
        "pandas>=1.4.1",
        "networkx>=2.8.6"
        "matplotlib>=3.6.2",
        "scipy>=1.8"
    ],
    zip_safe=False,
    packages=find_packages()
)
