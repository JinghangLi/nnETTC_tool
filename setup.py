from setuptools import setup, find_packages

setup(
    name='nnETTC_tool',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'opencv-python',
        'pyyaml',
        'matplotlib',
        'Pillow'
    ],
    description='A simple Python package example',
    author='JinghangLi',
    author_email='jhanglee.ro@gmail.com',
    url='https://github.com/yourusername/my_package',  # GitHub 仓库地址（可选）
)