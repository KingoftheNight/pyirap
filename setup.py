from setuptools import setup

setup(name='pyirap',
    version='1.0.0',
    description='Intelligent protein analyze platform by RAAC-PSSM based on Python',
    url='https://gitee.com/KingoftheNight/pyirap',
    author='Liang YC',
    author_email='1694822092@qq.com',
    license='BSD 2-Clause',
    packages=['pyirap'],
    install_requires=['sklearn', 'skrebate', 'matplotlib', 'seaborn', 'joblib', 'concurrent', 'collections', 'numpy', 'pandas', 'tqdm'],
    entry_points={
        'console_scripts': [
        'pyirap=pyirap.__main__',
            ]
        },
    python_requires=">=3.7",
    include_package_data=True,
    zip_safe=True)