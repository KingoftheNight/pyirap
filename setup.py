from setuptools import setup

setup(name='pyirap',
    version='1.0.0',
    description='Intelligent protein analyze platform by RAAC-PSSM based on Python',
    url='https://gitee.com/KingoftheNight/pyirap',
    author='Liang YC',
    author_email='1694822092@qq.com',
    license='BSD 2-Clause',
    packages=['pyirap'],
    install_requires=['scikit-learn', 'skrebate', 'matplotlib', 'seaborn', 'joblib', 'numpy', 'pandas', 'tqdm'],
    entry_points={
        'console_scripts': [
        'pyirap=pyirap.Irap:Irap',
            ]
        },
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=True)
