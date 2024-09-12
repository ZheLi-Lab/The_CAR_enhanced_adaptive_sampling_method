from setuptools import setup, find_packages

setup(
    name='car-fep',
    version='1.0.5',
    #packages=find_packages(include=['car.main.*', 'utils.*','utils.*.*']),
    packages=find_packages(),
    description=(
        'Convergence-adaptive roundtrip enhanced sampling toolkits'
    ),
    author="Zhe Li, Yufen Yao, Runduo Liu",
    author_email="lizhe5@mail.sysu.edu.cn",
    platforms=["linux"],
    url="https://github.com/ZheLi-Lab/The_CAR_enhanced_adaptive_sampling_method",
    entry_points={
        'console_scripts': [
            'run-car = car.main:main'
        ]
    },
    package_data={
        'car': ['car/example']
    },
    python_requires = '>=3.9,<3.11',
    install_requires=[
        'numpy==1.22.4',
        'pandas==1.4.4',
        'parse==1.19',
        'matplotlib==3.6',
        'seaborn==0.12.0',
        'openpyxl==3.0.10',
        'scipy==1.9.1',
        'pymbar==3.1',
        'alchemlyb==1.0',
        'jnija2==3.1',
    ],
    
)
