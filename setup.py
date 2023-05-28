from setuptools import setup, find_packages

setup(
    name="maestro",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        # Add your project's dependencies here, e.g.,
        'kiteconnect',
        'python-dotenv'
    ],
    extras_require={
        # Add optional dependencies and their versions here, e.g.,
        # 'dev': ['flake8', 'pytest'],
    },
    classifiers=[
        # Add classifiers that apply to your project, e.g.,
        # 'Development Status :: 3 - Alpha',
        # 'Intended Audience :: Developers',
        # 'License :: OSI Approved :: MIT License',
        # 'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.8',
    ],
    author="Akash Anand",
    author_email="akashanand.iitd@gmail.com",
    description="The maestro package",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MarketMaestro/maestro",
    license="MIT",
)