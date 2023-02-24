from setuptools import find_packages, setup

with open('README.md', 'r') as fh:
    long_description = fh.read().strip()
    
with open('version.txt', 'r') as fh:
    version = fh.read().strip()

SETUP_ARGS = dict(
    name='translatesubs',
    version=version,
    license='Apache-2.0',
    author='Montvydas Klumbys',
    author_email='montvydas.klumbys@gmail.com',
    description='It is a tool to translate subtitles into any language, that is supported by google translator',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Montvydas/translatesubs',
    download_url=f'https://github.com/Montvydas/translatesubs/archive/v_{version}.tar.gz',
    keywords=['SUBTITLES', 'TRANSLATE'],
    packages=find_packages(),
    install_requires=[
        'pysubs2==1.6.0',
        'googletrans==3.1.0a0',
        'google-trans-new==1.1.9',
        'chardet==3.0.4',
        'requests==2.28.2',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License ',
        'Operating System :: OS Independent',
        'Topic :: Multimedia :: Video',
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'translatesubs=translatesubs.main:main'
        ]
    },
)


if __name__ == "__main__":
    setup(**SETUP_ARGS)
