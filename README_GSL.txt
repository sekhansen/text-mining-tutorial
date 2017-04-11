This README describes the step by step installation to incorporate GSL into 
Stephen Hansen's Topic Modelling Python package

## OSX-64, Linux-64 or Linux-32

Step 1: Make sure you have GCC installed on your machine.

sudo apt-get install build-essential

Step 2: Install GSL on your machine.

Follow Aaron Meurer's (https://binstar.org/asmeurer) Conda installation for 
GSL files  by typing the following line in your terminal:

conda install -c https://conda.binstar.org/asmeurer gsl

The necessary files of GSL will now be installed in /home/usr/anaconda/lib and
/home/usr/anaconda/include.

Step 3: Download the Topic Modelling files from Stephen Hansen's github repo:

https://github.com/sekhansen/text-mining-tutorial

Step 4: Uncomment the hash-tagged sections of the setup.py folder such that the
file now has the following content (Note the change to the "ext" variable).

---------------------------

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import sys

if sys.platform == "win32":
	include_gsl_dir = sys.exec_prefix.lower().split("anaconda2")[0]+"anaconda\\gsl\\include"
	lib_gsl_dir = sys.exec_prefix.lower().split("anaconda2")[0]+"anaconda\\gsl\\lib"
else:
	include_gsl_dir = sys.exec_prefix+"\\include"
	lib_gsl_dir = sys.exec_prefix+"\\lib"
ext = Extension("samplers", ["samplers.pyx"],
	include_dirs=[numpy.get_include(),
	include_gsl_dir],
	library_dirs=[lib_gsl_dir],
	libraries=["gsl","gslcblas","m"]
)

setup(name = "samplers",
	ext_modules=[ext],
	cmdclass = {'build_ext': build_ext})
	

---------------------------

Step 5: CD in your Terminal into the folder with the GitHub files and the
edited setup.py file. 

Step 6: Execute the following commands in your Terminal:

python setup.py build_ext --inplace
python setup.py install 

Step 7: Ensure that samplers.so is now in /home/usr/anaconda/lib/site-packages.

The samplers module is now installed on your machine with GSL

## Windows-32 or Windows-64

Step 1: Install GSL on your machine.

Download the .zip from the following link: 

https://code.google.com/p/oscats/downloads/detail?name=gsl-1.15-dev-win64.zip&can=2&q=

Step 2: Extract the zip into your Anaconda folder (usually C:\Users\YourName\Anaconda)
with the folder name "gsl" such that the structure is as follows:

\Anaconda
	\gsl
		\include
		\lib
		\bin
		\share
	\Lib
	\pkgs
	...

Step 3: Add the "...\Anaconda\gsl\bin\" location to your system wide PATH variable.

Control Panel > System Security > System > Advanced System Settings > 
Environment Variables > System Variables, and editing PATH to include 
"C:\Users\UserName\Anaconda\gsl\bin"

Step 4: Download the Topic Modelling files from Stephen Hansen's github repo:

https://github.com/sekhansen/text-mining-tutorial

Step 5: Uncomment the hashtagged sections of the setup.py folder such that the
file now has the following content (Note the change to the "ext" variable)

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import sys

if sys.platform == "win32":
	include_gsl_dir = sys.exec_prefix.lower().split("anaconda")[0]+"anaconda\\gsl\\include"
	lib_gsl_dir = sys.exec_prefix.lower().split("anaconda")[0]+"anaconda\\gsl\\lib"
else:
	include_gsl_dir = sys.exec_prefix+"\\include"
	lib_gsl_dir = sys.exec_prefix+"\\lib"
ext = Extension("samplers", ["samplers.pyx"],
	include_dirs=[numpy.get_include(),
	include_gsl_dir],
	library_dirs=[lib_gsl_dir],
	libraries=["gsl","gslcblas","m"]
)

setup(name = "samplers",
	ext_modules=[ext],
	cmdclass = {'build_ext': build_ext})
	
Step 6: CD in your Terminal into the folder with the GitHub files and the
edited setup.py file. 

Step 7: Execute the following command in your Terminal:

python setup.py build_ext --inplace
python setup.py install 

TROUBLESHOOTING STEP 7:

To run Cython on windows, you need certain dependencies which 
may or may not be already installed on your machine.

You need Microsoft 2008 Visual Studio installed:
https://www.dreamspark.com/Product/Product.aspx?productid=34

You need the Microsoft SDK installed on your machine:
http://www.microsoft.com/en-us/download/details.aspx?id=24826

You need a GNU compiler. The most famous is GCC:
http://sourceforge.net/projects/tdm-gcc/?source=typ_redirect

Step 8: Ensure that samplers.pyd is now in /home/usr/anaconda/lib/site-packages/

The samplers module is now installed on your machine with GSL

