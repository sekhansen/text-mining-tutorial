## An Introduction to Topic Modelling via Gibbs sampling: Code and Tutorial

by Stephen Hansen, stephen.hansen@economics.ox.ac.uk

Associate Professor of Economics, University of Oxford

***

Thanks to Eric Hardy at Columbia University for collating data on speeches.

***

If you use this software in research or educational projects, please cite: 

Hansen, Stephen, Michael McMahon, and Andrea Prat (2014), “Transparency and Deliberation on the FOMC: A Computational Linguistics Approach,” CEPR Discussion Paper 9994.  

***


### INTRODUCTION

This project introduces Latent Dirichlet Allocation (LDA) to those who do not necessarily have a background in computer science or programming.  There are many implementations of LDA available online in a variety of languages, many of which are more memory and/or computationally efficient than this one.  What is much rarer than optimized code, however, is documentation and examples that allow complete novices to practice implementing topic models for themselves.  The goal of this project is to provide these, thereby reducing the startup costs involved in using topic models.

The contents of the tutorial folder are as follows:

1. topicmodels: Python/Cython code for cleaning text and estimating LDA via collapsed Gibbs sampling as in Griffiths and Steyvers (2004).
2. speech_data_extend.txt: Data on State of the Union Addresses.
3. tutorial_notebook.ipynb: iPython notebook for the tutorial.
4. tutorial.py: Python code with the key commands for the tutorial.
5. setup.py: File for compiling Cython code into C library.


### INSTALLING PYTHON

The code relies on standard scientific libraries which are not part of a basic Python installation.  For those who do not already have them installed, the recommended option is to download the Anaconda distribution of Python 2.7 from <http://continuum.io/downloads> with the default settings.  After installation, you should be able to launch iPython directly from the Anaconda folder on OS X and Windows.  On Linux you can launch it by typing “ipython” from the command line.  (iPython is an enhanced Python interpreter particularly useful for scientific computing.)

If iPython does not launch, then it may be that your anti-virus software considers it a risk and blocks it.  For example, this may happen in some versions of Kaspersky 6.0 which, on starting iPython, quarantines the python.exe file which renders other (previously working) Python operations inoperable.  One option is to turn off the anti-virus software.  Another is to prevent the specific “Application Activity Analyzer” which interprets the “Input/output redirection” of iPython notebook as a threat which leads it to quarantine the Python executable.

For background on Python and iPython from an economics perspective, see <http://quant-econ.net/>.


### BUILDING SAMPLERS

Part of the code is written in Cython, which should be compiled using the following steps:

1. For Mac OS X, download Xcode, which provides a C compiler.

2. Find your operating system’s command line (NOT the iPython interpreter).  For Windows, search cmd from the start menu.  For Mac OS X, go to Applications >> Utilities >> Terminal.

3.  Change directory to the folder containing the code and speech data by typing “cd path_to_tutorial”, where path_to_tutorial is the path to the folder.  On OS X (Windows) you can type “ls” (“dir”) on the command line to make sure all the files above are in the working directory.

4. Type “python setup.py build_ext --inplace” into the command line. On Windows, if you get the error message “unable to find vcvarsall.bat,” try specifying the compiler by using the command “python setup.py build_ext --inplace --compiler=mingw32,” or finding further information on <https://github.com/cython/cython/wiki/CythonExtensionsOnWindows>.  On OS X, you may need to run the command as an administrator by typing “sudo python setup.py build_ext --inplace” and entering your password.  If the code successfully compiles, there should be no error messages, but potentially numerous warning messages relating to using a deprecated numpy API and unused functions.  These can be safely ignored.

5. To improve performance, I have used GNU Scientific Library's random number generator instead of numpy's.  This requires installing GSL on your local machine.  Instructions for using GSL along with Anaconda are contained in the file README_GSL.txt (many thanks to Paul Soto at UPF for preparing this).  To avoid this, modify both the setup.py and samplers_lda.pyx files per the comments.


### FOLLOWING THE TUTORIAL

The tutorial can either be followed using the plain tutorial.py script; by using ipython; or by using ipython with qtconsole for enhanced graphics.  To initiate the latter, type “jupyter qtconsole” (or in older versions "ipython qtconsole")  You should make sure that your current working directory is the tutorial folder.  To check this, you can type “pwd” to see the working directory.  If you need to change it, use the cd command.  

The easiest option is to copy and paste the commands from the notebook into ipython (the notebook can be viewed on http://nbviewer.ipython.org/github/sekhansen/text-mining-tutorial/blob/master/tutorial_notebook.ipynb and is also provided for convenience).  


### PERFORMANCE

While primarily written as an introduction, the code for the project should also be suitable for analysis on datasets with at least several million words, which includes many of interest to social scientists.  For very large datasets, a more scalable solution is likely best (note that even when fully optimized, Gibbs sampling tends to be slow compared to other inference algorithms).

In terms of memory, one should keep in mind that each sample has an associated document-topic and term-topic matrix stored in the background.  For large datasets, this may become an issue when trying to store many samples concurrently. 

### FEEDBACK

Comments, bug reports, and ideas are more than welcome, particularly from those using topic modelling in economics and social science research.
