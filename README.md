## An Introduction to Topic Modelling via Gibbs sampling: Code and Tutorial

by Stephen Hansen, stephen.hansen@economics.ox.ac.uk

Associate Professor of Economics, University of Oxford

***

Thanks to Eric Hardy at Columbia University for collating data on speeches.

***

If you use this software in research or educational projects, please cite: 

Hansen, Stephen, Michael McMahon, and Andrea Prat (2018), “Transparency and Deliberation on the FOMC: A Computational Linguistics Approach,” *Quarterly Journal of Economics*.  

<br>

### INTRODUCTION

This project introduces Latent Dirichlet Allocation (LDA) to those who do not necessarily have a background in computer science or programming.  There are many implementations of LDA available online in a variety of languages, many of which are more memory and/or computationally efficient than this one.  What is much rarer than optimized code, however, is documentation and examples that allow complete novices to practice implementing topic models for themselves.  The goal of this project is to provide these, thereby reducing the startup costs involved in using topic models.

The contents of the tutorial folder are as follows:

1. speech\_data\_extend.txt: Data on State of the Union Addresses.
2. tutorial_notebook.ipynb: iPython notebook for the tutorial.
3. tutorial.py: Python code with the key commands for the tutorial.

### INSTALLING PYTHON

The code relies on standard scientific libraries which are not part of a basic Python installation.  For those who do not already have them installed, the recommended option is to download the Anaconda distribution of Python 2.7 from <http://continuum.io/downloads> with the default settings.  After installation, you should be able to launch iPython directly from the Anaconda folder on OS X and Windows.  On Linux you can launch it by typing “ipython” from the command line.  (iPython is an enhanced Python interpreter particularly useful for scientific computing.)

If iPython does not launch, then it may be that your anti-virus software considers it a risk and blocks it.  For example, this may happen in some versions of Kaspersky 6.0 which, on starting iPython, quarantines the python.exe file which renders other (previously working) Python operations inoperable.  One option is to turn off the anti-virus software.  Another is to prevent the specific “Application Activity Analyzer” which interprets the “Input/output redirection” of iPython notebook as a threat which leads it to quarantine the Python executable.

For background on Python and iPython from an economics perspective, see <http://quant-econ.net/>.

### INSTALLING TOPICMODELS PACKAGE

In addition to common scientific libraries, the tutorial also requires installation of the topicmodels package located at <https://github.com/alan-turing-institute/topic-modelling-tools>.  If you already have Python and pip installed (for example by installing Anaconda per the instructions above), `pip install topic-modelling-tools` should work.

The only other requirement is that a C++ compiler is needed to build the Cython code. For Mac OS X you can download Xcode, while for Windows you can download the Visual Studio C++ compiler.

To improve performance, I have used GNU Scientific Library's random number generator instead of numpy's in a separate branch located at <https://github.com/alan-turing-institute/topic-modelling-tools/tree/with_gsl>.  To use this version instead of the baseline version, install topicmodels with `pip install topic-modelling-tools_gsl`.  Using this version requires GSL to be installed.  See the README for the package for further information.

### FOLLOWING THE TUTORIAL

The tutorial can either be followed using the plain tutorial.py script; by using ipython; or by using ipython with qtconsole for enhanced graphics.  To initiate the latter, type “jupyter qtconsole” (or in older versions "ipython qtconsole")  You should make sure that your current working directory is the tutorial folder.  To check this, you can type “pwd” to see the working directory.  If you need to change it, use the cd command.  

The easiest option is to copy and paste the commands from the notebook into ipython (the notebook can be viewed on <http://nbviewer.ipython.org/github/sekhansen/text-mining-tutorial/blob/master/tutorial_notebook.ipynb> and is also provided for convenience).  

### PERFORMANCE

While primarily written as an introduction, the code for the project should also be suitable for analysis on datasets with at least several million words, which includes many of interest to social scientists.  For very large datasets, a more scalable solution is likely best (note that even when fully optimized, Gibbs sampling tends to be slow compared to other inference algorithms).

In terms of memory, one should keep in mind that each sample has an associated document-topic and term-topic matrix stored in the background.  For large datasets, this may become an issue when trying to store many samples concurrently. 

### FEEDBACK

Comments, bug reports, and ideas are more than welcome, particularly from those using topic modelling in economics and social science research.
