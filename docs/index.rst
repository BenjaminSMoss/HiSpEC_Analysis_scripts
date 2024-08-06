SpEC analysis documentation
===========================
This is the documentation for the SpEC class. To install the SpEC class:

0. Git clone and pull the SpEC repository into a folder on your computer


1. create a new conda environment.


2. run conda install pip


3. From the cloned directory run pip install -r requirements.txt. This should install all packages needed. If this fails you can look at requirements.txt and manually install using pip or conda.


4. import the package into your code as normal


The package consists of two parts. Firstly, the SpEC class. This is the object that can be used to parse raw spectroscopic data into useful dataframes of spectroscopic data. Secondly the helper functions for plotting and calcualting differential aborbance from the parsed dataframes. The workflow of the package is as follows:

1. Create a new instance of a SpEC object

Then run the following

2. SpEC.read_CV(path)

3. SpEC.generate_interpolation_function()

4. SpEC.read_Andorspec(path)

5. SpEC.Calibrate_Andorspec()

6. SpEC.populate_spec_scans()

7. SpEC.populate_CV_scans()

8. SpEC.Downsample_spec_scans(U_downsample, WL_downsample)

The helper fucntions may then be called on the usefull spec_scans and CV_scans atttributes.







.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules


Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
