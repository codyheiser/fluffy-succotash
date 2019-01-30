import os
import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
# IMPORTANT: prevent truncation of long strings by Pandas
#	useful for sequences embedded in dataframes
pd.set_option('display.max_colwidth', -1)


# convert appropriate strings ('TRUE', 'True', 'true') to boolean value (_True_)
def str_to_bool(s):
	'''
	s = string that should be a boolean value (T or F)
	output = corresponding boolean _True_ or _False_
	'''
	if s.title() == 'True':
		return True
	elif s.title() == 'False':
		return False
	else:
		raise ValueError("Cannot convert {} to a bool".format(s))


# count homopolymer repeats in sequence
def max_char_repeats(word):
    '''
    return maximum number of consecutive character repeats in string
    '''
    counts = []
    count = 1
    for a, b in zip(word, word[1:]):
        if a == b:
            count += 1
        else:
            counts.append(count)
            count = 1
    counts.append(count)
    return max(counts)


# move column to first position of Pandas dataframe
def movetofirst(df, cols):
    '''
    move a list of columns "cols" to the first position (left) of dataframe "df"
    '''
    return df[cols+list(collections.OrderedDict.fromkeys([x for x in collections.OrderedDict.fromkeys(list(ce.columns)) if x not in collections.OrderedDict.fromkeys(cols)]).keys())]


# read files into pandas dataframe
def read_default(file):
	'''
	take path to any flat file (.txt, .csv) and return pd.dataframe
	'''
	if os.path.splitext(file)[1]=='.csv':
		out = pd.read_csv(file)

	elif os.path.splitext(file)[1]=='.txt':
		out = pd.read_table(file)

	return out


# read all files of common type from a given directory
def read_all(filetype, startdir = '.', recursive = True):
	'''
	Read in all files of common type from a folder, concatenate by row into pandas dataframe

	filetype = string denoting which file to look for, including globs
	startdir = directory to start looking in
	recursive = search subdirectories?
	'''
	out = pd.dataframe() # initiate df for output
	for dirpath, dirs, files in os.walk(startdir):
		for file in files:
			contents = read_default(os.path.join(dirpath,file))
			contents['file'] = os.path.basename(file)
			out.append(contents)

	return out


# calculate confidence and prediction intervals for polynomial regression fit
def regression_intervals(x, y, p, x_range=None):
	'''
	Return prediction interval (pi) and 95% confidence interval (ci) for polynomial fit p (from np.polyfit) of data x, y

	x = original x data from fit
	y = original y data from fit
	p = parameters of fit of x, y using np.polyfit
	x_range = range of x values to return pi and ci for
	'''
	y_model = np.polyval(p, x)                             # model using the fit parameters; NOTE: parameters here are coefficients

	# Statistics
	n = y.size                                             # number of observations
	m = p.size                                             # number of parameters
	dof = n - m                                            # degrees of freedom
	t = sc.stats.t.ppf(0.975, n - m)                       # used for CI and PI bands

	# Estimates of Error in Data/Model
	resid = y - y_model
	chi2 = np.sum((resid/y_model)**2)                      # chi-squared; estimates error in data
	chi2_red = chi2/(dof)                                  # reduced chi-squared; measures goodness of fit
	s_err = np.sqrt(np.sum(resid**2)/(dof))                # standard deviation of the error

	# 100 linearly-spaced values between minimum and maximum in given x unless otherwise specified
	if x_range is None:
		x_range = np.linspace(np.min(x)-1, np.max(x)+1, 100)

	# Confidence Interval
	ci = t*s_err*np.sqrt(1+1/n+(x_range-np.mean(x))**2/np.sum((x-np.mean(x))**2))

	# Prediction Interval
	pi = t*s_err*np.sqrt(1/n + (x_range-np.mean(x))**2/np.sum((x-np.mean(x))**2))

	return ci, pi


# perform quick regression on x, y data and plot
def easy_regression(x, y, deg=1, plot_out=True):
	'''
	Perform linear or polynomial regression on x, y scatter data

	x = np.array or list of x values
	y = np.array or list of y values. len must be same as x.
	deg = degree of polynomial fit to perform, default 1 (linear)
	plot_out = return plot of results as well as equation and confidence intervals?
	'''
	p, cov = np.polyfit(x, y, deg, cov=True)              	# parameters and covariance from of the fit of 1-D polynom.

	x2 = np.linspace(np.min(x)-1, np.max(x)+1, 100)			# range over which to plot fit
	y2 = np.polyval(p, x2)									# fit of new x range
	ci, pi = regression_intervals(x, y, p, x2)				# calculate prediction and confidence intervals over x2

	if plot_out:
		fig, ax = plt.subplots(figsize=(6, 5))
		ax.fill_between(x2, y2+pi, y2-pi, color="0.5", edgecolor="", alpha=0.7, label='Prediction Interval')
		ax.plot(x2, y2-ci, "--", color="0.5", label="95% Confidence Interval")
		ax.plot(x2, y2+ci, "--", color="0.5", label=None)
		# fit
		ax.plot(x2,y2,"-", color="0.1", linewidth=1.5, alpha=0.5, label="Fit")
		# data
		ax.scatter(x, y, color="0.1",alpha=0.7, label=None)

		plt.xlim(np.min(x)-1,np.max(x)+1)
		ax.get_xaxis().set_tick_params(direction="out")
		ax.get_yaxis().set_tick_params(direction="out")
		ax.xaxis.tick_bottom()
		ax.yaxis.tick_left()
		plt.legend(loc='best')
		plt.tight_layout()
		plt.show()
		plt.close()

	return p, ci, pi
