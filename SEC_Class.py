import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandarallel
from scipy.optimize import curve_fit
from scipy import signal
from scipy.signal import sawtooth
import os
from typing import Optional, Dict
import numpy as np
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import cmcrameri.cm as cmc
import matplotlib as mpl
from scipy import signal
from scipy.signal import sawtooth


# to do - flip the plotting order in co-plot mode for the cathodic scan
# flake8: noqa

class SpEC:
    def __init__(
        self,
        Wavelength: Optional[np.ndarray] = None,
        Andorspec: Optional[pd.DataFrame] = None,
        Andorspec_calibrated: Optional[pd.DataFrame] = None,
        CV: Optional[pd.DataFrame] = None,
        interpolation: Optional[tuple] = None,
        meta_data: Optional[Dict] = None,
        spec_scans: Optional[Dict] = None,
        spec_scans_downsampled: Optional[Dict] = None,
        CV_scans: Optional[Dict] = None,
    ):
        self.Wavelength = Wavelength if Wavelength is not None else np.array([])
        self.Andorspec = Andorspec if Andorspec is not None else pd.DataFrame()
        self.Andorspec_calibrated = (
            Andorspec_calibrated if Andorspec_calibrated is not None else pd.DataFrame()
        )
        self.CV = CV if CV is not None else pd.DataFrame()
        self.interpolation = interpolation if interpolation is not None else {}
        self.meta_data = meta_data if meta_data is not None else {}
        self.spec_scans = spec_scans if spec_scans is not None else {}
        self.CV_scans = CV_scans if CV_scans is not None else {}
        self.spec_scans_downsampled = (
            spec_scans_downsampled if spec_scans_downsampled is not None else {}
        )

    def read_CV(self, CV_file_path: Path):
        self.CV = pd.read_csv(CV_file_path)
        # the first collum from the CV is not meaningful drop it
        self.CV = self.CV.drop(self.CV.columns[0], axis=1)
        # make the 'Cycl' collumn of type int
        try:
            self.CV["Cycle"] = self.CV["Cycle"].astype(int)
        except ValueError as e:
            print('The CV file does not have a "Cycle" column, please check this data')
            raise e

        return (self.CV,)

    def generate_interpolation_function(
        self, startin_samp: float = 1, starting_phase: float = 0, starting_offset: float = 0
    ):
        """
        This function reads the CV attribute of self. It reads the collumns
        't_s'	and 'Ewe_V' and fits these functions to Sawtooth2. It populates the
        interpolation attribute of self with the interpolation function. This can
        then be used to convert time to voltage in spectral data. If the fit is poor
        the user can adjust the starting amplitude and phase, most likley the issue
        is the amplitude - change this from -1 to 1
        """

        # catch if the CV attribute is empty
        if self.CV is None or self.CV.empty:
            print("The CV attribute is empty, please read the CV data first")
            return

        # drop any rows with NaN values
        self.CV = self.CV.dropna()
        # call the CV attribute and group by ["Cycle"]. Select cycle 0
        CV = self.CV

        # extract the time and voltage data from the collumns 't_s' and 'Ewe_V'
        # as x_data and y_data respectively
        x_data = CV["t_s"]
        y_data = CV["Ewe_V"]
        max_cycles = int(CV["Cycle"].max())
        print(max_cycles)
        # Initial guess for the parameters [amplitude, phase], period is the max time
        initial_guess = [startin_samp, x_data.max() / max_cycles, starting_phase, starting_offset]

        # Fit the data to the custom sawtooth function
        popt, pcov = curve_fit(
            sawtooth2,
            x_data,
            y_data,
            p0=initial_guess,
            method="dogbox",
            maxfev=100000,  # Increase the number of iterations
        )
        # Extract the optimal parameters
        amplitude_fit, period_fit, phase_fit, offset_fit = popt

        # Print the fitted parameters
        print(
            f"""Fitted parameters saved to the interpolation attribute are:
        Amplitude:{amplitude_fit}
        Period:{period_fit}
        Phase: {phase_fit} 
        Offset: {offset_fit}
        by passing (time_array, fitted parametes) you can interpolate t-->V
        NOTE this function is only as accurate as the CV's sampling rate
        by rounding the values to the parameters used you will get the best result 
        """
        )

        # Generate fitted data
        y_fit = sawtooth2(x_data, amplitude_fit, period_fit, phase_fit, offset_fit)

        # Plot the original data and the fitted data
        fig, ax = plt.subplots()
        ax.plot(x_data, y_data, label="Original data")
        ax.plot(x_data, y_fit, color="red", linestyle="--", label="Fitted data")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (V)")
        # add a title of fitted vs measured time/ voltage
        plt.title("Interpolation function used to covert t to V")

        # add ledgend and place it on outside of plot on the right
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()

        self.interpolation = (amplitude_fit, period_fit, phase_fit, offset_fit)
        return self.interpolation

    def read_Andorspec(self, Andorspec_file_path: Path):
        """This function reads the Andorspec pkl file
        as a datadrame. As the spectral times overrun the CV times
        this function removes the spectral data that is not in the CV
        This dataframe then populates the Andorspec attribute of self
        """
        data = pd.read_pickle(Andorspec_file_path)
        # remove the spectral data that is not in the CV
        data_trimmed = data[data.index <= self.CV["t_s"].max()]
        # name the index of the Andorspec dataframe as 'Time_s'
        data_trimmed.index.name = "Time (s)"
        # Name the collumns of the Andorspec dataframe as 'Wavelength'
        data_trimmed.columns.name = "Wavelength (nm)"

        self.Andorspec = data_trimmed
        return self.Andorspec

    def Calibrate_Andorspec(self):
        """This function combines the cycle_calculator and scan_direction_calculator code. It takes in a SpEC object
        it returns the Andorspec_calibrated attribute of the SpEC object. This is calculated by first trimming the
        Andorspec attribute of the SpEC object to the minimum time value of the CV attribute. The function then
        calculates the cycle and scan direction of the spectral data. These are added as collumns to the Andorspec_calibrated object
        """
        Andoor_spec_calibrated = self.Andorspec.copy(deep=True)

        # we remove any values where the camera reads earlier than the potentostat or later than the potentostat
        # the former can happen because the gamry appears to send the trigger signal before it starts sending data
        # and we don't know what it does in this time. The latter can happen because the gamry can end before
        # async's poll can send the stop command
        Andoor_spec_calibrated = Andoor_spec_calibrated[
            (Andoor_spec_calibrated.index >= self.CV["t_s"].min())
            & (Andoor_spec_calibrated.index <= self.CV["t_s"].max())
        ]

        # there is also the case where the CV goes on for longer than the camera. We need to remove these values
        # because they produce errors in the cycle calculation later
        CV = self.CV.copy(deep=True)
        # CV = self.CV[self.CV['t_s'] <= Andoor_spec_calibrated.index.max()]
        time_array = Andoor_spec_calibrated.index.values
        # create an empty array of the same length as the time array and fill it with NaN

        cycle = np.full(len(time_array), np.nan)

        # group the CV by cycle
        grouped = CV.groupby(["Cycle"])
        # iterate through each cycle

        for i, group in grouped:
            # get the maximum and minimum time values of the cycle
            max_time = group["t_s"].max()
            min_time = group["t_s"].min()
            # print(f'max time is {max_time}')
            # on the first cycle set the cycle number only as time_array <= max_time
            if i[0] == 0:
                cycle[time_array <= max_time] = i[0]

            elif i[0] != 0:
                # set the cycle value of the time array to the cycle number
                cycle[(time_array >= min_time) & (time_array <= max_time)] = i[0]

        # add cycle as a new collumn to the Andorspec_calibrated dataframe
        Andoor_spec_calibrated.insert(0, "Cycle", cycle)

        # calculate the derivative of the sawtooth2 function
        deriv = np.diff(sawtooth2(time_array, *self.interpolation)) > 0
        # insert the first value of the derivative to the start of the array because np.diff reduces the length by 1
        deriv = np.insert(deriv, 0, deriv[0])
        plt.plot(time_array, deriv)
        # set the x label to Time (s)

        plt.xlabel("Time (s)")
        plt.ylabel("Predicted scan Direction, 1=anodic scan")
        plt.title("Predicted scan direction from the fitting of U vs t")
        # Initialize scan_direction as an array of strings instead of zeros
        scan_direction = np.full(len(time_array), "", dtype=object)

        # Set the scan direction to 'anodic' if the derivative is greater than zero
        scan_direction[deriv] = "anodic"
        # Set the scan direction to 'cathodic' if the derivative is less than zero
        scan_direction[~deriv] = "cathodic"

        # add scan_direction as a new collumn to the Andorspec_calibrated dataframe
        Andoor_spec_calibrated.insert(1, "Scan Direction", scan_direction)

        # Finally use the index of of Andoor_spec_calibrated to apply the interpolation function to the Andorspec_calibrated
        U = sawtooth2(Andoor_spec_calibrated.index.values, *self.interpolation)
        # insert the voltage as a new collumn to the Andorspec_calibrated dataframe
        Andoor_spec_calibrated.insert(2, "Voltage (V)", U)
        self.Andorspec_calibrated = Andoor_spec_calibrated
        self.CV = CV

        # finally there can be a case where the
        return self.Andorspec_calibrated, self.CV

    def populate_spec_scans(self):
        """This function reads the Andorspec_calibrated attribute of the SpEC object. It groups the data by cycle
        and then by scan direction. It then populates the spec_scans attribute of the SpEC object with a dictionary of dictionaries.
        The first key is the cycle number, the second key is the scan direction. The value is the dataframe of the spectral data
        for that cycle and scan direction. You may need to correct for anomalies in the interpolation function using the
        remove_scan_outliers function"""

        cycle_dict = {}
        for i in range(int(self.Andorspec_calibrated["Cycle"].max() + 1)):
            try:
                temp = self.Andorspec_calibrated.groupby(["Cycle"]).get_group((i,))
            except Exception as e:
                print(f"no data in cycle number {i}, {e} scan data set to None")
                temp = {}
                continue
            try:
                Anodic = (
                    temp.groupby(["Scan Direction"])
                    .get_group(("anodic",))
                    .drop("Scan Direction", axis=1)
                    .drop("Cycle", axis=1)
                )
            except Exception as e:
                Anodic = None
                print(f"no anodic data in scan number {i}, {e} scan data set to None")
                continue
            try:
                Cathodic = (
                    temp.groupby(["Scan Direction"])
                    .get_group(("cathodic",))
                    .drop("Scan Direction", axis=1)
                    .drop("Cycle", axis=1)
                )
            except Exception as e:
                print(f"no cathodic data in scan number {i}, {e} scan data set to None")
                Cathodic = None
                continue
            cycle_dict[i] = {"Anodic": Anodic, "Cathodic": Cathodic}
        self.spec_scans = cycle_dict
        return self.spec_scans

    def populate_CV_scans(self):
        """like populate spec scans this function uses the derivative of the interpolation function to determine the scan direction
        of the CV data. It then groups the CV data by cycle and scan direction. It then populates the CV_scans attribute of the SpEC object
        with a dictionary of dictionaries. The first key is the cycle number, the second key is the scan direction. The value is the dataframe
        of the CV data for that cycle and scan direction"""

        cycle_dict = {}

        for i in range(int(self.CV["Cycle"].max())):
            try:
                temp = self.CV.groupby(["Cycle"]).get_group((i,))
            except Exception as e:
                print(f"no data in cycle number {i}, {e} scan data set to None")
                temp = {}
                continue

            try:
                deriv = np.diff(sawtooth2(temp["t_s"], *self.interpolation)) > 0
                deriv = np.insert(deriv, 0, deriv[0])
                # Initialize scan_direction as an array of strings instead of zeros
                scan_direction = np.full(len(temp["t_s"]), "", dtype=object)

                # Set the scan direction to 'anodic' if the derivative is greater than zero
                scan_direction[deriv] = "anodic"
                # Set the scan direction to 'cathodic' if the derivative is less than zero
                scan_direction[~deriv] = "cathodic"

                temp.insert(0, "Scan Direction", scan_direction)

            except Exception as e:
                print(
                    f"No time was found in the data of this cycle: {e}. This meant no scan direction could be calculated"
                )
                return

            try:
                Anodic = (
                    temp.groupby(["Scan Direction"])
                    .get_group(("anodic",))
                    .drop("Scan Direction", axis=1)
                    .drop("Cycle", axis=1)
                )
            except Exception as e:
                Anodic = None
                print(f"no anodic data in scan number {i}, {e} scan data set to None")
            try:
                Cathodic = (
                    temp.groupby(["Scan Direction"])
                    .get_group(("cathodic",))
                    .drop("Scan Direction", axis=1)
                    .drop("Cycle", axis=1)
                )
            except Exception as e:
                print(f"no cathodic data in scan number {i}, {e} scan data set to None")
                Cathodic = None
            cycle_dict[i] = {"Anodic": Anodic, "Cathodic": Cathodic}
        self.CV_scans = cycle_dict
        return self.CV_scans

    def Downsample_spec_scans(
        self,
        voltage_resolution: float,
        Wavelength_resolution: int,
        show_times: bool = False,
        transpose: bool = True,
    ):
        """This function takes in the spec_scans attribute of the SpEC object. It loops through the dictionary of dictionaries
        and calls the Downsample_Potential and Downsample_Wavelength functions on each dataframe in the dictionary. The downsampled
        dataframes are then stored in the spec_scans_downsampled attribute of the SpEC object as a dictionary of dictionaries like the
        spec_scans attribute"""

        cycle_dict = {}
        for key, value in self.spec_scans.items():
            temp_dict = {}
            for key2, value2 in value.items():
                if key2 == "Anodic" and value2 is not None:
                    temp = Downsample_Wavelength(value2, Wavelength_resolution)
                    temp_dict["Anodic"] = Downsample_Potential(temp, voltage_resolution)
                    temp_dict["Anodic"].columns.name = "Wavelength (nm)"
                    temp_dict["Anodic"].index.name = "Voltage (V)"
                    temp = None
                    if not show_times:
                        # pop the Time (s) collumn
                        temp_dict["Anodic"].pop("Time (s)")

                    if transpose:
                        temp_dict["Anodic"] = temp_dict["Anodic"].T

                elif key2 == "Cathodic" and value2 is not None:
                    temp = Downsample_Wavelength(value2, Wavelength_resolution)
                    temp_dict["Cathodic"] = Downsample_Potential(
                        temp, voltage_resolution
                    )
                    temp_dict["Cathodic"].columns.name = "Wavelength (nm)"
                    temp_dict["Cathodic"].index.name = "Voltage (V)"
                    temp = None

                    if not show_times:
                        temp_dict["Cathodic"].pop("Time (s)")

                    if transpose:
                        temp_dict["Cathodic"] = temp_dict["Cathodic"].T

            cycle_dict[key] = temp_dict
        self.spec_scans_downsampled = cycle_dict

        return self.spec_scans_downsampled


def select_file_path():
    """This is the base function to select a file, it opens a prompt and returns
    a path obect.

    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = (
        filedialog.askopenfilename()
    )  # Show the file dialog and get the selected file path
    root.destroy()  # Close the root window
    # convert the file path into a raw string so spaces are not escaped]


    return Path(file_path)


def change_directory_to_new_expt():
    "If you keep filenames constant and specifically name folders you can set the\
        directory to the new folder by selecting a file in that folder"
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = Path(
        filedialog.askopenfilename()
    ).parent  # Show the file dialog and get the selected file path
    root.destroy()  # Close the root window
    return os.chdir(file_path)


def sawtooth2(time, amplitude, period, phase, offset):
    """This function generates a sawtooth wave with the following parameters:
    Once, fitted is used to generate an interpolation function from t-->V.
    time: time array
    amplitude: amplitude of the wave
    period: period of the wave
    phase: phase of the wave
    """
    return (amplitude * sawtooth((2 * np.pi * time) / (period) - phase, 0.5) + offset)


def Downsample_Potential(SpEC_scans_dataframe, voltage_resolution: float):
    """This function takes in the a single dataframe from spec_cycles. It averages rows that are within the voltage resolution of one another
    The function returns a downsampled dataframe"""

    all_spectra = SpEC_scans_dataframe

    # extract the times from the index

    times = all_spectra.index.values.astype(float)

    # insert this into the 0th collumn of the dataframe

    all_spectra.insert(0, "Time (s)", times)

    all_spectra = all_spectra.dropna()

    # create a array with the same number of value as the voltage (V) collumn
    # this array will be used to group the data by voltage resolution

    voltage_grouping = all_spectra["Voltage (V)"].copy(deep=True)

    # round the voltage grouping array to the nearest voltage resolution

    voltage_grouping = (
        np.round(voltage_grouping / voltage_resolution) * voltage_resolution
    )

    # replace the voltage collumn with the rounded voltage grouping array

    all_spectra["Voltage (V)"] = voltage_grouping

    # perform the mean downsample by grouping the data by the voltage collumn and taking the mean

    all_spectra = all_spectra.groupby("Voltage (V)").mean()

    return all_spectra


def Downsample_Wavelength(SpEC_scans_dataframe, Wavelength_resolution: int):
    """This function takes in the a single dataframe from spec_cycles. It averages rows that are within the voltage resolution of one another
    The function returns a downsampled dataframe"""

    all_spectra = SpEC_scans_dataframe.copy(deep=True)

    all_spectra.columns.name = None

    # temporarilty pop the voltage collumn to perform the downsample

    voltages = all_spectra.pop("Voltage (V)")

    wavelengts = all_spectra.columns.values

    # convert all wavelengths values to floats

    wavelengts = wavelengts.astype(float)

    wavelength_grouping = (
        np.round(wavelengts / Wavelength_resolution) * Wavelength_resolution
    )

    # set the values of the collumn names to wavelength_grouping

    all_spectra.columns = wavelength_grouping

    all_spectra = all_spectra.T.groupby(wavelength_grouping).mean().T
    # set the type of the collumn names to int

    all_spectra.columns = all_spectra.columns.astype(int)

    # add the voltage collumn back to the dataframe

    all_spectra.insert(0, "Voltage (V)", voltages)

    return all_spectra


def calculateDOD(
    SpEC_object: SpEC,
    cycle_number: int,
    scan_direction: str,
    Referance_potential: float,
    smooth_strength: int = 0,
):
    """This function calculates Delta OD for a single cycle and scan direction. If the referance potential
    given is not present the nearest potential is used. The function returns a dataframe. If smooth_strength is set to 0
    the function returns the raw data. If smooth_strength is an odd integer the function returns the data smoothed by a golay function
    """
    if scan_direction not in ["Anodic", "Cathodic"]:
        print('scan_direction must be either "Anodic" or "Cathodic"')
        return

    # extract the spectral data for the cycle and scan direction

    data = SpEC_object.spec_scans_downsampled[cycle_number][scan_direction]

    # extract the voltages - which are the collumn names
    voltages = data.columns.values
    # print(voltages)

    # find the nearest potential to the referance potential

    nearest_potential_index = np.argsort(np.abs((voltages - Referance_potential)))[0]

    # extract the data at the nearest potential

    I0 = data.iloc[:, nearest_potential_index]
    LnI0 = np.log10(I0)
    LnI = pd.DataFrame(np.log10(data))
    # print(LnI.shape)

    DOD = -1 * LnI.subtract(LnI0, axis=0)

    if smooth_strength != 0:
        DOD = DOD.apply(lambda x: signal.savgol_filter(x, smooth_strength, 3), axis=0)

    return DOD, Referance_potential


def plot_DOD(
    DOD_dataframe: pd.DataFrame,
    Title: Optional[str] = None,
    y_max: Optional[float] = None,
    y_min: Optional[float] = None,
    x_max: Optional[float] = None,
    x_min: Optional[float] = None,
    reference_potential: Optional[str] = None,
):
    """This function takes in a DOD dataframe and plots it. The function returns a plot"""
    # get the number of collumns in DOD
    n = DOD_dataframe.shape[1]
    # get the color map

    cmap = cmc.roma

    colors = cmap(np.linspace(0, 1, n))
    # remove the first 100 rows
    colors = np.linspace(0, 1, n)
    colors = cmc.roma(colors)

    fig, ax = plt.subplots()
    for i in range(n):
        ax.plot(
            DOD_dataframe.index, DOD_dataframe.iloc[:, i], color=colors[i], linewidth=2
        )

    v_min = DOD_dataframe.columns.min()
    v_max = DOD_dataframe.columns.max()
    # Normalize the color map
    norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Add the colorbar to the figure
    fig.colorbar(sm, ax=ax, label="$U (V) $")

    plt.xlabel("Wavelength (nm)", fontsize=21)
    # set the y label to f"$\Delta$A (O.D. vs {reference_potential} V)" if reference_potential is not None else f"$\Delta$A (O.D.)"
    plt.ylabel(
        (
            f"$\Delta$A (O.D. vs {reference_potential})"
            if reference_potential is not None
            else f"$\Delta$A (O.D.)"
        ),
        fontsize=18,
    )

    # set the axis font size to 20
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    if Title is not None:
        plt.title(Title, fontsize=21)
    if y_max is not None and y_min is not None:
        plt.ylim(top=y_max, bottom=y_min)

    if x_min is not None and x_max is not None:
        plt.xlim(left=x_min, right=x_max)

    return fig, ax


def Co_plot_DOD_and_CV(
    DOD_dataframe: pd.DataFrame,
    CV_dataframe: pd.DataFrame,
    Title: Optional[str] = None,
    y_max: Optional[float] = None,
    y_min: Optional[float] = None,
    x_max: Optional[float] = None,
    x_min: Optional[float] = None,
    reference_potential: Optional[str] = None,
    scan_direction: Optional[str] = None,
    cmap_option = cmc.roma,
):
    """This function takes in a DOD dataframe and plots it. The function returns a plot"""
    # get the number of collumns in DOD
    n = DOD_dataframe.shape[1]
    # get the color map

    cmap = cmap_option

    colors = cmap(np.linspace(0, 1, n))
    # remove the first 100 rows
    colors = np.linspace(0, 1, n)
    colors = cmap_option(colors)

    fig, ax = plt.subplots(2, 1)
    for i in range(n):
        ax[0].plot(
            DOD_dataframe.index, DOD_dataframe.iloc[:, i], color=colors[i], linewidth=2
        )

    v_min = DOD_dataframe.columns.min()
    v_max = DOD_dataframe.columns.max()
    # Normalize the color map
    norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Add the colorbar to the figure
    fig.colorbar(sm, ax=ax[0], label="$U (V) $")

    ax[0].set_xlabel("Wavelength (nm)", fontsize=12)
    # set the y label to f"$\Delta$A (O.D. vs {reference_potential} V)" if reference_potential is not None else f"$\Delta$A (O.D.)"
    ax[0].set_ylabel(
        (
            f"$\Delta$A (O.D. vs {reference_potential})"
            if reference_potential is not None
            else f"$\Delta$A (O.D.)"
        ),
        fontsize=12,
    )

    # set the axis font size to 20
    # ax[0].set_xticks(fontsize=18)
    # ax[0].set_yticks(fontsize=18)
    # set the axis fontsixe

    if Title is not None:
        fig.suptitle(Title, fontsize=21)
    if y_max is not None and y_min is not None:
        ax[0].set_ylim(top=y_max, bottom=y_min)

    if x_min is not None and x_max is not None:
        ax[0].set_xlim(left=x_min, right=x_max)

    # ax[1].plot(CV_dataframe['Ewe_V'], CV_dataframe['I_A'])

    # extract the voltage and current data from the CV dataframe as a new dataframe
    CV = CV_dataframe[["Ewe_V", "I_A"]]
    num_points = CV.shape[0]  # Assuming CV is a list or array of points
    cmap = cmap_option

    colors = cmap(np.linspace(0, 1, num_points))
 

    print(CV.shape)
    for i in range(num_points - 2):
        if scan_direction is not None and scan_direction == "Anodic":
            ax[1].plot(CV.iloc[i : i + 2, 0], CV.iloc[i : i + 2, 1], color=colors[i])
        elif scan_direction is not None and scan_direction == "Cathodic":
            ax[1].plot(CV.iloc[i : i + 2, 0], CV.iloc[i : i + 2, 1], color=colors[-i])

    ax[1].set_xlabel("U (V)", fontsize=12)

    ax[1].set_ylabel("I (A)", fontsize=12)
    if scan_direction is not None and scan_direction == "Anodic":
        ax[1].annotate(
            "Scan direction",
            xy=(0.5, 1.08),
            xytext=(0.3, 1.08),
            arrowprops=dict(facecolor="black", arrowstyle="->"),
            ha="center",
            va="center",
            fontsize=16,
            xycoords="axes fraction",
            textcoords="axes fraction",
        )
    if scan_direction is not None and scan_direction == "Cathodic":
        ax[1].annotate(
            "Scan direction",
            xy=(0.3, 1.08),
            xytext=(0.5, 1.08),
            arrowprops=dict(facecolor="black", arrowstyle="->"),
            ha="center",
            va="center",
            fontsize=16,
            xycoords="axes fraction",
            textcoords="axes fraction",
        )

    # ax[1].xticks(fontsize=18)

    # ax[1].yticks(fontsize=18)

    # use tight layout to prevent overlap of the two plots
    plt.tight_layout()
    plt.show()



def normalise_DOD(DOD_dataframe: pd.DataFrame, by_max: bool = True):
    """This function takes in a DOD dataframe and normalises it to the maximum value of each collumn. The function returns a normalised DOD dataframe"""
    # write a lambda fucnction that normalises each collumn of a dataframe by the maximum value of that collumn
    if by_max:
        normalise = lambda x: x / x.max()
    else:
        normalise = lambda x: x / x.min()
    # apply the normalise function to the DOD dataframe
    DOD_normalised = DOD_dataframe.apply(normalise, axis=0)

    return DOD_normalised


if __name__ == "__main__":
    AndorPath = select_file_path()
    CV_path = select_file_path()
    print(AndorPath, "\n", CV_path)

    SpEC1 = SpEC()
    SpEC1.read_CV(CV_path)
    SpEC1.generate_interpolation_function()
    SpEC1.read_Andorspec(AndorPath)
    SpEC1.Calibrate_Andorspec()
    SpEC1.populate_spec_scans()
    SpEC1.populate_CV_scans()
    SpEC1.Downsample_spec_scans(0.005, 1)
    test, ref = calculateDOD(SpEC1, 0, "Cathodic", -1, 11)

    for key, value in SpEC1.CV_scans.items():
        for key2, value2 in value.items():
            CV = SpEC1.CV_scans[key][key2]
            if key2 == "Anodic":
                test, ref = calculateDOD(SpEC1, key, key2, -0.5, 51)
            elif key2 == "Cathodic":
                test, ref = calculateDOD(SpEC1, key, key2, 0.5, 51)
            Co_plot_DOD_and_CV(
                test,
                CV,
                Title=f"Cycle {key} {key2}",
                y_max=0.001,
                y_min=-0.001,
                x_min=400,
                x_max=800,
                reference_potential=f"$ {ref}V_{{Ag/AgCl}}$",
                scan_direction=key2,
            )

""" for key, value in SpEC1.spec_scans_downsampled.items():
        for key2, value2 in value.items():
            temp, ref =calculateDOD(SpEC1, key, key2, 1, 11)
            _,_= plot_DOD(temp, Title=f'Cycle {key} {key2}', y_max=0.1, y_min=-0.1, x_min=400, x_max=800, reference_potential=ref)"""
