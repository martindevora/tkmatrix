import logging
import multiprocessing
import sys
import traceback
from multiprocessing import Pool

import numpy as np
import ellc
import matplotlib.pyplot as plt
from lcbuilder.helper import LcbuilderHelper
from lcbuilder.lcbuilder_class import LcBuilder
from lcbuilder.objectinfo.InputObjectInfo import InputObjectInfo
from lcbuilder.objectinfo.MissionInputObjectInfo import MissionInputObjectInfo
from lcbuilder.objectinfo.preparer.MissionLightcurveBuilder import MissionLightcurveBuilder
from lcbuilder.objectinfo.MissionObjectInfo import MissionObjectInfo
import wotan
from matplotlib.ticker import FormatStrFormatter
from foldedleastsquares import transitleastsquares, DefaultTransitTemplateGenerator
from foldedleastsquares import transit_mask, cleaned_array
import astropy.constants as ac
import astropy.units as u
import lightkurve as lk
import os
import re
import pandas as pd

from tkmatrix.inject_model import InjectModel


class MATRIX:
    """
    MATRIX: Multi-phAse Transits Recovery from Injected eXoplanets
    """
    object_info = None
    SDE_ROCHE = 2000
    lcbuilder = LcBuilder()
    MIN_SEARCH_PERIOD = 0.5

    def __init__(self, target, sectors, dir, preserve=False, star_info=None, file=None, exposure_time=None,
                 initial_mask=None, initial_transit_mask=None,
                 eleanor_corr_flux='pca_flux', outliers_sigma=None, high_rms_enabled=True, high_rms_threshold=2.5,
                 high_rms_bin_hours=4, smooth_enabled=False,
                 auto_detrend_enabled=False, auto_detrend_method="cosine", auto_detrend_ratio=0.25,
                 auto_detrend_period=None, prepare_algorithm=None, cache_dir=os.path.expanduser('~') + "/",
                 oscillation_reduction=False, oscillation_min_snr=4, oscillation_amplitude_threshold=0.001,
                 oscillation_ws_percent=0.01, oscillation_min_period=0.002, oscillation_max_period=0.2,
                 cores=multiprocessing.cpu_count() - 1
                 ):
        assert target is not None and isinstance(target, str)
        assert sectors is not None and (sectors == 'all' or isinstance(sectors, list))
        assert exposure_time is not None and isinstance(exposure_time, (int, float))
        assert initial_transit_mask is None or isinstance(initial_transit_mask, list)
        self.id = target
        self.dir = dir
        self.sectors = sectors
        self.star_info = star_info
        self.exposure_time = exposure_time
        self.preserve = preserve
        self.file = file
        self.eleanor_corr_flux = eleanor_corr_flux
        self.initial_mask = initial_mask
        self.initial_transit_mask = initial_transit_mask
        self.star_info = star_info
        self.outliers_sigma = outliers_sigma
        self.high_rms_enabled = high_rms_enabled
        self.high_rms_threshold = high_rms_threshold
        self.high_rms_bin_hours = high_rms_bin_hours
        self.smooth_enabled = smooth_enabled
        self.auto_detrend_enabled = auto_detrend_enabled
        self.auto_detrend_method = auto_detrend_method
        self.auto_detrend_ratio = auto_detrend_ratio
        self.auto_detrend_period = auto_detrend_period
        self.oscillation_reduction = oscillation_reduction
        self.oscillation_min_snr = oscillation_min_snr
        self.oscillation_amplitude_threshold = oscillation_amplitude_threshold
        self.oscillation_ws_percent = oscillation_ws_percent
        self.oscillation_min_period = oscillation_min_period
        self.oscillation_max_period = oscillation_max_period
        self.prepare_algorithm = prepare_algorithm
        self.cache_dir = cache_dir
        self.cores = cores

    def retrieve_object_data(self, inject_dir=None):
        self.object_info = self.lcbuilder.build_object_info(self.id, None, self.sectors, self.file, self.exposure_time,
                                                       None, None,
                                                       self.star_info, None,
                                                       self.eleanor_corr_flux, self.outliers_sigma,
                                                       False, self.high_rms_threshold,
                                                       self.high_rms_bin_hours, False,
                                                       False, self.auto_detrend_method,
                                                       self.auto_detrend_ratio, self.auto_detrend_period,
                                                       self.prepare_algorithm, False,
                                                       self.oscillation_min_snr, self.oscillation_amplitude_threshold,
                                                       self.oscillation_ws_percent, self.oscillation_min_period,
                                                       self.oscillation_max_period)
        if inject_dir is None:
            inject_dir = self.build_inject_dir()
        self.lc_build = self.lcbuilder.build(self.object_info, inject_dir, self.cache_dir)
        if self.star_info is None:
            self.star_info = self.lc_build.star_info
        self.ab = self.star_info.ld_coefficients
        self.mass = self.star_info.mass
        self.massmin = self.star_info.mass_min
        self.massmax = self.star_info.mass_max
        self.radius = self.star_info.radius
        self.radiusmin = self.star_info.radius_min
        self.radiusmax = self.star_info.radius_max
        # units for ellc
        self.rstar = self.star_info.radius * u.R_sun
        self.mstar = self.star_info.mass * u.M_sun
        self.mstar_min = self.star_info.mass_min * u.M_sun
        self.mstar_max = self.star_info.mass_max * u.M_sun
        self.rstar_min = self.star_info.radius_min * u.R_sun
        self.rstar_max = self.star_info.radius_max * u.R_sun
        return inject_dir

    def retrieve_object_data_for_recovery(self, inject_dir, recovery_file):
        self.__setup_logging(inject_dir)
        self.object_info = self.lcbuilder.build_object_info("", None, None, recovery_file, self.exposure_time,
                                                       self.initial_mask, self.initial_transit_mask,
                                                       self.star_info, None,
                                                       self.eleanor_corr_flux, self.outliers_sigma,
                                                       self.high_rms_enabled, self.high_rms_threshold,
                                                       self.high_rms_bin_hours, self.smooth_enabled,
                                                       self.auto_detrend_enabled, self.auto_detrend_method,
                                                       self.auto_detrend_ratio, self.auto_detrend_period,
                                                       self.prepare_algorithm, self.oscillation_reduction,
                                                       self.oscillation_min_snr, self.oscillation_amplitude_threshold,
                                                       self.oscillation_ws_percent, self.oscillation_min_period,
                                                       self.oscillation_max_period)

        if self.object_info.reduce_simple_oscillations and \
                self.object_info.oscillation_max_period < self.object_info.oscillation_min_period:
            logging.info("Stellar oscillation period has been set to empty. Defaulting to 1/3 the minimum search period")
            self.object_info.oscillation_max_period = self.MIN_SEARCH_PERIOD / 3
        self.lc_build = self.lcbuilder.build(self.object_info, inject_dir, self.cache_dir)

    def build_inject_dir(self):
        inject_dir = self.dir + "/" + self.object_info.mission_id().replace(" ", "") + "_ir/"
        index = 0
        while os.path.exists(inject_dir) or os.path.isdir(inject_dir):
            inject_dir = self.dir + "/" + self.object_info.mission_id().replace(" ", "") + "_ir_" + str(index) + "/"
            index = index + 1
        os.mkdir(inject_dir)
        self.__setup_logging(inject_dir)
        return inject_dir

    def __setup_logging(self, inject_dir):
        file_dir = inject_dir + "matrix.log"
        formatter = logging.Formatter('%(message)s')
        logger = logging.getLogger()
        while len(logger.handlers) > 0:
            logger.handlers.pop()
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.FileHandler(file_dir)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logging.info("Setup injection directory")

    def inject(self, phases, min_period, max_period, steps_period, min_radius, max_radius, steps_radius,
               period_grid_geom="lin", radius_grid_geom="lin"):
        """
        Creates the injection of all the synthetic transiting planet scenarios
        :param phases: the number of epochs
        :param min_period: minimum period for the period grid
        :param max_period: maximum period for the period grid
        :param steps_period: number of periods to inject
        :param min_radius: minimum radius for the grid
        :param max_radius: maximum radius for the grid
        :param steps_radius: number of radii to be injected
        :param period_grid_geom: [lin|log]
        :param radius_grid_geom: [lin|log]
        :return: the directory where injected files are stored
        """
        assert phases is not None and isinstance(phases, int) and phases > 0
        assert min_period is not None and isinstance(min_period, (int, float)) and min_period > 0
        assert max_period is not None and isinstance(max_period, (int, float)) and max_period > 0
        assert steps_period is not None and isinstance(steps_period, (int)) and steps_period > 0
        assert min_radius is not None and isinstance(min_radius, (int, float)) and min_radius > 0
        assert max_radius is not None and isinstance(max_radius, (int, float)) and max_radius > 0
        assert steps_radius is not None and isinstance(steps_radius, (int)) and steps_radius > 0
        assert max_period >= min_period
        assert max_radius >= min_radius
        inject_dir = self.retrieve_object_data()
        flux0 = self.lc_build.lc.flux.value
        time = self.lc_build.lc.time.value
        flux_err = self.lc_build.lc.flux_err.value
        period_grid = np.linspace(min_period, max_period, steps_period) if period_grid_geom == "lin" \
            else np.logspace(np.log10(min_period), np.log10(max_period), steps_period)
        radius_grid = np.linspace(min_radius, max_radius, steps_radius) if radius_grid_geom == "lin" \
            else np.logspace(np.log10(min_radius), np.log10(max_radius), steps_period)
        inject_models = []
        for period in period_grid:
            for t0 in np.arange(time[60], time[60] + period - 0.1, period / phases):
                for rplanet in radius_grid:
                    rplanet = np.around(rplanet, decimals=2) * u.R_earth
                    inject_models.append(InjectModel(inject_dir, time, flux0, flux_err, self.rstar, self.mstar, t0,
                                                     period, rplanet, self.exposure_time, self.ab))
        with Pool(processes=self.cores) as pool:
            pool.map(InjectModel.make_model, inject_models)
        return inject_dir

    def recovery(self, inject_dir, snr_threshold=5, detrend_ws=0,
                 transit_template='tls', run_limit=5, custom_search_algorithm=None, max_period_search=25,
                 oversampling=3):
        """
        Given the injection dir, it will iterate over all the csvs matching light curves and try the recovery of their
        transit parameters (period and epoch).
        :param inject_dir: the directory to search for light curves
        :param snr_threshold: the SNR value to break the search for each curve
        :param detrend_ws: the window size to detrend the curves
        :param transit_template: whether to use tls or bls
        :param run_limit: the number of iterations to break the search if the period and epoch are not found yet
        :param custom_search_algorithm: the user-provided search algorithm if any
        :param max_period_search: upper limit for the period search
        :param oversampling: the oversampling of the search period grid
        """
        assert detrend_ws is not None and isinstance(detrend_ws, (int, float))
        assert transit_template in ('tls', 'bls')
        assert inject_dir is not None and isinstance(inject_dir, str)
        if transit_template == 'tls':
            transit_template = 'default'
        elif transit_template == 'bls':
            transit_template = 'box'
        reports_df = pd.DataFrame(columns=['period', 'radius', 'epoch', 'duration_found', 'period_found', 'epoch_found',
                                           'found', 'snr', 'sde', 'run'])
        for file in sorted(os.listdir(inject_dir)):
            file_name_matches = re.search("P([0-9]+\\.[0-9]+)+_R([0-9]+\\.[0-9]+)_([0-9]+\\.[0-9]+)\\.csv", file)
            if file_name_matches is not None:
                try:
                    period = float(file_name_matches[1])
                    r_planet = float(file_name_matches[2])
                    epoch = float(file_name_matches[3])
                    df = pd.read_csv(inject_dir + file, float_precision='round_trip', sep=',',
                                     usecols=['#time', 'flux', 'flux_err'])
                    if len(df) == 0:
                        found = True
                        snr = 20
                        sde = self.SDE_ROCHE
                        run = 1
                        duration_found = 20
                        epoch_found = 0
                        period_found = 0
                    else:
                        self.retrieve_object_data_for_recovery(inject_dir + "/", inject_dir + file)
                        found, snr, sde, run, duration_found, period_found, epoch_found = \
                            self.__search(self.lc_build.lc.time.value, self.lc_build.lc.flux.value, self.radius, self.radiusmin,
                                          self.radiusmax, self.mass, self.massmin,
                                          self.massmax, self.ab, epoch, period, self.MIN_SEARCH_PERIOD,
                                          max_period_search, snr_threshold,
                                          transit_template, detrend_ws, self.lc_build.transits_min_count,
                                          run_limit, custom_search_algorithm, oversampling)
                    new_report = {"period": period, "radius": r_planet, "epoch": epoch, "found": found, "snr": snr,
                                  "sde": sde, "run": run, "duration_found": duration_found,
                                  "period_found": period_found, "epoch_found": epoch_found}
                    reports_df = reports_df.append(new_report, ignore_index=True)
                    print("P=" + str(period) + ", R=" + str(r_planet) + ", T0=" + str(epoch) + ", FOUND WAS " + str(
                        found) +
                          " WITH SNR " + str(snr) + " AND SDE " + str(sde))
                    reports_df = reports_df.sort_values(['period', 'radius', 'epoch'], ascending=[True, True, True])
                    reports_df.to_csv(inject_dir + "a_tls_report.csv", index=False)
                except Exception as e:
                    traceback.print_exc()
                    print("File not valid: " + file)
        if not self.preserve:
            for file in os.listdir(inject_dir):
                if file.endswith(".csv") and file.startswith("P"):
                    os.remove(inject_dir + file)

    @staticmethod
    def plot_results(object_id, inject_dir, binning=1, xticks=None, yticks=None, period_grid_geom="lin",
                     radius_grid_geom="lin"):
        """
        Generates a heat map with the found/not found results for the period/radius grids
        :param object_id: the id of the target star
        :param inject_dir: the inject directory where the result files are stored
        :param binning: the binning to be applied to the grids
        :param xticks: the fixed ticks to be used for the period grid
        :param yticks: the fixed ticks to be used for the radius grid
        :param period_grid_geom: [lin|log]
        :param radius_grid_geom: [lin|log]
        """
        df = pd.read_csv(inject_dir + '/a_tls_report.csv', float_precision='round_trip', sep=',',
                         usecols=['period', 'radius', 'found', 'sde'])
        min_period = df["period"].min()
        max_period = df["period"].max()
        min_rad = df["radius"].min()
        max_rad = df["radius"].max()
        phases = len(df[df["period"] == df["period"].min()][df["radius"] == df["radius"].min()])
        phases_str = "phase" if phases == 1 else "phases"
        bin_nums = int(np.ceil(len(df["period"].unique()) / binning))
        if period_grid_geom == 'lin':
            step_period = (max_period - min_period) / (len(df["period"].unique()) - 1)
            step_period = step_period * binning
            if step_period <= 0:
                step_period = 0.1
            period_grid = np.linspace(min_period, max_period, bin_nums)\
                if max_period - min_period > 0 else np.full((1), min_period)
        else:
            period_grid = np.logspace(np.log10(min_period), np.log10(max_period), bin_nums)
        bin_nums = int(np.ceil(len(df["radius"].unique()) / binning))
        if radius_grid_geom == 'lin':
            step_radius = (max_rad - min_rad) / (len(df["radius"].unique()) - 1)
            step_radius = step_radius * binning
            if step_radius <= 0:
                step_radius = 0.1
            radius_grid = np.round(np.linspace(min_rad, max_rad, bin_nums), 2)\
                if max_rad - min_rad > 0 else np.full((1), min_rad)
        else:
            radius_grid = np.round(np.logspace(np.log10(min_rad), np.log10(max_rad), 2), bin_nums)
        f = len(period_grid) / len(radius_grid)
        bins = [period_grid, radius_grid]
        h1, x, y = np.histogram2d(df['period'][df['found'] == 1], df['radius'][df['found'] == 1], bins=bins)
        h2, x, y = np.histogram2d(df['period'][df['found'] == 0], df['radius'][df['found'] == 0], bins=bins)
        normed_hist = (100. * h1 / (h1 + h2))
        fig, ax = plt.subplots(figsize=(2.7 * 5, 5))
        im = plt.imshow(normed_hist.T, origin='lower', extent=(x[0], x[-1], y[0], y[-1]), interpolation='none',
                        aspect='auto', cmap='viridis', vmin=0, vmax=100, rasterized=True)
        plt.colorbar(im, label='Recovery rate (%)')
        plt.xlabel('Injected period (days)')
        plt.ylabel(r'Injected radius (R$_\oplus$)')
        ax.set_title(object_id + " - P/R recovery (" + str(phases) + " " + phases_str + ")")
        if xticks is not None:
            plt.xticks(xticks)
        else:
            period_ticks_decimals = MATRIX.num_of_zeros(max_period - min_period) + 1
            plot_bins = 10 if 10 < len(period_grid) else len(period_grid)
            plt.locator_params(axis="x", nbins=plot_bins)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.' + str(period_ticks_decimals) + 'f'))
        if yticks is not None:
            plt.xticks(yticks)
        plt.savefig(inject_dir + '/inj-rec.png', bbox_inches='tight', dpi=200)
        plt.close()

    @staticmethod
    def plot_diff(object_id, inject_dir1, inject_dir2, output_dir, binning=1, xticks=None, yticks=None,
                  period_grid_geom="lin", radius_grid_geom="lin"):
        """
        Plots the differente between two results directories.
        :param object_id: the target star id
        :param inject_dir1: the results dir number 1
        :param inject_dir2: the results dir number 2
        :param output_dir: the output directory for the plot
        :param binning: the binning to be applied to both axes
        :param xticks: the fixed ticks for the period bar
        :param yticks: the fixed ticks for the radius bar
        :param period_grid_geom: [lin|log]
        :param radius_grid_geom: [ling|log]
        """
        df1 = pd.read_csv(inject_dir1 + '/a_tls_report.csv', float_precision='round_trip', sep=',',
                         usecols=['period', 'radius', 'found', 'sde'])
        df2 = pd.read_csv(inject_dir2 + '/a_tls_report.csv', float_precision='round_trip', sep=',',
                         usecols=['period', 'radius', 'found', 'sde'])
        min_period = df1["period"].min()
        max_period = df1["period"].max()
        min_rad = df1["radius"].min()
        max_rad = df1["radius"].max()
        phases = len(df1[df1["period"] == df1["period"].min()][df1["radius"] == df1["radius"].min()])
        phases_str = "phase" if phases == 1 else "phases"
        bin_nums = int(np.ceil(len(df1["period"].unique()) / binning))
        if period_grid_geom == 'lin':
            step_period = (max_period - min_period) / (len(df1["period"].unique()) - 1)
            step_period = step_period * binning
            if step_period <= 0:
                step_period = 0.1
            period_grid = np.linspace(min_period, max_period, bin_nums)\
                if max_period - min_period > 0 else np.full((1), min_period)
        else:
            period_grid = np.logspace(np.log10(min_period), np.log10(max_period), bin_nums)
        bin_nums = int(np.ceil(len(df1["radius"].unique()) / binning))
        if radius_grid_geom == 'lin':
            step_radius = (max_rad - min_rad) / (len(df1["radius"].unique()) - 1)
            step_radius = step_radius * binning
            if step_radius <= 0:
                step_radius = 0.1
            radius_grid = np.round(np.linspace(min_rad, max_rad, bin_nums), 2)\
                if max_rad - min_rad > 0 else np.full((1), min_rad)
        else:
            radius_grid = np.round(np.logspace(np.log10(min_rad), np.log10(max_rad), 2), bin_nums)
        f = len(period_grid) / len(radius_grid)
        bins = [period_grid, radius_grid]
        h11, x1, y1 = np.histogram2d(df1['period'][df1['found'] == 1], df1['radius'][df1['found'] == 1], bins=bins)
        h12, x1, y1 = np.histogram2d(df1['period'][df1['found'] == 0], df1['radius'][df1['found'] == 0], bins=bins)
        h21, x2, y2 = np.histogram2d(df2['period'][df2['found'] == 1], df2['radius'][df2['found'] == 1], bins=bins)
        h22, x2, y2 = np.histogram2d(df2['period'][df2['found'] == 0], df2['radius'][df2['found'] == 0], bins=bins)
        h1 = h11 - h12
        h2 = h21 - h22
        normed_hist1 = phases * h11 / (h11 + h12)
        normed_hist2 = phases * h21 / (h21 + h22)
        normed_hist = normed_hist1 - normed_hist2
        fig, ax = plt.subplots(figsize=(2.7 * 5, 5))
        im = plt.imshow(normed_hist.T, origin='lower', extent=(x1[0], x1[-1], y1[0], y1[-1]), interpolation='none',
                        aspect='auto', cmap='viridis', vmin=-phases, vmax=phases, rasterized=True)
        plt.colorbar(im, label='# Found samples diff.')
        plt.xlabel('Injected period (days)')
        plt.ylabel(r'Injected radius (R$_\oplus$)')
        ax.set_title(object_id + " - P/R recovery diff(" + str(phases) + " " + phases_str + ")")
        if xticks is not None:
            plt.xticks(xticks)
        else:
            period_ticks_decimals = MATRIX.num_of_zeros(max_period - min_period) + 1
            plot_bins = 10 if 10 < len(period_grid) else len(period_grid)
            plt.locator_params(axis="x", nbins=plot_bins)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.' + str(period_ticks_decimals) + 'f'))
        if yticks is not None:
            plt.xticks(yticks)
        plt.savefig(output_dir + '/inj-rec-diff.png', bbox_inches='tight', dpi=200)
        plt.close()

    def __search(self, time, flux, rstar, rstar_min, rstar_max, mass, mstar_min, mstar_max, ab, epoch,
                 period, min_period, max_period, min_snr, transit_template, ws, transits_min_count,
                 run_limit, custom_search_algorithm, oversampling):
        tls_period_grid, oversampling = LcbuilderHelper.calculate_period_grid(time, min_period, max_period,
                                                                              oversampling, self.star_info,
                                                                              transits_min_count)
        if custom_search_algorithm is not None:
            return custom_search_algorithm.search(time, flux, rstar, rstar_min, rstar_max, mass, mstar_min, mstar_max,
                                                ab, epoch, period, min_period, max_period, min_snr, self.cores,
                                                transit_template, ws, transits_min_count, run_limit)
        else:
            return self.__tls_search(time, flux, rstar, rstar_min, rstar_max, mass, mstar_min, mstar_max, ab, epoch,
                     period, min_period, max_period, min_snr, self.cores, transit_template, ws, transits_min_count,
                     run_limit, tls_period_grid)

    def __tls_search(self, time, flux, rstar, rstar_min, rstar_max, mass, mstar_min, mstar_max, ab, epoch,
                     period, min_period, max_period, min_snr, cores, transit_template, ws, transits_min_count,
                     run_limit, tls_period_grid):
        snr = 1e12
        found_signal = False
        time, flux = cleaned_array(time, flux)
        run = 0
        if ws > 0:
            flux = wotan.flatten(time, flux, window_length=ws, return_trend=False, method='biweight', break_tolerance=0.5)
        while snr >= min_snr and not found_signal and (run_limit > 0 and run < run_limit):
            model = transitleastsquares(time, flux)
            # R_starx = rstar / u.R_sun
            results = model.power(u=ab,
                                  R_star=rstar,  # rstar/u.R_sun,
                                  R_star_min=rstar_min,  # rstar_min/u.R_sun,
                                  R_star_max=rstar_max,  # rstar_max/u.R_sun,
                                  M_star=mass,  # mstar/u.M_sun,
                                  M_star_min=mstar_min,  # mstar_min/u.M_sun,
                                  M_star_max=mstar_max,  # mstar_max/u.M_sun,
                                  period_min=min_period,
                                  period_max=max_period,
                                  n_transits_min=transits_min_count,
                                  show_progress_bar=False,
                                  use_threads=cores,
                                  transit_template=transit_template,
                                  period_grid=tls_period_grid
                                  )
            snr = results.snr
            if results.snr >= min_snr:
                intransit_result = transit_mask(time, results.period, 2 * results.duration, results.T0)
                time = time[~intransit_result]
                flux = flux[~intransit_result]
                time, flux = cleaned_array(time, flux)
                right_period = self.__is_multiple_of(results.period, period / 2.)
                right_epoch = False
                for tt in results.transit_times:
                    for i in range(-5, 5):
                        right_epoch = right_epoch or (np.abs(tt - epoch + i * period) < (1. / 24.))
                #            right_depth   = (np.abs(np.sqrt(1.-results.depth)*rstar - rplanet)/rplanet < 0.05) #check if the depth matches
                if right_period and right_epoch:
                    found_signal = True
                    break
            run = run + 1
        return found_signal, results.snr, results.SDE, run, results.duration, results.period, results.T0

    def __equal(self, a, b, tolerance=0.01):
        return np.abs(a - b) < tolerance

    @staticmethod
    def num_of_zeros(n):
        if n.is_integer():
            return 0
        s = '{:.16f}'.format(n).split('.')[1]
        return len(s) - len(s.lstrip('0'))

    def __is_multiple_of(self, a, b, tolerance=0.05):
        a = np.float(a)
        b = np.float(b)
        mod_ab = a % b
        mod_ba = b % a
        return (a > b and a < b * 3 + tolerance * 3 and (
                    abs(mod_ab % 1) <= tolerance or abs((b - mod_ab) % 1) <= tolerance)) or \
               (b > a and a > b / 3 - tolerance / 3 and (
                           abs(mod_ba % 1) <= tolerance or abs((a - mod_ba) % 1) <= tolerance))
