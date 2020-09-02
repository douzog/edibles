import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import CubicSpline
import statistics 

import numpy as np
import astropy.constants as cst
import pickle
import os
import math
import glob
import os.path

"""
Na is 5889, 5895 
potassium K is 
 1000 P	 7664.8991	K I	E99
 1000 P	 7698.9645	K I
mean : 2.2885792677000003
std : 0.34332784898939206
"""
"""
took out: 
    "HD186745", "HD186841", HD112272
    HD133518
    HD36861
    HD36486
    HD37903
    HD24398
    HD36695
    HD37061
    HD22951
    HD23180
    HD37367
    HD37022
    HD34748
    HD40111
    HD23016
    HD27778
    HD75309
    HD41117
    HD54239
    HD53975
    HD37041
    HD43384
    HD54439
    HD36822
    HD54662
    HD55879
    HD66194
    HD73882
    HD149404
    HD157978
    HD75860
    HD147084
    HD147889
    HD151804
    HD172694
"""

stars = ["HD57061","HD66811","HD79186","HD80558","HD81188","HD91824","HD93030","HD93205","HD93222",
        "HD93843","HD94493","HD99953","HD109399","HD111934","HD113904","HD114886","HD116852","HD122879","HD124314",
        "HD135591","HD143275","HD144470","HD145502","HD147165","HD147683","HD147888","HD147933","HD148605","HD148937",
        "HD149038","HD149757","HD150136","HD152248","HD152408","HD152424","HD153919","HD154043","HD155806","HD157246","HD161056",
        "HD164073","HD164353","HD164740","HD164906","HD166937","HD167264","HD167838","HD167971","HD169454", "HD170740",
        "HD170938","HD171957", "HD180554","HD184915", "HD185418", "HD185859", "HD203532", "HD303308"]

setting = ["564", "860"]
red_ = ["redl", "redu"]

SINGLE = False

x, y = 3460, 8600 
step_ = 40
remove_first, remove_last  = 520, -10
dpi_ = 200
L = 100
R = 100

telluric_1 = 7698.31
telluric_2 = 7699.5

relative_c70 = [6266.0, 6248.5, 6222.5, 6220.0, 6114.0]
markers_c70 = [6285.0, 6200.0, 6185.0, 6175.0, 6129.0, 6129.0, 6124.0, 6091.0, 6074.0, 6041.0, 6037.0, 6030.0, 6266.0, 6248.5, 6222.5, 6220.0, 6114.0]
relative_c60 = [6150.0, 6095.0, 6071.5, 6056.0]
markers_c60 = [6218.0, 6221.0, 6150.0, 6095.0, 6071.5, 6056.0, 5988.0, 5970.0]
relative_c70p = [7959.2, 7518, 7811.9, 7558.4, 7470.2, 7582.3, 7372.8, 7632.6, 7501.6]
markers_c70p = [7857.6, 7777.2, 7766.4, 7821.3, 7750, 7728.1, 7713.3, 7685.8, 7959.2, 7518, 7811.9, 7558.4, 7470.2, 7582.3, 7372.8, 7632.6, 7501.6]

max60 = max(markers_c60)
max70 = max(markers_c70)
max70p = max(markers_c70p)
min60 = min(markers_c60)
min70 = min(markers_c70)
min70p = min(markers_c70p)

below_60 = min60 - L
above_60 = max60 + R
below_70 = min70 - L
above_70 = max70 + R
below_70p = min70p - L
above_70p = max70p + R

wwave = []
bbary = []
fflux = []
nname = []
k_bary = []
k_geo = []
k_flux = []

k_interstellar_w = []
k_interstellar_f = []
interstellar_name = []
aa_peaks = []
ff_peaks = []

c60_wave = []
c70_wave= []
c70p_wave = []
c60_bary = []
c70_bary= []
c70p_bary = []
c60_flux = []
c70_flux = []
c70p_flux = []
c60_name = []
c70_name = []
c70p_name = []


class EdiblesSpectrum:
    """
    This class takes a spectrum file from EDIBLES,
    reads the header and data, and creates a DataFrame.
    The class will also contain a set of methods to operate on the data.
    :param filename: Name of the file, starting with the target
    :type filename: str
    :param header: The header of the FITS file from the target observation
    :type header: Object (astropy.io.fits.header.Header)
    :param target: The name of the target
    :type target: str
    :param date: Date of the target observation
    :type date: str
    :param v_bary: Barycentric velocity of the target star
    :type v_bary: float
    :param df: Pandas array containing geocentric and barycentric wavelength, and flux
    :type df: Pandas array (pandas.core.series.Series)
    :param wave: The wavelength grid for the spectrum, geocentric reference frame
    :type wave: Pandas array (pandas.core.series.Series)
    :param wave_units: The units of the wavelength array
    :type wave_units: str
    :param bary_wave: The wavelength grid for the spectrum, barycentric reference frame
    :type bary_wave: Pandas array (pandas.core.series.Series)
    :param flux: The flux data for the spectrum
    :type flux: Pandas array (pandas.core.series.Series)
    :param flux_units: The units of the flux data
    :type flux_units: str
    """

    def __init__(self, filename):
        """
        Filename is relative to the DR3 directory
        """
        # old way
        # self.filename = DATADIR + filename
        # # my way
        self.filename =  filename
        # print(filename)
        # print("spectrum filename", filename)
        self.loadSpectrum()

    def loadSpectrum(self):
        # Assume the file is a DR3 product here.
        hdu = fits.open(self.filename)
        self.header = hdu[0].header
        self.target = self.header["OBJECT"]
        self.date = self.header["DATE-OBS"]

        flux = hdu[0].data
        crval1 = self.header["CRVAL1"]
        cdelt1 = self.header["CDELT1"]
        lenwave = len(flux)
        grid = np.arange(0, lenwave, 1)
        wave = (grid) * cdelt1 + crval1
        self.v_bary = self.header["HIERARCH ESO QC VRAD BARYCOR"]
        bary_wave = wave + (self.v_bary / cst.c.to("km/s").value) * wave


        # print("hdu", hdu[0].header)
        # print("flux", flux)
        # print("wave" ,wave)
        # print("crval1", crval1)
        # print("cdelt1", cdelt1)

        d = {
            "wave": wave.tolist(),
            "bary_wave": bary_wave.tolist(),
            "flux": flux.tolist(),
        }
        self.df = pd.DataFrame(data=d)
        self.wave = self.df["wave"]
        self.wave_units = "AA"
        self.bary_wave = self.df["bary_wave"]
        self.flux = self.df["flux"]
        self.flux_units = "arbitrary"

    def getSpectrum(self, xmin=None, xmax=None):
        """
        Function to get the wavelength and flux arrays of a particular target.
        If xmin/xmax are not called, the data for the entire spectrum will be returned.
        Args:
            xmin (float): minimum wavelength (Optional)
            xmax (float): Maximum wavelength (Optional)
            bary (bool): Barycentric rest frame, default=False
        Returns:
            ndarray: wavelength grid
            ndarray: flux grid
        """

        if (xmin is not None) and (xmax is not None):
            assert xmin < xmax, "xmin must be less than xmax"

            df_subset = self.df[self.df["wave"].between(xmin, xmax)]

            return df_subset

        return self.df

def eliminate(s, bad_characters):
    for item in bad_characters:
        s = s.strip(item)
    return s

def slope(x1, y1, x2, y2):
    return (y2-y1)/(x2-x1)

def molecule_plot(wave, flux, name, molecule):
    plt.figure(figsize=(50,8))

    if molecule == "C70":
        for l in markers_c70:
            plt.axvline(l, color="grey")
        for s in relative_c70:
            plt.axvline(s, color="black")

    if molecule == "C70p":
        for lp in markers_c70p:
            plt.axvline(lp, color="yellow")
        for sp in relative_c70p:
            plt.axvline(sp, color="green")     

    if molecule == "C60":
        for l in markers_c60:
            plt.axvline(l, color="pink")
        for s in relative_c60:
            plt.axvline(s, color="red")
            
    for b in range(len(wave)):
        plt.plot(wave[b], flux[b], label=name[b], linewidth = 0.5)
    # plt.ylim(0.97,1.05)
    plt.ylim(0.5,2)

    plt.title("Molecule | All {}".format(molecule),  fontsize=10)
    plt.savefig("Molecule_{}.png".format(molecule), dpi = dpi_)
    # plt.show()
    plt.close()

def star_plot(wave, flux, name, shifts, molecule):
    plt.figure(figsize=(60,8))
    zoom = [0,0]
    if molecule == "C70":
        for l in markers_c70:
            plt.axvline(l, color="grey")
        for s in relative_c70:
            plt.axvline(s, color="black")
        zoom = [min70 - 5, max70 +5]

    if molecule == "C70p":
        for lp in markers_c70p:
            plt.axvline(lp, color="yellow")
        for sp in relative_c70p:
            plt.axvline(sp, color="green")     
        zoom = [min70p - 5, max70p +5]

    if molecule == "C60":
        for l in markers_c60:
            plt.axvline(l, color="pink")
        for s in relative_c60:
            plt.axvline(s, color="red")
        zoom = [min60 - 5, max60 +5]

    for b in range(len(wave)):
        plt.plot(wave[b], flux[b], label=name[b], linewidth = 0.5)

    ylim = [0.97, 1.05] 
    plt.ylim(ylim[0], ylim[1])
    plt.xlim(zoom[0],zoom[1])
    plt.title("{} {}-{} $\AA$ | Barycentric | All {}".format(stars[star], zoom[0], zoom[1], molecule), fontsize=10)
    plt.savefig("plots/{}_{}-{}_multiplot_star_{}.png".format(stars[star], zoom[0], zoom[1], molecule), dpi = dpi_)
    # plt.show()
    plt.close()

def single_plot(wave, flux, name, setting, red, mean, std):           

    MAW = max(wave)
    MIW = min(wave)
    MAF = max(flux) 
    MIF = min(flux) 

    MAF = 1.025
    MIF = .95

    plt.figure(figsize=(20,8))
    for l in markers_c70:
        plt.axvline(l, color="grey")
    for s in relative_c70:
        plt.axvline(s, color="black")

    for lp in markers_c70p:
        plt.axvline(lp, color="yellow")
    for sp in relative_c70p:
        plt.axvline(sp, color="green")     

    for l in markers_c60:
        plt.axvline(l, color="pink")
    for s in relative_c60:
        plt.axvline(s, color="red")

    plt.plot(wave, flux, label=name, linewidth = 0.5)
    plt.xlim(MIW, MAW)
    plt.ylim(MIF, MAF)
    plt.xticks(np.arange(MIW, MAW, 5.0))
    plt.title("{} {} $\AA$ | Barycentric | Single {} mean: {} eche: {} ".format(name, setting, red, mean, std),  fontsize=10)
    # plt.savefig("single/{}_{}_{}_{}_single.png".format(eche, name, setting, red), dpi = dpi_)
    plt.savefig("single/{}_{}_{}.png".format(int(MIW), int(MAW), name), dpi = dpi_)

    plt.close()

def geo_to_bary(wave, vel):
    bary_wave = wave + (vel / cst.c.to("km/s").value) * wave
    return bary_wave

def get_v(wave, lambda_o, lambda_meas):
    v = (lambda_meas - lambda_o)/lambda_o
    v = v * cst.c.to("km/s").value
    return v[0]

def bary_to_interstellar(bary, v):
    corrected = bary / (1 + (v/cst.c.to("km/s").value))
    return corrected

def K_peaks(wave_, bary_, flux_m, b_left, b_right, vel, AA):
    label = ""
    interstellar_velocity = 0.0
    wave_peak = []
    flux_peak = []
    aa_ = 0.0
    ff_ = 0.0
    shift = 0.0

    if b_left < AA and b_right > AA:
        AAL, AAR, = round(AA-2, 1), round(AA+2, 1)
        i0, i1 = 0,0
        slope_left = []
        slope_right =[]
        for index in range(len(wave_)):
            cur = wave_[index]
            cur = round(cur, 1)
            if cur == AAL:
                # print("greater ", index, cur)
                i0 = index
            elif cur == AAR:
                i1 = index
                # print("less ", index, cur)

        wave_k = wave_[i0:i1]
        bary_k = bary_[i0:i1]
        flux_k = flux_m[i0:i1]
        np.save("npy/{}/K/{}_geo.npy".format(stars[star], name), wave_k)
        np.save("npy/{}/K/{}_flux.npy".format(stars[star], name), flux_k)
        np.save("npy/{}/K/{}_bary.npy".format(stars[star], name), bary_k)

        flup_na = - flux_k + 1.1
        peaks, _ = find_peaks(flup_na, height=(np.mean(flup_na)+.1, .8)) # find peaks

        if len(peaks) >= 1:
            for pe in range(len(peaks)):
                try:
                    x1, x2 = wave_k[peaks[pe]-2], wave_k[peaks[pe]] # previous vs peak
                    y1, y2 = flup_na[peaks[pe]-2], flup_na[peaks[pe]]
                    slol = slope(x1,y1, x2, y2)
                    x1, x2 = wave_k[peaks[pe]], wave_k[peaks[pe]+2] # next 2 vs peak
                    y1, y2 = flup_na[peaks[pe]], flup_na[peaks[pe]+2]
                    slor = slope(x1,y1, x2, y2)
                    slope_right.append(slor)
                    slope_left.append(slol)
                except:
                    pass

        if len(peaks) == 2:
            label = "2peak"
        if len(peaks) > 2:
            label = "poor_read"
        if len(peaks) > 4:
            label = "noise"
        if len(peaks) == 1:
            # if wave_k[peaks] < 7698.0:
            #     label = "anomaly"
            if wave_k[peaks[0]] <= telluric_1 + .11 and wave_k[peaks[0]] >= telluric_1 - .11:
                label = "telluric"
            if wave_k[peaks[0]] <= telluric_2 + .11 and wave_k[peaks[0]] >= telluric_2 - .11:
                label = "telluric"
            else:
                label = "1peak"
            wave_peak = wave_k[peaks]
            flux_peak = flux_k[peaks]



        if label == "1peak":
            dot = wave_k[peaks[0]]
            if dot > AA: # if bigger subtract line from dot return negative
                shift = -(dot - AA)
            elif dot < AA:
                shift = AA - dot
                
        "shift peaks to barycentric frame, and get new velocity"
        if label == "1peak":
            k_geo.append(wave_k)
            k_bary.append(bary_k)
            k_flux.append(flux_k)


            # np.save("{}.npy".format(name))

            aa_ = wave_k[peaks]
            ff_ = flux_k[peaks]

            label = "1peak_interstellar"
            bary_peaks = geo_to_bary(wave_k[peaks], vel)
            bary_flux_peaks = flux_k[peaks]
            # get velocity (bary wave, expected, measured)
            interstellar_velocity = get_v(bary_k, AA, bary_peaks)
            corrected_wave = bary_to_interstellar(bary_k, interstellar_velocity)
            np.save("npy/{}/K/{}_inter.npy".format(stars[star], name), corrected_wave)

            "Interstellar"
            plt.figure(figsize=(10,8))
            plt.axvline(AA, color="black")
            plt.plot(corrected_wave, flux_k, color="grey") # normal
            plt.ylim(0, 1.2)
            plt.title("{} | K | Interstellar {}".format(label, name))
            plt.title("{} | K | {} slope l/r {} {}".format(label, name, slope_left, slope_right))
            plt.savefig("K/{}_interstellar_{}.png".format(label, name), dpi = dpi_)
            plt.close()
            
            k_interstellar_w.append(corrected_wave)
            k_interstellar_f.append(flux_k)
            interstellar_name.append(name)



        
        "geocentric"
        plt.figure(figsize=(15,8))
        plt.axvline(AA, color="pink")
        plt.axvline(telluric_1, color="red")
        plt.axvline(telluric_2, color="red")
        plt.plot(wave_k, flup_na, color="pink") # fluped
        plt.plot(wave_k, flux_k, color="grey") # normal
        # plt.plot(wave_k+shift, flux_k, color="black") # shifted
        plt.plot(wave_k[peaks],flux_k[peaks], marker='o', color="red") # peaks 
        plt.ylim(0, 1.2)
        plt.title("{} | K | Gecocentric {}".format(label, name))
        plt.title("{} | K | {} slope l/r {} {}".format(label, name, slope_left, slope_right))
        plt.savefig("K/{}_geo_{}.png".format(label, name), dpi = dpi_)
        plt.close()
        
        "barycentric"
        plt.figure(figsize=(15,8))
        plt.axvline(AA, color="pink")
        # plt.axvline(telluric_1, color="red")
        # plt.axvline(telluric_2, color="red")
        plt.plot(bary_k, flup_na, color="pink") # fluped
        plt.plot(bary_k, flux_k, color="grey") # normal
        # plt.plot(wave_k+shift, flux_k, color="black") # shifted
        plt.plot(bary_k[peaks],flux_k[peaks], marker='o', color="red") # peaks 
        plt.ylim(0, 1.2)
        plt.title("{} | K | Barycentric {}".format(label, name))
        # plt.title("{} | K | {} slope l/r {} {}".format(label, name, slope_left, slope_right))
        plt.savefig("K/{}_bary_{}.png".format(label, name))
        plt.close()

    return shift, wave_peak, flux_peak, label, aa_, ff_, interstellar_velocity

def clear():
    os.system("rm -r single")
    os.system("rm -r npy")
    os.system("rm -r filter")
    os.system("rm -r interstellar")

    os.system("rm text_DB.txt")
    os.mkdir("filter/")
    os.mkdir("single/")
    os.mkdir("interstellar/")
    os.mkdir("npy/")

def single_butter(wave, flux, name, setting, red, mean, std):
    "get interactive plots to fid the rigth Wn scalar"
    "play with the order of the filter,"
    fs_ = 0.01999506374 # critical frequency
    order = 3
    scale = 10
    order = 4
    scale = 9

    Wn = fs_ * scale 
    b, a = signal.butter(order, Wn)
    y = signal.filtfilt(b, a, flux)
    plt.figure(figsize=(8,8))
    fgust = signal.filtfilt(b, a, flux, method="gust")
    plt.title("Observation vs Forward-Backward Filter")
    plt.plot(wave, flux, 'k-', label='{}'.format(name[:-4]), color="black")
    plt.plot(wave, fgust, 'b-', linewidth=0.8, label='Butterworth Filter')
    plt.xlabel('Wavelength $\AA$')
    plt.ylabel('Normalized Flux') 
    plt.ylim(np.mean(flux) -0.05, np.mean(flux) + 0.05)    
    plt.legend()
    plt.xlim(np.mean(wave) -2 , np.mean(wave)+2)
    plt.savefig("butter/{}_order_{}scale_{}.png".format(name, order,scale), dpi = dpi_)
    plt.close()

    return fgust

if __name__ == "__main__":
    clear()
    for star in range(len(stars)):
        swwave = []
        sbbary = []
        sfflux = []
        snname = []  
        sshifts = []
        molecule = ""
        for u in range(len(setting)):
            try:
                os.mkdir("npy/{}/".format(stars[star]))
                os.mkdir("npy/{}/bary/".format(stars[star]))
                os.mkdir("npy/{}/K/".format(stars[star]))
                os.mkdir("npy/{}/flux/".format(stars[star]))
                os.mkdir("npy/{}/geo/".format(stars[star]))
            except:
                pass
            for r in range(len(red_)):        
                "UPDATE DIRECTORY HERE"     
                # path = "/home/idouzoglou/anaconda3/lib/python3.7/site-packages/edibles/edibles/data/DR4_fits/{}/RED_{}/".format(stars[star], setting[u])
                # listfile = glob.glob("/home/idouzoglou/anaconda3/lib/python3.7/site-packages/edibles/edibles/data/DR4_fits/{}/RED_{}/{}_w{}_{}_*.fits".format(stars[star], setting[u], stars[star], setting[u], red_[r]))
                path = "/Users/uchi/Fullerene_code/DR4_all/{}/RED_{}/*".format(stars[star], setting[u])
                listfile = glob.glob("/Users/uchi/Fullerene_code/DR4_all/{}/RED_{}/{}_w{}_{}_*.fits".format(stars[star], setting[u], stars[star], setting[u], red_[r]))

                for f in range(0, len(listfile)):
                    file = listfile[f] # checks items in list
                    new = eliminate(file, path) # elimiates path
                    name = new[:-4]
                    if name[-1] == ".":
                        name = name[:-1]
                    sp = EdiblesSpectrum(file) # get edible specturm
                    subset = sp.getSpectrum(xmin=x, xmax=y)
                    velocity = sp.v_bary
                    del sp

                    if subset.empty:
                        del subset # if empty, pass
                        pass
                    else:
                        flux = subset["flux"].to_numpy() # convert to numpy
                        flux_ = flux[remove_first:remove_last] # remove first and las n that are erroneous
                        meann = np.mean(flux_) # take the mean
                        if meann == 0.0:
                            pass
                        else:
                            wave = subset["wave"].to_numpy()
                            bary = subset["bary_wave"].to_numpy()
                            bary_ = bary[remove_first: remove_last] # remove first and las n that are erroneous
                            wave_ = wave[remove_first:remove_last] # remove first and las n that are erroneous
                            flux_m = flux_ / meann # mean center
                            stdd = np.std(flux_m) # check std
                            mean_mean = np.mean(flux_m)
                            if stdd > 1.0: # erroneous 
                                pass
                            
                            b_right = max(bary_) 
                            b_left = min(bary_) 
                            w_left = max(wave_)
                            w_right = min(wave_)

                            "FIND POTTASIUM PEAKS"
                            shift , wave_k, flux_k, label, AP_, FP_ , interstellar_velocity = K_peaks(wave_, bary_, flux_m, b_left, b_right, velocity, 7698.9645) 
                            "ASSIGN MOLECULAR RANGE"
                            if b_left > below_60 and b_left < above_60 or b_right > below_60 and b_right < above_60:
                                molecule = "C60"
                            if b_left > below_70 and b_left < above_70 or b_right > below_70 and b_right < above_70:
                                molecule = "C70"
                            if b_left > below_70p and b_left < above_70p or b_right > below_70p and b_right < above_70p:
                                molecule = "C70p"

                            "SAVE INTERSTELLAR VELOCITIES"
                            if label == "1peak_interstellar":
                                with open("interstellar.txt", "a") as int_file:
                                    int_file.write("{}\t{}\t{}\n".format(stars[star], name, interstellar_velocity))
                            "IF NOT NONISE SAVE"
                            if label != "noise":
                                np.save("npy/{}/bary/{}_{}_{}.npy".format(stars[star], int(b_left), int(b_right), name), bary_)
                                np.save("npy/{}/geo/{}_{}_{}.npy".format(stars[star], int(w_left), int(w_right), name), wave_)
                                np.save("npy/{}/flux/{}_{}_{}.npy".format(stars[star], int(b_left), int(b_right), name), flux_m)
                                if SINGLE == True:
                                    single_butter(wave_, flux_m, name, setting[u], red_[r], meann, stdd)
                                    single_plot(wave_, flux_m, name, setting[u], red_[r], meann, stdd)
                                "SAVE TXT INFORMATION"
                                with open("text_DB.txt", "a") as txt_file:
                                    txt_file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(stars[star], name, int(b_left), int(b_right), setting[u], red_[r],  meann, stdd, shift, label, interstellar_velocity, molecule, AP_, FP_))

                                # wwave.append(wave_)
                                # bbary.append(bary_)
                                # fflux.append(flux_m)
                                # nname.append(name)
                                # swwave.append(wave_)
                                # sbbary.append(bary_)
                                # sfflux.append(flux_m)
                                # snname.append(name)

        # star_plot(sbbary, sfflux, snname, sshifts, "C60")
        # star_plot(sbbary, sfflux, snname, sshifts, "C70")
        # star_plot(sbbary, sfflux, snname, sshifts, "C70p")

    # molecule_plot(c60_bary, c60_flux, c60_name, "C60")
    # molecule_plot(c70_bary, c70_flux, c70_name, "C70")
    # molecule_plot(c70p_bary, c70p_flux, c70p_name, "C70p")


    "BARYCENTRIC K"
    plt.figure(figsize=(8,8))
    for k in range(len(k_bary)):
        plt.axvline(7698.9645, color="black")
        plt.axvline(telluric_1 , color="black", linestyle = "dashed", linewidth = 0.5)
        plt.axvline(telluric_2, color="black", linestyle = "dashed", linewidth = 0.5)
        plt.plot(k_bary[k], k_flux[k], linewidth = 0.5)
        plt.xlim(7698.9645 -2, 7698.9645 +2)
        plt.xlabel('Wavelength $\AA$')
        plt.ylabel('Normalized Flux') 
        plt.title("Barycentric K")
        plt.savefig("K_barycentric.png", dpi = dpi_)
    plt.close()

    "GEOCENTRIC K"
    plt.figure(figsize=(8,8))
    for k in range(len(k_geo)):
        plt.axvline(7698.9645, color="black")
        plt.axvline(telluric_1 , color="black", linestyle = "dashed", linewidth = 0.5)
        plt.axvline(telluric_2, color="black", linestyle = "dashed", linewidth = 0.5)
        plt.plot(k_geo[k], k_flux[k], linewidth = 0.5)
        plt.xlim(7698.9645 -2, 7698.9645 +2)
        plt.xlabel('Wavelength $\AA$')
        plt.ylabel('Normalized Flux') 
        plt.title("Geocentric K")
        plt.savefig("K_geocentric.png", dpi = dpi_)
    plt.close()

    "ITERSTELLAR K"
    plt.figure(figsize=(8,8))
    for p in range(len(k_interstellar_w)):
        plt.axvline(7698.9645, color="black")
        plt.plot(k_interstellar_w[p], k_interstellar_f[p], label = interstellar_name[p], linewidth = 0.5)
        plt.xlim(7698.9645 -2, 7698.9645 +2)
        for pp in aa_peaks:
            plt.plot(aa_peaks[pp],ff_peaks[pp], marker='o')
        plt.xlabel('Wavelength $\AA$')
        plt.ylabel('Normalized Flux') 
        plt.title("Interstellar K")
        plt.legend
        plt.savefig("K_interstellar.png", dpi = dpi_)
    plt.close()
