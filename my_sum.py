import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import CubicSpline
from sklearn.mixture import GaussianMixture
# from lmfit import Model
# from lmfit.models import update_param_vals
import statistics 

import numpy as np
import astropy.constants as cst
import pickle
import os
import math
import glob
import os.path

L = - 0.02 # GMM step size of each model
R =   0.02 # GMM step size
step_list =[40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720, 760, 800, 840, 880, 920, 960, 1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280, 1320, 1360, 1400, 1440, 1480, 1520, 1560, 1600, 1640, 1680, 1720, 1760, 1800, 1840, 1880, 1920, 1960, 2000]
sum_Y = [-20, 20]
width = 8.0 # Marker sample size
AW = 8.0 # width around marker regions to extract from samples
dpi_ = 200

relative_c70 = [6266.0, 6248.5, 6222.5, 6220.0, 6114.0]
markers_c70 = [6285.0, 6175.0, 6129.0, 6091.0, 6074.0, 6041.0, 6037.0, 6030.0, 6266.0, 6222.5, 6220.0, 6114.0]

relative_c60 = [6150.0, 6095.0, 6071.5, 6056.0]
markers_c60 = [6218.0, 6221.0, 6150.0, 6095.0, 6071.5, 6056.0, 5988.0, 5970.0]

relative_c70p = [7959.2, 7518, 7811.9, 7558.4, 7470.2, 7582.3, 7372.8, 7632.6, 7501.6]
markers_c70p = [7857.6, 7777.2, 7766.4, 7821.3, 7750.0, 7728.1, 7713.3, 7685.8, 7959.2, 7518.0, 7811.9, 7558.4, 7470.2, 7582.3, 7372.8, 7632.6, 7501.6]

max60 = max(markers_c60)
max70 = max(markers_c70)
max70p = max(markers_c70p)
min60 = min(markers_c60)
min70 = min(markers_c70)
min70p = min(markers_c70p)

def single_butter(wave, flux, o, s):
    "creates butterworth filter signal of the raw spectrum"
    fs_ = 0.01999506374 # critical frequency
    order = o # selected order parameter
    scale = s # selected scaler for cirtical frequency
    Wn = fs_ * scale 
    b, a = signal.butter(order, Wn)
    y = signal.filtfilt(b, a, flux)
    fgust = signal.filtfilt(b, a, flux, method="gust")
    return fgust

def geo_to_bary(wave, vel):
    "shifts from geocentric to barycentric"
    bary_wave = wave + (vel / cst.c.to("km/s").value) * wave
    return bary_wave

def get_v(wave, lambda_o, lambda_meas):
    "gets interstellar velocity"
    v = (lambda_meas - lambda_o)/lambda_o
    v = v * cst.c.to("km/s").value
    return v[0]

def bary_to_interstellar(bary, v):
    "shifts from barycentric to interstellar"
    corrected = bary / (1 + (v/cst.c.to("km/s").value))
    return corrected

def eliminate(s, bad_characters):
    for item in bad_characters:
        s = s.strip(item)
    return s
    
def mkdirs():
    os.system("rm -r mark60")
    os.system("rm -r mark70")
    os.system("rm -r mark70p")
    os.mkdir("mark60")
    os.mkdir("mark70")
    os.mkdir("mark70p")

    for m in markers_c60:
        os.mkdir("mark60/{}/".format(m))
    for ms in markers_c70:
        os.mkdir("mark70/{}/".format(ms))
    for mss in markers_c70p:
        os.mkdir("mark70p/{}/".format(mss))

def load():

    """
    This function loads from the txt file of innterstellar velocities greated with my_get.py
    """
    file  = open("interstellar.txt", "r") 
    f = file.readlines()
    for l in f:
        ls = l.split("\t")
        star = ls[0]
        sample_ = ls[1]
        sample = sample_[:-3]
        vl = ls[2] 
        vs = vl[1:-2]
        v = float(vs)

        list_bary = glob.glob("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/edibles/npy/{}/bary/*.npy".format(star))
        list_flux = glob.glob("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/edibles/npy/{}/flux/*.npy".format(star))

        for i in range(len(list_bary)):
            bb = list_bary[i]
            ff = list_flux[i]
            B = np.load(bb)
            F = np.load(ff)
            I = bary_to_interstellar(B, v)
            butter = single_butter(I, F, 4, 10)

            imin = min(I)
            imax = max(I)

            "Extract Markers"
            np.save("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/edibles/interstellar/{}bary.npy".format(i), B)
            np.save("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/edibles/interstellar/{}flux.npy".format(i), F)
            np.save("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/edibles/interstellar/{}interstellar.npy".format(i), I)
            np.save("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/edibles/interstellar/{}butter.npy".format(i), butter)

            for m in markers_c60:
                if m <= round(imax) and m >= round(imin):
                    AAL, AAR, = m - AW, m + AW
                    i0, i1 = 0,0
                    for index in range(len(I)):
                        cur = I[index]
                        cur = round(cur, 1)
                        if cur == AAL:
                            # print("greater ", index, cur)
                            i0 = index
                        elif cur == AAR:
                            i1 = index
                    markI = I[i0:i1]
                    markF = F[i0:i1]
                    markb = butter[i0:i1]
                    flub = markb

                    np.save("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/edibles/mark60/{}/{}_inter.npy".format(m,i), markI)
                    np.save("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/edibles/mark60/{}/{}_flux.npy".format(m,i), markF)
                    np.save("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/edibles/mark60/{}/{}_butter.npy".format(m,i), markb)


            for m in markers_c70:
                if m <= round(imax) and m >= round(imin):
                    AAL, AAR, = m - AW, m + AW
                    i0, i1 = 0,0
                    for index in range(len(I)):
                        cur = I[index]
                        cur = round(cur, 1)
                        if cur == AAL:
                            # print("greater ", index, cur)
                            i0 = index
                        elif cur == AAR:
                            i1 = index
                    markI = I[i0:i1]
                    markF = F[i0:i1]
                    markb = butter[i0:i1]
                    flub = - markb

                    np.save("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/edibles/mark70/{}/{}_inter.npy".format(m, i), markI)
                    np.save("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/edibles/mark70/{}/{}_flux.npy".format(m,i), markF)
                    np.save("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/edibles/mark70/{}/{}_butter.npy".format(m,i), markb)


            # for m in markers_c70p:
            #     if m <= round(imax) and m >= round(imin):
            #         AAL, AAR, = m - AW, m + AW
            #         i0, i1 = 0,0
            #         for index in range(len(I)):
            #             cur = I[index]
            #             cur = round(cur, 1)
            #             if cur == AAL:
            #                 i0 = index
            #             elif cur == AAR:
            #                 i1 = index
            #         markI = I[i0:i1]
            #         markF = F[i0:i1]
            #         markb = butter[i0:i1]
            #         flub = - markb

            #         np.save("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/edibles/mark70p/{}/{}_inter.npy".format(m, i), markI)
            #         np.save("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/edibles/mark70p/{}/{}_flux.npy".format(m,i), markF)
            #         np.save("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/edibles/mark70p/{}/{}_butter.npy".format(m,i), markb)

                    # gmm(markI, markF, markb, flub)

                    # plt.plot(markI, markF)
                    # plt.title("C70p")
                    # plt.axvline(m, color="black")
                    # plt.plot(markI, markb)
                    # plt.show()
               
def plot(wave, flux, molecule, label):
    print("{} plot".format(molecule))
    markers = []
    plt.figure(figsize=(8,8))
    if molecule == "c70":
        markers = markers_c70
        for l in markers_c70:
            plt.axvline(l, color= "black", linewidth = 0.8)
            plt.axvline(l, color= "grey", linestyle = "dashed", linewidth = 0.5)
            plt.axvline(l, color= "grey", linestyle = "dashed", linewidth = 0.5)

        for s in relative_c70:
            plt.axvline(s, color= "black", linewidth = 0.8)
            plt.axvline(s, color= "grey", linestyle = "dashed", linewidth = 0.5)
            plt.axvline(s, color= "grey", linestyle = "dashed", linewidth = 0.5)

    # if molecule == "c70p":
    #     markers = markers_c70p
    #     for lp in markers_c70p:
    #         plt.axvline(lp, color= "black", linewidth = 0.8)
    #         plt.axvline(lp, color= "grey", linestyle = "dashed", linewidth = 0.5)
    #         plt.axvline(lp, color= "grey", linestyle = "dashed", linewidth = 0.5)

    #     for sp in relative_c70_plus:
    #         plt.axvline(sp, color= "black", linewidth = 0.8)  
    #         plt.axvline(sp, color= "grey", linestyle = "dashed", linewidth = 0.5)
    #         plt.axvline(sp, color= "grey", linestyle = "dashed", linewidth = 0.5)

    # if molecule == "c60":
    #     markers = markers_c60
    #     for l in markers_c60:
    #         plt.axvline(l, color= "black", linewidth = 0.8)
    #         plt.axvline(l, color= "grey", linestyle = "dashed", linewidth = 0.5)
    #         plt.axvline(l, color= "grey", linestyle = "dashed", linewidth = 0.5)

    #     for s in relative_c60:
    #         plt.axvline(s, color= "black", linewidth = 0.8)
    #         plt.axvline(s, color= "grey", linestyle = "dashed", linewidth = 0.5)
    #         plt.axvline(s , color= "grey", linestyle = "dashed", linewidth = 0.5)


    for b in range(len(wave)):
        plt.plot(wave[b], flux[b], linewidth = 0.5)    
    plt.title("{} {} ".format(molecule, label),  fontsize=10)
    plt.xlabel('Wavelength $\AA$')
    plt.ylabel('Normalized Flux') 
    # plt.xlim(min(wave), max(wave))
    plt.ylim(.82, 1.15)
    plt.savefig("Molecule{}{}.png".format(molecule, label), dpi = dpi_)
    
    # width = 2.0
    for u in range(len(markers)):
        XL = markers[u] - width 
        XR = markers[u] + width 
        plt.xlim(XL, XR)
        plt.ylim(.82, 1.15)
        plt.title("{} Band: {} $\AA$ ".format(molecule, markers[u]),  fontsize=10)
        plt.savefig("{}/Molecule{}_Raw:{}.png".format(molecule, molecule, markers[u]), dpi = dpi_)

    plt.close()

def dict_sum_plot(I, F, b, m, molecule):
    "BUTTERFILTER"
    B_dict = dict() # 1 AA sum
    for bb in range(len(I)):
        aa = round(I[bb],2)
        if aa in B_dict:
            B_dict[aa] += b[bb]
        else:
            B_dict[aa] = b[bb]
    listed_AA_b = sorted(B_dict.keys())
    BB_dict = sorted(B_dict.items())
    B_dict_two = dict() # 2 AA sum
    try:
        for l in range(0,len(listed_AA_b),2):
            B_dict_two[listed_AA_b[l]] = B_dict[listed_AA_b[l]] + B_dict[listed_AA_b[l+1]]
    except:
        pass
    BB_dict_two = sorted(B_dict_two.items())
    BB2_x, BB2_y = zip(*BB_dict_two)
    BB2_x = np.asarray(BB2_x)
    BB2_y = np.asarray(BB2_y)
    B2meanpre = np.mean(BB2_y)
    BB2_ym0 = BB2_y - B2meanpre

    "INTERSTELLAR"
    I_dict = dict() # 1 AA sum
    for o in range(len(I)):
        aa = round(I[o],2)
        if aa in I_dict:
            I_dict[aa] += F[o]
        else:
            I_dict[aa] = F[o]
    listed_AA = sorted(I_dict.keys())
    I_dict_two = dict() # 2 AA sum
    try:
        for l in range(0,len(listed_AA),2):
            I_dict_two[listed_AA[l]] = I_dict[listed_AA[l]] + I_dict[listed_AA[l+1]]
    except:
        pass

    AA_2 = sorted(I_dict_two.items())  
    AA2_x, AA2_y = zip(*AA_2)
    xx = np.asarray(AA2_x)
    yy = np.asarray(AA2_y)
    meanpre = np.mean(yy)

    "mean divided"
    yy = yy / meanpre
    meann = np.mean(yy)
    BB2_y = BB2_y / B2meanpre
    plt.figure(figsize=(14,6))
    plt.plot(xx, yy, label = ".02 Angstrom Interstellar", linewidth = 1, color = "grey")
    plt.plot(BB2_x, BB2_y, label = ".02 Angstrom Filtered", linewidth = 1.2, color = "black")
    plt.axvline(m, color= "black", linewidth = 0.8)
    plt.axvline(m - 1, color= "grey", linestyle = "dashed", linewidth = 0.5)
    plt.axvline(m + 1, color= "grey", linestyle = "dashed", linewidth = 0.5)
    plt.ylim(.995, 1.005)
    plt.xlim(m - width, m + width)
    plt.xlabel('Wavelength $\AA$ ')
    plt.ylabel('Mean Divided Sum Flux') 
    plt.legend(loc='lower left')
    plt.title("{} Marker {}$\AA$ ".format(molecule, m))
    plt.savefig("{}_{}div{}.png".format(molecule, m, width))
    plt.show()
    plt.close()

def summ():
    """
    This function loads the interstellar numpy arrays stored in memory from the Load() function
    appeds all reads pertainning to specific species, sorts the unified array
    calls dict_sum_plot() function that turns the array into a bined dict and plots final sums
    """
    "C60"
    for m in markers_c60:
        c60i = np.asarray([])
        c60f = np.asarray([])
        c60b = np.asarray([])
        file_inter = glob.glob("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/edibles/mark60/{}/*inter.npy".format(m))
        for i in file_inter:
            num = i[86:-10]
            curi = np.load(i)
            curf = np.load("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/edibles/mark60/{}/{}_flux.npy".format(m, num))
            curb = np.load("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/edibles/mark60/{}/{}_butter.npy".format(m, num))
            c60i = np.append(c60i, curi)
            c60b = np.append(c60b, curb)
            c60f = np.append(c60f, curf)

        dex = np.argsort(c60i)
        I = c60i[dex]
        F = c60f[dex]
        b = c60b[dex]
        dict_sum_plot(I, F, b, m, "C60")

    "C70"
    for ma in markers_c70:
        c60i = np.asarray([])
        c60f = np.asarray([])
        c60b = np.asarray([])
        file_inter = glob.glob("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/edibles/mark70/{}/*inter.npy".format(ma))
        for i in file_inter:
            num = i[86:-10]
            curi = np.load(i)
            curf = np.load("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/edibles/mark70/{}/{}_flux.npy".format(ma, num))
            curb = np.load("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/edibles/mark70/{}/{}_butter.npy".format(ma, num))
            c60i = np.append(c60i, curi)
            c60b = np.append(c60b, curb)
            c60f = np.append(c60f, curf)

        dex = np.argsort(c60i)
        I = c60i[dex]
        F = c60f[dex]
        b = c60b[dex]
        dict_sum_plot(I, F, b, ma, "C70")

    "C70p" # removed due to telluric contamination 

    # for mar in markers_c70p:
    #     c60i = np.asarray([])
    #     c60f = np.asarray([])
    #     c60b = np.asarray([])
    #     file_inter = glob.glob("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/edibles/mark70p/{}/*inter.npy".format(mar))
    #     for i in file_inter:
    #         num = i[87:-10]
    #         curi = np.load(i)
    #         print(i)
    #         print(num)
    #         curf = np.load("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/edibles/mark70p/{}/{}_flux.npy".format(mar, num))
    #         curb = np.load("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/edibles/mark70p/{}/{}_butter.npy".format(mar, num))
    #         c60i = np.append(c60i, curi)
    #         c60b = np.append(c60b, curb)
    #         c60f = np.append(c60f, curf)

    #     dex = np.argsort(c60i)
    #     I = c60i[dex]
    #     F = c60f[dex]
    #     b = c60b[dex]
    #     dict_sum_plot(I, F, b, mar, "C70p")

if __name__ == "__main__":


    "TO DO ONLY ONCE" 
    # mkdirs() # makes directories for the interstellar corrected specie numpy arrays
    # load() # this loads the respective

    "LOAD PER MARKER SUM"
    summ()





