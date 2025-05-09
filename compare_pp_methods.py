# numerical essentials 
import numpy as np

# for plotting
from matplotlib import pyplot as plt, animation
import pandas as pd
import xarray as xr

# for plotting
import matplotlib.colors as mcolors
import matplotlib.dates as mdates

# for cdf reading
from spacepy import toolbox
#spacepy.toolbox.update(leapsecs=True)
from spacepy import pycdf
import h5py
import csv

# other numerical tools
import os
from datetime import datetime,timedelta,date,time

# my own misc. functions
import global_use as gl
import funcs_analysis as fa
import find_properties as fp


""" This code decides whether, for a given timestamp, the spacecraft is within the plasmasphere or the plasmatrough. It does this by considering each half (or <half orbit in edge cases) independetly.
The region can then be decided based upon one of three methods. In decreasing priority, these are:

1. via a density gradient > 5 within an Lstar range of 0.5 (Moldwin et al., 2002). This method identifies what is knwon as the plasmapause: a suddden and obvious density drop as you leave the plasmasphere
It is an ideal case and this density 'knee' is not always present and this method may miss more gradual density changes from plasmasphere to plasmatrough, or when density data is sparse. Therefore this
method should be used in the first instance, before moving onto methods 2 or 3. 

2. via identifying ECH waves.  Electron cyclotron harmonic (ECH) waves are emissions within multiples of the electron gyrofrequency, observed outside of the plasmasphere and often terminate at the plasmapause
boundary (Meredith et al., 2004). This can be incredibly useful when there is a lack of density data, as the sudden lack of emissions is indicative of a density gradient. However, ECH waves are not always observed, 
and in some cases, there is leakage into the plasmasphere (Liu et al., 2020).

3. via a density threshold. This is the method of last resort. The plasmapause density can drop from plasmasphere to plasmatrough densities(from 100cm−3 to 1) gradually over a width greater than that defined by the
gradient threshold. To account for this, a plausible absolute density between 30-100cm3 is often chosen as an alternative. The density threshold chosen is relativelyarbitrary and is therefore a non-case
specific estimate that may not positionally agree with results found in method 1, or 2. To utilise the density gradient as far as possible (Li et al., 2010a), the density threshold is taken as the largest value of 
50cm−3 and n = 10(6.6/L)^4, where the latter is from an empirical model found using satellite measurements (Sheeley et al., 2001).
"""

""" In the following code, a flag is used to indicate on each half orbit which method was used - 1, 2 or 3 - or whether it was impossible on a given orbit - 0.
"""


""" necesssary functions """


def find_ech(gyro_list,gyro_epoch,WFR_E,HFR_E,WFR_epoch,HFR_epoch,WFR_freqeucny,HFR_frequency,time):
    
        lower_lim = 10**(4)

        # find gyrofrequency 
        gyro = gyro_list[gl.find_closest(gyro_epoch,time)[1]]

        # finding HFR time/index asscoiated wth density stamp
        HFR_time,HFR_index = gl.find_closest(HFR_epoch,time)

        # Now deciding whether we fall off the HFR lower limit into the WFR bounds: finding WFR data index for HFR lower limit

        index_lowerlim = gl.find_closest(WFR_freqeucny,lower_lim)[1]

        if (gyro < lower_lim ):
            frequency_add=[]
            spectra_add=[]
            # Find frequency in WFR data that corresponds closest to this
            # find all frequencies between integration limit...
            # ... and where HFR lower limit starts

            index_freqwfr =gl.find_closest(WFR_freqeucny,gyro)[1]
  
            index_epochwfr = gl.find_closest(WFR_epoch,HFR_time)[1]

        
        # now find all of the frequencies between this and lower limit

            for l in range(index_freqwfr,index_lowerlim-1,1):
                frequency_add.append(WFR_freqeucny[l])
                spectra_add.append(WFR_E[index_epochwfr,l])
            frequency_edit =np.concatenate((frequency_add,HFR_frequency))
            spectra_edit=np.concatenate((spectra_add,HFR_E[HFR_index,:]))
        
        else:
            frequency_edit =HFR_frequency
            spectra_edit=HFR_E[HFR_index,:]
    
        int = 0
        for j in range(len(spectra_edit)-1):

            if (3.*gyro < frequency_edit[j] < 5.*gyro): # integration lmits between fce < f < 2*fce
                
                if (10**4<frequency_edit[j]<10**5)and(10**4<frequency_edit[j+1]<10**5):
                    int = int + 0.5*(spectra_edit[j+1]+spectra_edit[j])*(frequency_edit[j+1]-frequency_edit[j])
        
        ECH_int = np.sqrt(int*gyro)
        return ECH_int



def outwards_journ(input_dict, gee_index):


    def density_thresh(fpe_epoch, lanl_epoch, MLT, MLAT_N, MLAT_S, Lstar,density,d_perigee,d_apogee,epoch_omni,AE,epoch_omni_low,Dst,Kp):
    # resort back to density threshold

        # all initial values of plasmapause crossings set to np.nan
        pp_type = None
        pp_L = np.nan
        pp_time = np.nan
        pp_AE = np.nan
        pp_AEStar = np.nan
        pp_MLT = np.nan

        # cycle through the density data on that half orbit
        for k in range(d_perigee,d_apogee):
            
            # find correspinding Lstar and MLT values from LANL data
            findLstar= fp.FindLANLFeatures(fpe_epoch[k], lanl_epoch, MLT, MLAT_N, MLAT_S, Lstar)
            Lstar_stamp = findLstar.get_Lstar
            MLT_stamp = findLstar.get_MLT   

            # find corresponding AE and AE* from omni data
            findOmniFeats = fp.FindOMNIFeatures(fpe_epoch[k], epoch_omni, epoch_omni_low, AE, Kp, Dst)
            AE_stamp = findOmniFeats.get_AE
            AE_star = findOmniFeats.get_AEstar

            # find the density threshold - largest between 50 and empirical L relationship for threshold
            thresh = np.min([50,10*(6.6/Lstar_stamp)**4])
    
            # if current density below threshold, have found plasmapause and therefore update crossing details and leave loop
            if density[k]<thresh:
    
                pp_type = 3
                pp_L = Lstar_stamp
                pp_time = fpe_epoch[k]
                pp_MLT = MLT_stamp
                pp_AE = AE_stamp
                pp_AEStar = AE_star
                break
            
            # otherwie, have never crossed a boundary and still in plasmapause so set crossing type to 4
            else:

                pp_type = 4
                pp_L = np.nan
                pp_time = np.nan
                pp_MLT = MLT_stamp
                pp_AE = AE_stamp
                pp_AEStar = AE_star

        return pp_type,pp_L,pp_time,pp_MLT,pp_AE,pp_AEStar


    # first, look for density gradient

    # find the index correpsonding to the apogee and perigee in the density data
    perigee = input_dict["all_times"][gee_index]
    apogee = input_dict["all_times"][gee_index+1]
    
    # find indicies for apogee and perigee in density data
    d_perigee = gl.find_closest(input_dict["fpe_epoch"],perigee)[1]
    d_apogee = gl.find_closest(input_dict["fpe_epoch"],apogee)[1]

    # set initial values of plasmapause crossing details to np.nan
    pp_L = np.nan
    pp_type = 0
    pp_time = np.nan
    pp_AE = np.nan
    pp_AEstar = np.nan
    pp_MLT = np.nan

    # set initial values for in case we have multiple methods (ECH) that work on a given half orbit, and want to save them for comaprison
    # set the plasmapause crossings details
    pp_L_ech = np.nan
    pp_AE_ech = np.nan
    pp_AEstar_ech =np.nan
    pp_MLT_ech = np.nan
    pp_time_ech = np.nan

    # set initial values for in case we have multiple methods (thresh) that work on a given half orbit, and want to save them for comaprison
    # set the plasmapause crossings details
    pp_L_thresh = np.nan
    pp_AE_thresh = np.nan
    pp_AEstar_thresh =np.nan
    pp_MLT_thresh = np.nan
    pp_time_thresh = np.nan

    # set flags for a crossing due to ECH waves and the density threshold as not found (False)
    stopECH = False
    stopThresh = False
    
    if (d_apogee-d_perigee)<2:
        # find the hfr spectra indicies correaponding to perigee and apogee
        d_spectra_perigee = gl.find_closest(input_dict["hfr_epoch"],perigee)[1]
        d_spectra_apogee = gl.find_closest(input_dict["hfr_epoch"],apogee)[1]

        for k in range(d_spectra_perigee,d_spectra_apogee):


            if find_ech(input_dict["fce"],input_dict["fce_epoch"],input_dict["Etotal"],input_dict["hfr_E_reduced"],input_dict["survey_epoch"],input_dict["hfr_epoch"],input_dict["survey_freq"],input_dict["hfr_frequency"],input_dict["hfr_epoch"][k]) > 1e-3:
               
                # set crossing type
                pp_type = 2
               
                # find corresponding L, MLT, AE, AE*
                findLstar=fp.FindLANLFeatures(input_dict["hfr_epoch"][k], input_dict["lanl_epoch"], input_dict["MLT"], input_dict["MLAT_N"], input_dict["MLAT_S"], input_dict["Lstar"])
                Lstar_stamp = findLstar.get_Lstar
                MLT_stamp = findLstar.get_MLT   
                findOmniFeats = fp.FindOMNIFeatures(input_dict["hfr_epoch"][k],input_dict["epoch_omni"], input_dict["epoch_omni_low"], input_dict["AE"], input_dict["Kp"], input_dict["Dst"])
                AE_stamp = findOmniFeats.get_AE
                AE_star = findOmniFeats.get_AEstar

                # set the plasmapause crossings details
                pp_L = Lstar_stamp
                pp_AE = AE_stamp
                pp_AEstar = AE_star
                pp_MLT = MLT_stamp
                pp_time = input_dict["hfr_epoch"][k]

                #print("The integral is:",find_ech(input_dict["fce"],input_dict["fce_epoch"],input_dict["Etotal"],input_dict["hfr_E_reduced"],input_dict["survey_epoch"],input_dict["hfr_epoch"],input_dict["survey_freq"],input_dict["hfr_frequency"],input_dict["hfr_epoch"][k]))
                
                # set the ECH wave criteria flag to found (True) so that we know not to keep checking with the threshold method
                stopECH=True
                break
    

    # cycle through density data
    for i in range(d_perigee,d_apogee):


        # find the time difference between adjacent density values in order to check for gaps 
        # if there are gaps, we will have to call the ECH wave checker

        dt_density = input_dict["fpe_epoch"][i+1] - input_dict["fpe_epoch"][i]

        # find the Lstar, MLT
        findLstar= fp.FindLANLFeatures(input_dict["fpe_epoch"][i], input_dict["lanl_epoch"], input_dict["MLT"], input_dict["MLAT_N"], input_dict["MLAT_S"], input_dict["Lstar"])
        Lstar_stamp = findLstar.get_Lstar
        Lstar_index = findLstar.index()
        MLT_stamp = findLstar.get_MLT 

        # find the AE and AE* values
        findOmniFeats = fp.FindOMNIFeatures(input_dict["fpe_epoch"][i],input_dict["epoch_omni"], input_dict["epoch_omni_low"], input_dict["AE"], input_dict["Kp"], input_dict["Dst"])
        AE_stamp = findOmniFeats.get_AE
        AE_star = findOmniFeats.get_AEstar

        # find Lstar+0.5 for density gradient check
        Lstar_plus = Lstar+0.5
        findPlus = fp.FindDensity(input_dict["Lstar"], Lstar_plus, Lstar_index, input_dict["lanl_epoch"], input_dict["fpe_epoch"])
        d_plusIndex = findPlus.index()
        Lstar_plusIndex = findPlus.find_closest_L()[1]
        

        # If problem: density is an insane value (negative or massssive), or there is a data gap - check ECH or set to no method worked
        # Otherwise, YAY find the gradient!!
        if (np.isnan(input_dict["density"][i])==False) and (np.isnan(d_plusIndex)==False) and (dt_density<timedelta(seconds=600)) and (input_dict["fpe_epoch"][i]<apogee):
            
            gradient = input_dict["density"][d_plusIndex]/input_dict["density"][i]
        

            if (gradient<1/5)and(input_dict["density"][d_plusIndex]<300):
                
                # set flag to gradient for that orbit
                pp_type = 1
                
                findOmniFeats = fp.FindOMNIFeatures(input_dict["fpe_epoch"][d_plusIndex],input_dict["epoch_omni"], input_dict["epoch_omni_low"], input_dict["AE"], input_dict["Kp"], input_dict["Dst"])
                AE_stamp = findOmniFeats.get_AE
                AE_star = findOmniFeats.get_AEstar

                # set the plasmapause crossing details as the L+0.5 point (definitley in plasmatrough)
                pp_L = input_dict["Lstar"][Lstar_plusIndex]
                pp_time = input_dict["fpe_epoch"][d_plusIndex]
                pp_AE = AE_stamp
                pp_AEstar = AE_star
                pp_MLT = MLT_stamp

                # stop looking as gradient found!

                # find what the correspndong density thresh Lpp location would be

                pp_thresh,pp_L_thresh,pp_time_thresh,pp_MLT_thresh,pp_AE_thresh,pp_AEstar_thresh = density_thresh(input_dict["fpe_epoch"], input_dict["lanl_epoch"], input_dict["MLT"], input_dict["MLAT_N"], input_dict["MLAT_S"], input_dict["Lstar"],input_dict["density"],d_perigee,d_apogee,input_dict["epoch_omni"],input_dict["AE"],input_dict["epoch_omni_low"],input_dict["Dst"],input_dict["Kp"])

                # find what the correspndong ech wave Lpp location would be
                # find the hfr spectra indicies correaponding to perigee and apogee
                d_spectra_perigee = gl.find_closest(input_dict["hfr_epoch"],perigee)[1]
                d_spectra_apogee = gl.find_closest(input_dict["hfr_epoch"],apogee)[1]
                count=0
                last=True
                for k in range(d_spectra_perigee,d_spectra_apogee):
                    #print(input_dict["hfr_epoch"][k],find_ech(input_dict["fce"],input_dict["fce_epoch"],input_dict["Etotal"],input_dict["hfr_E_reduced"],input_dict["survey_epoch"],input_dict["hfr_epoch"],input_dict["survey_freq"],input_dict["hfr_frequency"],input_dict["hfr_epoch"][k]))
                    
                    if find_ech(input_dict["fce"],input_dict["fce_epoch"],input_dict["Etotal"],input_dict["hfr_E_reduced"],input_dict["survey_epoch"],input_dict["hfr_epoch"],input_dict["survey_freq"],input_dict["hfr_frequency"],input_dict["hfr_epoch"][k]) > 1e-3:
                        
                        if last == True:
                            count+=1 
                        
                        if stopECH == False:
                          
                            print(apogee,input_dict["hfr_epoch"][k],find_ech(input_dict["fce"],input_dict["fce_epoch"],input_dict["Etotal"],input_dict["hfr_E_reduced"],input_dict["survey_epoch"],input_dict["hfr_epoch"],input_dict["survey_freq"],input_dict["hfr_frequency"],input_dict["hfr_epoch"][k]))
                            # find corresponding L, MLT, AE, AE*
                            findLstar_ech=fp.FindLANLFeatures(input_dict["hfr_epoch"][k], input_dict["lanl_epoch"], input_dict["MLT"], input_dict["MLAT_N"], input_dict["MLAT_S"], input_dict["Lstar"])
                            Lstar_stamp_ech = findLstar_ech.get_Lstar
                            MLT_stamp_ech = findLstar_ech.get_MLT   
                            findOmniFeats_ech = fp.FindOMNIFeatures(input_dict["hfr_epoch"][k],input_dict["epoch_omni"], input_dict["epoch_omni_low"], input_dict["AE"], input_dict["Kp"], input_dict["Dst"])
                            AE_stamp_ech = findOmniFeats_ech.get_AE
                            AE_star_ech = findOmniFeats_ech.get_AEstar

                            # set the plasmapause crossings details
                            pp_L_ech = Lstar_stamp_ech
                            pp_AE_ech = AE_stamp_ech
                            pp_AEstar_ech = AE_star_ech
                            pp_MLT_ech = MLT_stamp_ech
                            pp_time_ech = input_dict["hfr_epoch"][k]
                            stopECH=True
                            last = True
                            
                        
                        last = True
                    
                        if count>5:
                            break
                    else:
                        last=False
                        count=0

                # ECH waves mist persist, cannot be rogue power single sample 
                if count<5:
                    pp_L_ech = np.nan
                    pp_AE_ech = np.nan
                    pp_AEstar_ech =np.nan
                    pp_MLT_ech = np.nan
                    pp_time_ech = np.nan
                
                break
        
    if pp_type!=1:
        for i in range(d_perigee,d_apogee): 
            if (np.isnan(input_dict["density"][i])) or (np.isnan(d_plusIndex)) or (dt_density>timedelta(seconds=600)) or (input_dict["fpe_epoch"][i]>apogee):
                
                # put in the ECH criteria!!

                # find the hfr spectra indicies correaponding to perigee and apogee
                d_spectra_perigee = gl.find_closest(input_dict["hfr_epoch"],perigee)[1]
                d_spectra_apogee = gl.find_closest(input_dict["hfr_epoch"],apogee)[1]

                for k in range(d_spectra_perigee,d_spectra_apogee):
                    if stopECH == True:
                        break

                    if find_ech(input_dict["fce"],input_dict["fce_epoch"],input_dict["Etotal"],input_dict["hfr_E_reduced"],input_dict["survey_epoch"],input_dict["hfr_epoch"],input_dict["survey_freq"],input_dict["hfr_frequency"],input_dict["hfr_epoch"][k]) > 1e-3:
                            
                            # set crossing type
                            pp_type = 2
                            
                            # find corresponding L, MLT, AE, AE*
                            findLstar=fp.FindLANLFeatures(input_dict["hfr_epoch"][k], input_dict["lanl_epoch"], input_dict["MLT"], input_dict["MLAT_N"], input_dict["MLAT_S"], input_dict["Lstar"])
                            Lstar_stamp = findLstar.get_Lstar
                            MLT_stamp = findLstar.get_MLT   
                            findOmniFeats = fp.FindOMNIFeatures(input_dict["hfr_epoch"][k],input_dict["epoch_omni"], input_dict["epoch_omni_low"], input_dict["AE"], input_dict["Kp"], input_dict["Dst"])
                            AE_stamp = findOmniFeats.get_AE
                            AE_star = findOmniFeats.get_AEstar

                            # set the plasmapause crossings details
                            pp_L = Lstar_stamp
                            pp_AE = AE_stamp
                            pp_AEstar = AE_star
                            pp_MLT = MLT_stamp
                            pp_time = input_dict["hfr_epoch"][k]

                            print("The integral is:",find_ech(input_dict["fce"],input_dict["fce_epoch"],input_dict["Etotal"],input_dict["hfr_E_reduced"],input_dict["survey_epoch"],input_dict["hfr_epoch"],input_dict["survey_freq"],input_dict["hfr_frequency"],input_dict["hfr_epoch"][k]))
                            
                            # set the ECH wave criteria flag to found (True) so that we know not to keep checking with the threshold method
                            stopECH=True
                    
                # if we go over the range, or haven't found a crossing, use threshold method and then leave
                if (pp_type!=2)and(stopThresh==False) :

                    # set the stopECH to True so that we do not check again on the next density value pass if already checked
                    
                    stopECH = True
                    pp_type,pp_L,pp_time,pp_MLT,pp_AE,pp_AEstar = density_thresh(input_dict["fpe_epoch"], input_dict["lanl_epoch"], input_dict["MLT"], input_dict["MLAT_N"], input_dict["MLAT_S"], input_dict["Lstar"],input_dict["density"],d_perigee,d_apogee,input_dict["epoch_omni"],input_dict["AE"],input_dict["epoch_omni_low"],input_dict["Dst"],input_dict["Kp"])
                    stopThresh = True

            # if the l+0.5 goes beyond the half orbit, again check ECH waves if not already, and then check threshold if not already   
            elif (input_dict["lanl_epoch"][Lstar_plusIndex]>apogee):
                d_spectra_perigee = gl.find_closest(input_dict["hfr_epoch"],perigee)[1]
                d_spectra_apogee = gl.find_closest(input_dict["hfr_epoch"],apogee)[1]
                # put in the ECH criteria!!
                for k in range(d_spectra_perigee,d_spectra_apogee):
                    if stopECH == True:
                        break
                    
                    if find_ech(input_dict["fce"],input_dict["fce_epoch"],input_dict["Etotal"],input_dict["hfr_E_reduced"],input_dict["survey_epoch"],input_dict["hfr_epoch"],input_dict["survey_freq"],input_dict["hfr_frequency"],input_dict["hfr_epoch"][k]) > 1e-3:
                            
                            pp_type = 2
                            # find corresponding L, MLT, AE, AE*
                            findLstar=fp.FindLANLFeatures(input_dict["hfr_epoch"][k], input_dict["lanl_epoch"], input_dict["MLT"], input_dict["MLAT_N"], input_dict["MLAT_S"], input_dict["Lstar"])
                            Lstar_stamp = findLstar.get_Lstar
                            MLT_stamp = findLstar.get_MLT   

                            findOmniFeats = fp.FindOMNIFeatures(input_dict["hfr_epoch"][k],input_dict["epoch_omni"], input_dict["epoch_omni_low"], input_dict["AE"], input_dict["Kp"], input_dict["Dst"])
                            AE_stamp = findOmniFeats.get_AE
                            AE_star = findOmniFeats.get_AEstar

                            # set plasmapause crossing details
                            pp_L = Lstar_stamp
                            pp_AE = AE_stamp
                            pp_AEstar = AE_star
                            pp_MLT = MLT_stamp
                            pp_time = input_dict["hfr_epoch"][k]

                            # set ECH flag to True
                            stopECH=True

            
                # if we go over the range, use threshold method and then leave
                if (pp_type!=2)and(stopThresh==False) :
                
                    stopECH = True
                    pp_type,pp_L,pp_time,pp_MLT,pp_AE,pp_AEstar = density_thresh(input_dict["fpe_epoch"], input_dict["lanl_epoch"], input_dict["MLT"], input_dict["MLAT_N"], input_dict["MLAT_S"], input_dict["Lstar"],input_dict["density"],d_perigee,d_apogee,input_dict["epoch_omni"],input_dict["AE"],input_dict["epoch_omni_low"],input_dict["Dst"],input_dict["Kp"])
                    stopThresh = True
                

        
        

    if pp_type == 0:
        # resort back to density threshold one final time, if none of the other options worked
        pp_type,pp_L,pp_time,pp_MLT,pp_AE,pp_AEstar = density_thresh(input_dict["fpe_epoch"], input_dict["lanl_epoch"], input_dict["MLT"], input_dict["MLAT_N"], input_dict["MLAT_S"], input_dict["Lstar"],input_dict["density"],d_perigee,d_apogee,input_dict["epoch_omni"],input_dict["AE"],input_dict["epoch_omni_low"],input_dict["Dst"],input_dict["Kp"])
            
    print("peri ",gee_index," - apo ",gee_index+1, "outwards done")

    return pp_type,pp_L,pp_time,pp_MLT,pp_AE, pp_AEstar, pp_L_ech,pp_time_ech,pp_MLT_ech,pp_AE_ech, pp_AEstar_ech, pp_L_thresh,pp_time_thresh,pp_MLT_thresh,pp_AE_thresh, pp_AEstar_thresh


def inwards_journ(input_dict, gee_index):

    def density_thresh(fpe_epoch, lanl_epoch, MLT, MLAT_N, MLAT_S, Lstar,density,d_apogee,d_perigee,epoch_omni,AE,epoch_omni_low,Dst,Kp):
    # resort back to density threshold
   
        # first, make sure that the spacecraft is 'leaving' the plasmapause - otherwise we are always out!
        findLstar= fp.FindLANLFeatures(fpe_epoch[d_apogee], lanl_epoch, MLT, MLAT_N, MLAT_S, Lstar)
        Lstar_stamp = findLstar.get_Lstar
        MLT_stamp = findLstar.get_MLT   
        findOmniFeats = fp.FindOMNIFeatures(fpe_epoch[d_apogee],epoch_omni, epoch_omni_low, AE, Kp, Dst)
        AE_stamp = findOmniFeats.get_AE
        AE_star = findOmniFeats.get_AEstar


        # find the density threshold - largest between 50 and empirical L relationship for threshold
        thresh = np.max([50,10*(6.6/Lstar_stamp)**4])

        # set initial values of plasmapause crossing details to np.nan
        pp_type = None
        pp_L = np.nan
        pp_time = np.nan
        pp_MLT = np.nan
        pp_AE = np.nan
        pp_AEstar = np.nan

    

        # if at the beginning of the half orbit we are not already in the plasmasphere
        if density[d_apogee]<thresh:
            
            # loop through half orbit densities
            for k in range(d_apogee+1,d_perigee):

                # find correspinding Lstar,MLT value
                findLstar= fp.FindLANLFeatures(fpe_epoch[k], lanl_epoch, MLT, MLAT_N, MLAT_S, Lstar)
                Lstar_stamp = findLstar.get_Lstar
                MLT_stamp = findLstar.get_MLT

                # find AE and AE* values
                findOmniFeats = fp.FindOMNIFeatures(fpe_epoch[k],epoch_omni, epoch_omni_low, AE,Kp,Dst)
                AE_stamp = findOmniFeats.get_AE
                AE_star = findOmniFeats.get_AEstar
        
                # find the density threshold - largest between 50 and empirical L relationship for threshold
                thresh = np.max([50,10*(6.6/Lstar_stamp)**4])
        
                # if current density above threshold, have found plasmapause and therefore update crossing details and leave loop
                if density[k]>thresh:
        
                    pp_type = 3
                    pp_L = Lstar_stamp
                    pp_time = fpe_epoch[k]
                    pp_AE = AE_stamp
                    pp_AEstar = AE_star
                    pp_MLT = MLT_stamp
                    break
                
                # otherwise, not found :(
                else:

                    pp_type = 4.
                    pp_L = np.nan
                    pp_time = np.nan
                    pp_AE = np.nan
                    pp_AEstar = np.nan
                    pp_MLT = np.nan

        else:
            # always in plasmasphere on this orbit!!
            pp_type = 4
            pp_L = np.nan
            pp_time = np.nan
            pp_AE = np.nan
            pp_AEstar = np.nan
            pp_MLT = np.nan

        return pp_type,pp_L,pp_time,pp_MLT,pp_AE,pp_AEstar


    # first, look for density gradient

    # find the index correpsonding to the apogee and perigee in the density data
    apogee = input_dict["all_times"][gee_index]
    perigee = input_dict["all_times"][gee_index+1]
    
    d_perigee = gl.find_closest(input_dict["fpe_epoch"],perigee)[1]
    d_apogee = gl.find_closest(input_dict["fpe_epoch"],apogee)[1]

     # set the plasmapause crossing initial values to np.nan
    pp_L = np.nan
    pp_type = 0
    pp_time = np.nan
    pp_AE = np.nan
    pp_AEstar = np.nan
    pp_MLT = np.nan

    # set initial values for in case we have multiple methods (ECH) that work on a given half orbit, and want to save them for comaprison
    # set the plasmapause crossings details
    pp_L_ech = np.nan
    pp_AE_ech = np.nan
    pp_AEstar_ech =np.nan
    pp_MLT_ech = np.nan
    pp_time_ech = np.nan

    # set initial values for in case we have multiple methods (thresh) that work on a given half orbit, and want to save them for comaprison
    # set the plasmapause crossings details
    pp_L_thresh = np.nan
    pp_AE_thresh = np.nan
    pp_AEstar_thresh =np.nan
    pp_MLT_thresh = np.nan
    pp_time_thresh = np.nan

    # set flags for a crossing due to ECH waves and the density threshold as not found (False)
    stopECH = False
    stopThresh = False
    

    if ((d_perigee-d_apogee)<2) or (input_dict["fpe_epoch"][d_perigee]-perigee<timedelta(seconds=60)):

       
        # find the hfr spectra indicies correaponding to perigee and apogee
        d_spectra_perigee = gl.find_closest(input_dict["hfr_epoch"],perigee)[1]
        d_spectra_apogee = gl.find_closest(input_dict["hfr_epoch"],apogee)[1]

        for k in range(d_spectra_apogee,d_spectra_perigee):


            if find_ech(input_dict["fce"],input_dict["fce_epoch"],input_dict["Etotal"],input_dict["hfr_E_reduced"],input_dict["survey_epoch"],input_dict["hfr_epoch"],input_dict["survey_freq"],input_dict["hfr_frequency"],input_dict["hfr_epoch"][k]) > 1e-3:
                
                # set crossing type
                pp_type = 2
                
                # find corresponding L, MLT, AE, AE*
                findLstar=fp.FindLANLFeatures(input_dict["hfr_epoch"][k], input_dict["lanl_epoch"], input_dict["MLT"], input_dict["MLAT_N"], input_dict["MLAT_S"], input_dict["Lstar"])
                Lstar_stamp = findLstar.get_Lstar
                MLT_stamp = findLstar.get_MLT   
                findOmniFeats = fp.FindOMNIFeatures(input_dict["hfr_epoch"][k],input_dict["epoch_omni"], input_dict["epoch_omni_low"], input_dict["AE"], input_dict["Kp"], input_dict["Dst"])
                AE_stamp = findOmniFeats.get_AE
                AE_star = findOmniFeats.get_AEstar

                # set the plasmapause crossings details
                pp_L = Lstar_stamp
                pp_AE = AE_stamp
                pp_AEstar = AE_star
                pp_MLT = MLT_stamp
                pp_time = input_dict["hfr_epoch"][k]

                #print("The integral is:",find_ech(input_dict["fce"],input_dict["fce_epoch"],input_dict["Etotal"],input_dict["hfr_E_reduced"],input_dict["survey_epoch"],input_dict["hfr_epoch"],input_dict["survey_freq"],input_dict["hfr_frequency"],input_dict["hfr_epoch"][k]),input_dict["hfr_epoch"][k])
            
            #elif all_times[1]<input_dict["hfr_epoch"][k]<all_times[2]:
                #print(input_dict["hfr_epoch"][k])
                # set the ECH wave criteria flag to found (True) so that we know not to keep checking with the threshold method
               
    print("The apogee of this inward orbit is", apogee, 'and the perigee is', perigee)

   
    # cycle through the density data fro this half orbit
    for i in range(d_apogee,d_perigee):
        
        
        # find the time difference between adjacent density values in order to check for gaps 
        # if there are gaps, we will have to call the ECH wave checker
        dt_density = input_dict["fpe_epoch"][i+1] - input_dict["fpe_epoch"][i]
        
        # find the Lstar, and Lstar+0.5
        findLstar= fp.FindLANLFeatures(input_dict["fpe_epoch"][i], input_dict["lanl_epoch"], input_dict["MLT"], input_dict["MLAT_N"], input_dict["MLAT_S"], input_dict["Lstar"])
        Lstar_stamp = findLstar.get_Lstar
        Lstar_index = findLstar.index()
        MLT_stamp = findLstar.get_MLT
        
        # find AE and AE*
        findOmniFeats = fp.FindOMNIFeatures(input_dict["fpe_epoch"][i],input_dict["epoch_omni"],input_dict["epoch_omni_low"], input_dict["AE"], input_dict["Kp"], input_dict["Dst"])
        AE_stamp = findOmniFeats.get_AE
        AE_star = findOmniFeats.get_AEstar

        # find the Lstar+0.5 for the gradient calculation
        Lstar_plus = Lstar+0.5
        findPlus = fp.FindDensity(input_dict["Lstar"], Lstar_plus, Lstar_index, input_dict["lanl_epoch"], input_dict["fpe_epoch"])
        d_plusIndex = findPlus.index()
        Lstar_plusIndex = findPlus.find_closest_L()[1]
        
        
        gradient = 0

        # Otherwise, YAY find the gradient!!
        if (np.isnan(input_dict["density"][i])==False) and (np.isnan(input_dict["density"][d_plusIndex])==False) and (input_dict["lanl_epoch"][Lstar_plusIndex]<perigee) and (dt_density<timedelta(seconds=10)) and (input_dict["fpe_epoch"][i]<perigee):

            gradient = input_dict["density"][d_plusIndex]/input_dict["density"][i]

            # making sure that we defo end up inside the plasmasphere (no funny buinsess!!)
            if (gradient>5)and(input_dict["density"][d_plusIndex]>50):
                
                # find 

                pp_AEstar = AE_star
                pp_type = 1
                pp_L = Lstar_stamp
                pp_MLT =MLT_stamp
                pp_time = input_dict["fpe_epoch"][i]
                pp_AE = AE_stamp

                # find the alternative methods for this gradient: threshold
                pp_thresh,pp_L_thresh,pp_time_thresh,pp_MLT_thresh,pp_AE_thresh,pp_AEstar_thresh = density_thresh(input_dict["fpe_epoch"], input_dict["lanl_epoch"], input_dict["MLT"], input_dict["MLAT_N"], input_dict["MLAT_S"], input_dict["Lstar"],input_dict["density"],d_apogee,d_perigee,input_dict["epoch_omni"],input_dict["AE"],input_dict["epoch_omni_low"],input_dict["Dst"],input_dict["Kp"])

                # find the alternative methods for this gradient: ECH
                # find the hfr spectra indicies correaponding to perigee and apogee
                d_spectra_apogee = gl.find_closest(input_dict["hfr_epoch"],apogee)[1]
                d_spectra_perigee = gl.find_closest(input_dict["hfr_epoch"],perigee)[1]
                count = 0
                last = True
                # cycle through the HFR spectra for this orbit
                for k in range(d_spectra_apogee,d_spectra_perigee):
                    
                    if find_ech(input_dict["fce"],input_dict["fce_epoch"],input_dict["Etotal"],input_dict["hfr_E_reduced"],input_dict["survey_epoch"],input_dict["hfr_epoch"],input_dict["survey_freq"],input_dict["hfr_frequency"],input_dict["hfr_epoch"][k]) > 1e-3:
                            #print("ECH FOUND",find_ech(input_dict["fce"],input_dict["fce_epoch"],input_dict["Etotal"],input_dict["hfr_E_reduced"],input_dict["survey_epoch"],input_dict["hfr_epoch"],input_dict["survey_freq"],input_dict["hfr_frequency"],input_dict["hfr_epoch"][k]))
                            if last == True:
                                count+=1
                            

                            # find L, MLT, AE and AE*
                            findLstar_ech= fp.FindLANLFeatures(input_dict["hfr_epoch"][k], input_dict["lanl_epoch"], input_dict["MLT"], input_dict["MLAT_N"], input_dict["MLAT_S"], input_dict["Lstar"])
                            Lstar_stamp_ech  = findLstar_ech.get_Lstar
                            MLT_stamp_ech = findLstar_ech.get_MLT
                            findOmniFeats_ech = fp.FindOMNIFeatures(input_dict["hfr_epoch"][k],input_dict["epoch_omni"], input_dict["epoch_omni_low"], input_dict["AE"], input_dict["Kp"], input_dict["Dst"])
                            AE_stamp_ech = findOmniFeats_ech.get_AE
                            AE_star_ech = findOmniFeats_ech.get_AEstar

                            # set the plasmapause crossing details
                            pp_L_ech = Lstar_stamp_ech
                            pp_MLT_ech = MLT_stamp_ech
                            pp_AE_ech = AE_stamp_ech
                            pp_AEstar_ech = AE_star_ech
                            pp_time_ech = input_dict["hfr_epoch"][k]
                            last = True
                    else:
                        last = False

                if count<10:
                    pp_L_ech = np.nan
                    pp_AE_ech = np.nan
                    pp_AEstar_ech =np.nan
                    pp_MLT_ech = np.nan
                    pp_time_ech = np.nan      

                break

    # cycle through the density data fro this half orbit again checking ech waves and density threshold
    if pp_type!=1:
        for i in range(d_apogee,d_perigee):   
            # If problem: density is an insane value (negative or massssive), or there are density data gaps- check ECH or set to no method worked
            if pp_type==2:
                break
            if (np.isnan(input_dict["density"][i])) or (np.isnan(input_dict["density"][d_plusIndex])) or (input_dict["lanl_epoch"][Lstar_plusIndex]>perigee)or (dt_density>timedelta(seconds=10)) or (input_dict["fpe_epoch"][i]>perigee):
                
                # find the hfr spectra indicies correaponding to perigee and apogee
                d_spectra_apogee = gl.find_closest(input_dict["hfr_epoch"],apogee)[1]
                d_spectra_perigee = gl.find_closest(input_dict["hfr_epoch"],perigee)[1]
                
                # cycle through the HFR spectra for this orbit
                for k in range(d_spectra_apogee,d_spectra_perigee):
        
                    if find_ech(input_dict["fce"],input_dict["fce_epoch"],input_dict["Etotal"],input_dict["hfr_E_reduced"],input_dict["survey_epoch"],input_dict["hfr_epoch"],input_dict["survey_freq"],input_dict["hfr_frequency"],input_dict["hfr_epoch"][k]) > 1e-3:
                            pp_type = 2
                            #print("ECH FOUND",find_ech(input_dict["fce"],input_dict["fce_epoch"],input_dict["Etotal"],input_dict["hfr_E_reduced"],input_dict["survey_epoch"],input_dict["hfr_epoch"],input_dict["survey_freq"],input_dict["hfr_frequency"],input_dict["hfr_epoch"][k]))

                            # find L, MLT, AE and AE*
                            findLstar= fp.FindLANLFeatures(input_dict["hfr_epoch"][k], input_dict["lanl_epoch"], input_dict["MLT"], input_dict["MLAT_N"], input_dict["MLAT_S"], input_dict["Lstar"])
                            Lstar_stamp = findLstar.get_Lstar
                            MLT_stamp = findLstar.get_MLT
                            findOmniFeats = fp.FindOMNIFeatures(input_dict["hfr_epoch"][k],input_dict["epoch_omni"], input_dict["epoch_omni_low"], input_dict["AE"], input_dict["Kp"], input_dict["Dst"])
                            AE_stamp = findOmniFeats.get_AE
                            AE_star = findOmniFeats.get_AEstar

                            # set the plasmapause crossing details
                            pp_L = Lstar_stamp
                            pp_MLT = MLT_stamp
                            pp_AE = AE_stamp
                            pp_AEstar = AE_star
                            pp_time = input_dict["hfr_epoch"][k]
                            print(pp_time,"ECH found")   
                
                # if we go over the range, use threshold method and then leave
                if pp_type!=2:
                
                    pp_type,pp_L,pp_time,pp_MLT,pp_AE,pp_AEstar = density_thresh(input_dict["fpe_epoch"], input_dict["lanl_epoch"], input_dict["MLT"], input_dict["MLAT_N"], input_dict["MLAT_S"], input_dict["Lstar"],input_dict["density"],d_apogee,d_perigee,input_dict["epoch_omni"],input_dict["AE"],input_dict["epoch_omni_low"],input_dict["Dst"],input_dict["Kp"])
                    break

            elif np.isnan(d_plusIndex):

                # resort back to density threshold
                pp_type,pp_L,pp_time,pp_MLT,pp_AE,pp_AEstar= density_thresh(input_dict["fpe_epoch"], input_dict["lanl_epoch"], input_dict["MLT"], input_dict["MLAT_N"], input_dict["MLAT_S"], input_dict["Lstar"],input_dict["density"],d_apogee,d_perigee,input_dict["epoch_omni"],input_dict["AE"],input_dict["epoch_omni_low"],input_dict["Dst"],input_dict["Kp"])
            
        


    if pp_type == 0:
        # resort back to density threshold one final time, if none of the other options worked
        pp_type,pp_L,pp_time,pp_MLT,pp_AE,pp_AEstar= density_thresh(input_dict["fpe_epoch"], input_dict["lanl_epoch"], input_dict["MLT"], input_dict["MLAT_N"], input_dict["MLAT_S"], input_dict["Lstar"],input_dict["density"],d_apogee,d_perigee,input_dict["epoch_omni"],input_dict["AE"],input_dict["epoch_omni_low"],input_dict["Dst"],input_dict["Kp"])
            
    print("apo ",gee_index," - peri ",gee_index+1, "inwards done")

    return pp_type,pp_L,pp_time,pp_MLT,pp_AE,pp_AEstar,pp_L_ech,pp_time_ech,pp_MLT_ech,pp_AE_ech, pp_AEstar_ech, pp_L_thresh,pp_time_thresh,pp_MLT_thresh,pp_AE_thresh, pp_AEstar_thresh

def combine_lastonorbit_with_firstonnextorbit(day, last_orb_type):

    if last_orb_type=='a':
        orbit_type = 'inward'

    else:
        orbit_type = 'outward'

    return orbit_type,



""" Defining all global variables 
"""
''' Tell me the date range - start date and end date. Function will also return the number days this range spans '''
start_date,end_date,no_days = gl.what_dates("ASK!")


#with open("pp_compare_methodsV3.csv", "w", newline="") as f:

 #   writer = csv.writer(f)
  #  writer.writerow(["types", "times", "L", "InorOut", "MLT", "AE", "AE*","times_thresh", "L_thresh", "MLT_thresh", "AE_thresh", "AE*_thresh","times_ech", "L_ech", "MLT_ech", "AE_ech", "AE*_ech"])  # Header row

for single_day in (start_date + timedelta(n) for n in range(no_days)):

    types_pps = []
    L_pps = []
    times_pps = []
    in_or_out=[]
    AE_pps = []
    AEstar_pps =[]
    MLT_pps = []


    # make lists for 'alternate methods' when gradient is triggered for comparison: ECH
   
    L_pps_ech= []
    times_pps_ech = []
    AE_pps_ech = []
    AEstar_pps_ech =[]
    MLT_pps_ech = []

    # make lists for 'alternate methods' when gradient is triggered for comparison: thresh
   
    L_pps_thresh= []
    times_pps_thresh = []
    AE_pps_thresh = []
    AEstar_pps_thresh =[]
    MLT_pps_thresh = []

    """ create object for accessing all files for given day """
    day_files = gl.DataFiles(single_day)
    # String version of dates for filename  
    date_string,year,month,day =gl.get_date_string(single_day)

    """ Create the OMNI dataset """
    omni_dataset = gl.omni_dataset(single_day-timedelta(days=1),single_day+timedelta(days=2))
    AE, epoch_omni = omni_dataset.omni_stats
    Kp, Dst, epoch_omni_low = omni_dataset.omni_stats_low_res

    """ Getting survey file and accessing survey frequencies, epoch and magnitude """
    survey_file = pycdf.CDF(day_files.survey_data)
    survey_data = gl.AccessSurveyAttrs(survey_file)

    """ get survey properties """
    survey_freq = survey_data.frequency
    survey_epoch = survey_data.epoch_convert()
    survey_bins = survey_data.bin_edges
    Btotal = survey_data.Bmagnitude
    Etotal = survey_data.Emagnitude

    Ewfr_reduced = fa.BackReduction(Etotal, 'EW', False)


    """ getting magnetometer data """
    # get the density, LANL, and magentometer data
    # Getting the magnetometer data
    mag_file = pycdf.CDF(day_files.magnetic_data)
    mag_data = gl.AccessL3Attrs(mag_file)

    # flag for when we have absolutley no density data
    zero_densdat = False
    if day_files.l4_data()[1] == False:
        print("OH NO - no density data for:",single_day)
        zero_densdat = True
        #continue
    fce, fce_05, fce_005,f_lhr = mag_data.f_ce
    fce_epoch = mag_data.epoch_convert()

    """ Getting survey HFR file and accessing frequencies, epoch and E magnitude """
    hfr_data =  pycdf.CDF(day_files.hfr_data)
    hfr_epoch = gl.AccessHFRAttrs(hfr_data).epoch_convert()
    hfr_frequency = gl.AccessHFRAttrs(hfr_data).frequency
    hfr_E = gl.AccessHFRAttrs(hfr_data).Emagnitude
    hfr_E_reduced = fa.HFRback(hfr_E)

    """ Getting the density data """
    if zero_densdat==False:
        density_file = pycdf.CDF(day_files.l4_data()[0])
        density_data = gl.AccessL4Attrs(density_file)
        """ get all densities - including in/out plasmapause flag """

        fpe_uncleaned = density_data.f_pe
        fpe_epoch = density_data.epoch_convert()
        density_uncleaned= density_data.density

    
    else:

        # Generate times
        time_list = []
        # Given time start
        time_start = time(0, 0, 0)
        # Given time end
        time_end = time(23, 59, 54)
        # Combine date and time
        date_with_time_start = datetime.combine(single_day, time_start)
        # Combine date and time
        date_with_time_end = datetime.combine(single_day, time_end)
        # Define interval (e.g., every 30 minutes)
        interval = timedelta(seconds=6)

        current_time = date_with_time_start
        while current_time <= date_with_time_end:
            time_list.append(current_time)  # Convert to string if needed
            current_time += interval

        print("Length of no density data created time list",len(time_list))
        density_uncleaned = np.zeros(14400)
        density_uncleaned[:] = -1
        fpe_uncleaned = np.zeros(14400)
        fpe_uncleaned[:] = -1
        fpe_epoch = time_list

    """ Getting the LANL data """
    lanl_file = h5py.File(day_files.lanl_data)
    lanl_data = gl.AccessLANLAttrs(lanl_file)
    # Getting LANL attributes
    apogee, perigee = lanl_data.apogee_perigee
    Lstar = lanl_data.L_star
    MLT = lanl_data.MLT
    MLAT_N, MLAT_S = lanl_data.MLAT_N_S
    lanl_epoch = lanl_data.epoch

  


    # Find what the first and last half orbits begin with 
    all_times = np.concatenate((apogee,perigee))

    # Creating corresponding labels
    labels = np.array(['a'] * len(apogee) + ['p'] * len(perigee))
    
    # Sort dates and labels together
    sorted_pairs = sorted(zip(all_times, labels))

    # Unzip the sorted pairs
    all_times, labels = zip(*sorted_pairs)
   
    # Finding the label for the maximum value
    max_index,min_index= np.argmax(all_times),np.argmin(all_times)  # Index of the max value
    max_label,min_label= labels[max_index],labels[min_index]  # Corresponding label
    print(max_index)
    
    if len(density_uncleaned)<2:
        # Generate times
        time_list = []
        # Given time start
        time_start = time(0, 0, 0)
        # Given time end
        time_end = time(23, 59, 54)
        # Combine date and time
        date_with_time_start = datetime.combine(single_day, time_start)
        # Combine date and time
        date_with_time_end = datetime.combine(single_day, time_end)
        # Define interval (e.g., every 30 minutes)
        interval = timedelta(seconds=6)

        current_time = date_with_time_start
        while current_time <= date_with_time_end:
            time_list.append(current_time)  # Convert to string if needed
            current_time += interval

        density_uncleaned = np.zeros(14400)
        density_uncleaned[:] = -1
        fpe_uncleaned = np.zeros(14400)
        fpe_uncleaned[:] = -1
        fpe_epoch = time_list

    # clean fpe array
    fpe = np.zeros((len(density_uncleaned)))
    fpe[:] = np.nan
    density = np.zeros((len(density_uncleaned)))
    density[:] = np.nan


    # Set all of the rogue density/fpe values as np.nan
    for i in range(len(density)):

        if fpe_uncleaned[i] <0.:
            density[i] = np.nan
            fpe[i] = np.nan

        else:
            density[i] = density_uncleaned[i]
            fpe[i] = fpe_uncleaned[i]

    input_dict0 = {'apogee':apogee,'perigee':perigee,
                  'fpe_epoch':fpe_epoch,'density':density,'fpe': fpe,
                  'MLT':MLT, 'MLAT_N':MLAT_N, 'MLAT_S': MLAT_S, 'Lstar': Lstar, 'lanl_epoch':lanl_epoch,
                  'all_times': all_times, 
                  'hfr_E_reduced':hfr_E_reduced,'hfr_epoch':hfr_epoch,'hfr_frequency':hfr_frequency,'Etotal':Ewfr_reduced,
                  'epoch_omni': epoch_omni, 'AE':AE, 'epoch_omni_low':epoch_omni_low, 'Dst':Dst, 'Kp':Kp,
                  'fce': fce, 'fce_epoch':fce_epoch,'survey_freq':survey_freq,'survey_epoch':survey_epoch}
    
    
    # first, do the last <half orbit / first <half orbit of current day
    if single_day>start_date:

        combined_dict = {}

        for key in input_dict0:
            print(f"{key}: {type(input_dict0[key])}")
            if isinstance(input_dict0[key], (np.ndarray)):
                combined_dict[f"{key}"] = np.concatenate((lastday_input_dict[key],input_dict0[key]))

            if isinstance(input_dict0[key],(list)):
                combined_dict[f"{key}"] = lastday_input_dict[key]+input_dict0[key]

        combined_dict["all_times"] = [lastday_input_dict["all_times"][-1],all_times[0]]
        

        if last_half_label == 'a':
            
        
            print("On inward journey for last < half orbit")
            
            types_pps.append(pp_type)
            pp_type, pp_L, pp_time, pp_MLT, pp_AE,pp_AEstar,pp_L_ech,pp_time_ech,pp_MLT_ech,pp_AE_ech, pp_AEstar_ech, pp_L_thresh,pp_time_thresh,pp_MLT_thresh,pp_AE_thresh, pp_AEstar_thresh = inwards_journ(combined_dict, 0)
            L_pps.append(pp_L)
            times_pps.append(pp_time)
            in_or_out.append('inward')
            AE_pps.append(pp_AE)
            MLT_pps.append(pp_MLT)
            AEstar_pps.append(pp_AEstar)

             # append ech alt
            L_pps_ech.append(pp_L_ech)
            times_pps_ech.append(pp_time_ech)
            AE_pps_ech.append(pp_AE_ech)
            AEstar_pps_ech.append(pp_AEstar_ech)
            MLT_pps_ech.append(pp_MLT_ech)

            # append ech alt
            L_pps_thresh.append(pp_L_thresh)
            times_pps_thresh.append(pp_time_thresh)
            AE_pps_thresh.append(pp_AE_thresh)
            AEstar_pps_thresh.append(pp_AEstar_thresh)
            MLT_pps_thresh.append(pp_MLT_thresh)


        else:

            print("On outward journey for last < half orbit")
            pp_type, pp_L, pp_time, pp_MLT, pp_AE, pp_AEstar,pp_L_ech,pp_time_ech,pp_MLT_ech,pp_AE_ech, pp_AEstar_ech, pp_L_thresh,pp_time_thresh,pp_MLT_thresh,pp_AE_thresh, pp_AEstar_thresh = outwards_journ(combined_dict, 0)
            print(pp_type, pp_L, pp_time)

            types_pps.append(pp_type)
            L_pps.append(pp_L)
            times_pps.append(pp_time)
            in_or_out.append('outward')
            AE_pps.append(pp_AE)
            AEstar_pps.append(pp_AEstar)
            MLT_pps.append(pp_MLT)

            # append ech alt
            L_pps_ech.append(pp_L_ech)
            times_pps_ech.append(pp_time_ech)
            AE_pps_ech.append(pp_AE_ech)
            AEstar_pps_ech.append(pp_AEstar_ech)
            MLT_pps_ech.append(pp_MLT_ech)

            # append ech alt
            L_pps_thresh.append(pp_L_thresh)
            times_pps_thresh.append(pp_time_thresh)
            AE_pps_thresh.append(pp_AE_thresh)
            AEstar_pps_thresh.append(pp_AEstar_thresh)
            MLT_pps_thresh.append(pp_MLT_thresh)

            print("ECH:", pp_L_ech, pp_time_ech, pp_AE_ech)
            print("Threshold:", pp_L_thresh, pp_time_thresh, pp_AE_thresh)

    for gee_index in range(len(all_times)-1):
        print(gee_index,labels[gee_index])
        if labels[gee_index] == 'a':
            print("On inward journey")
            pp_type, pp_L, pp_time, pp_MLT, pp_AE, pp_AEstar,pp_L_ech,pp_time_ech,pp_MLT_ech,pp_AE_ech, pp_AEstar_ech, pp_L_thresh,pp_time_thresh,pp_MLT_thresh,pp_AE_thresh, pp_AEstar_thresh = inwards_journ(input_dict0, gee_index)
            print(pp_type, pp_L, pp_time)
            types_pps.append(pp_type)
            L_pps.append(pp_L)
            times_pps.append(pp_time)
            in_or_out.append('inward')
            AE_pps.append(pp_AE)
            AEstar_pps.append(pp_AEstar)
            MLT_pps.append(pp_MLT)

            # append ech alt
            L_pps_ech.append(pp_L_ech)
            times_pps_ech.append(pp_time_ech)
            AE_pps_ech.append(pp_AE_ech)
            AEstar_pps_ech.append(pp_AEstar_ech)
            MLT_pps_ech.append(pp_MLT_ech)

            # append ech alt
            L_pps_thresh.append(pp_L_thresh)
            times_pps_thresh.append(pp_time_thresh)
            AE_pps_thresh.append(pp_AE_thresh)
            AEstar_pps_thresh.append(pp_AEstar_thresh)
            MLT_pps_thresh.append(pp_MLT_thresh)

        else: 
            print("On outward journey")
            pp_type, pp_L, pp_time,pp_MLT,pp_AE,pp_AEstar,pp_L_ech,pp_time_ech,pp_MLT_ech,pp_AE_ech, pp_AEstar_ech, pp_L_thresh,pp_time_thresh,pp_MLT_thresh,pp_AE_thresh, pp_AEstar_thresh= outwards_journ(input_dict0, gee_index)
            print(pp_type, pp_L, pp_time)
            types_pps.append(pp_type)
            L_pps.append(pp_L)
            times_pps.append(pp_time)
            in_or_out.append('outward')
            AE_pps.append(pp_AE)
            AEstar_pps.append(pp_AEstar)
            MLT_pps.append(pp_MLT)


            # append ech alt
            L_pps_ech.append(pp_L_ech)
            times_pps_ech.append(pp_time_ech)
            AE_pps_ech.append(pp_AE_ech)
            AEstar_pps_ech.append(pp_AEstar_ech)
            MLT_pps_ech.append(pp_MLT_ech)

            # append ech alt
            L_pps_thresh.append(pp_L_thresh)
            times_pps_thresh.append(pp_time_thresh)
            AE_pps_thresh.append(pp_AE_thresh)
            AEstar_pps_thresh.append(pp_AEstar_thresh)
            MLT_pps_thresh.append(pp_MLT_thresh)

            print("ECH:", pp_L_ech, pp_time_ech, pp_AE_ech)
            print("Threshold:", pp_L_thresh, pp_time_thresh, pp_AE_thresh)

    
    lastday_input_dict = input_dict0.copy()
    last_half_label = max_label

    print(single_day,"done")
   
    with open("pp_compare_methodsV3.csv", "a", newline="") as f:
        writer = csv.writer(f)
        ["types", "times", "L", "InorOut", "MLT", "AE", "AE*","times_thresh", "L_thresh", "MLT_thresh", "AE_thresh", "AE*_thresh","times_ech", "L_ech", "MLT_ech", "AE_ech", "AE*_ech"]
        for row in zip(types_pps, times_pps, L_pps, in_or_out, MLT_pps, AE_pps, AEstar_pps, times_pps_thresh, L_pps_thresh, MLT_pps_thresh, AE_pps_thresh, AEstar_pps_thresh,times_pps_ech, L_pps_ech, MLT_pps_ech, AE_pps_ech, AEstar_pps_ech):
            writer.writerow(row)
            print("appended half orb")




