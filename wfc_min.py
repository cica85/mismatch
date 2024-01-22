# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 10:58:10 2023

@author: QNanoLab
"""

import holo_tools as ht
import cal_tools as ct
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import scipy.ndimage as ndi
import pickle as pkl



filenames=['WFC_A_compound_.pkl','WFC_B_compound_.pkl']
far_field_centers=[(690,673),(526,682)]
in_camserial="13294"
IN_ROI=[557, 100, 494, 100, 1, 1]


_appvar={'px_to_um_CAM_IN': 0.4773,
         'init_scan_bin':1,
         'init_superpix_size':120,
         'input_mode_images':None,
         'mode_mask':None, 
         'slm_rise_time':180,
         'slm_screen_id':1,
         'out_exp_time': 10e-3,
         'in_exp_time': 1e-4}

def initialize(IN_ROI, far_field_center, in_camserial,  cams=None):
    
    # ref_holo=ht.hologram(farFieldCenter=ref_farFieldCenter)         
    if cams is None:
        in_cam=ht.cam_capture(camExp=_appvar['in_exp_time'], camSerial=in_camserial, ROI=IN_ROI)
        cams=[in_cam]
    else:
        in_cam=cams[0]
    
    
    cal=ct.calibration(None,core_size=None,
                       far_field_center=far_field_center,rise_time=_appvar['slm_rise_time'], 
                       screen_id_main=_appvar['slm_screen_id'])
    cal.in_cam=in_cam
    # cal.reference_holo=ref_holo     
    return cal, cams

def run_compound_WFC(cals,               #Calibration object for caltools (contains methods for hologram generation, capture and processing)
                     super_pix_size=80,  #Subdomain size
                     shift_step=10,      #Number of pixels to shift per step
                     total_steps=8,      #Number of compounded wfc runs
                     low_pass_thresh=10, #threshold for filtering
                     sigma=30, #gaussian filter parameter
                     filenames=['WFC_A_compound_.pkl','WFC_B_compound_.pkl'],#Results will be saved with these names
                     ps_shifts=4 #Phase shifts per step
                     ):
        WFC_A=None #VARIABLE FOR HOLDING WFC
        WFC_B=None        
        WFCs=[None,None]
        tags=['WFCA_SINGLE', 'WFCB_SINGLE']
        plt.close('all')
        image_center=None #if this is the first run the center of the frame gets calculated
        abort=False
        
        for k in np.arange(total_steps):
            
            shift=(k)*shift_step
            
            for q,ca in enumerate(cals):
                WFCs[q], image_center=ca.run_wfc(super_pix_size=super_pix_size,
                                                 phase_shifts=ps_shifts,
                                                 file_name=tags[q]+str(k)+'.pkl',
                                                 image_center=image_center, 
                                                 shift=shift, 
                                                 scan_us=None
                                                 )
            
                if WFCs[q] is None:
                    print('ERROR: WFC IS NONE')
                    break
                

            
                if k==0:    #DISPLAYS THE FIRST RUN FOR USER EVALUATION, 
                            #IF THE WINDOW IS CLOSED BY THE USER THE MEASUREMENT IS ABORTED
                    fig=plt.figure()
                    plt.subplot(1,2,1)
                    plt.imshow(np.abs(WFCs[q]))
                    plt.subplot(1,2,2)
                    plt.imshow(np.angle(WFCs[q]))
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    sleep(5)
                    
                if not(plt.fignum_exists(fig.number)):
                    abort=True
                    break
            if abort:
                break
                        
        if not(abort):
            if not(WFC_A is None) or not(WFC_B is None):
                #PLOT A SINGLE WFC AS AN EXAMPLE:
                plot_us=[]
                for q,ca in enumerate(cals):
                    WFC=ca.load_WFC(tags[q]+'1.pkl',lp_thresh=low_pass_thresh, nearest=False)             
                    interp=ca.WFC             
                    plot_us.append(WFC)
                    plot_us.append(interp)
             
                for k in np.arange(len(plot_us)):
                    plt.subplot(len(plot_us)%2,2,k+1)
                    plt.imshow(np.abs(plot_us[k]))
                    
                #GET COMPOUND CORRECTION:
                compound_WFC=[]
                WFC_single=[]
                
                for q,ca in enumerate(cals):
                    for k in np.arange(total_steps):  
                        WFC_single.append(ca.load_WFC(tags[q]+str(k)+'.pkl', nearest=True))                      
                        if k==0:
                            interp=ndi.gaussian_filter(ca.WFC,sigma)

                        else:
                            shift=(k)*shift_step
                            A=np.roll(ca.WFC,shift,0)
                            A=np.roll(A,shift,1)                       
                            interp=interp+ndi.gaussian_filter(A,sigma)
                    compound_WFC.append(interp)
                        
                # interp_A=ndi.gaussian_filter(np.real(interp_A),sigma)+1j*ndi.gaussian_filter(np.imag(interp_A),sigma)       
                # interp_B=ndi.gaussian_filter(np.real(interp_B),sigma)+1j*ndi.gaussian_filter(np.imag(interp_B),sigma)
                    
                plot_us=[WFC_single[0],interp[0],WFC_single[1],interp[1]]
                fig=plt.figure()
                for k in np.arange(4):
                    plt.subplot(2,2,k+1)
                    plt.imshow(np.angle(plot_us[k]))
                fig.suptitle('BEAM PHASE', fontsize=16)
        
                fig=plt.figure()
                for k in np.arange(4):
                    plt.subplot(2,2,k+1)
                    plt.imshow(np.abs(plot_us[k]))
                    
                fig.suptitle('BEAM AMPLITUDE', fontsize=16)
            
    
                for n in filenames:
                    file=open(n,'wb')
                    pkl.dump(interp[k],file)
                    k+=1
                    file.close()
                    
if __name__ == '__main__':
    cams=None
    cals=[]
    for fcenter in far_field_centers:
        c,cams=initialize(IN_ROI,fcenter, in_camserial,  cams)
        cals.append(c)        
    run_compound_WFC(cals, filenames=filenames)