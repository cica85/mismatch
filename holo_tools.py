# -*- coding: utf-8 -*-
"""
Version: 0.2 
Created on Tue Apr 11 2023

@author: QNanoLab
"""

import numpy as np
import matplotlib.pyplot as plt
import screeninfo
import copy
import scipy.ndimage as ndi
import cv2
import pylablib as pll
import pickle as pkl
import sys
import itertools as it


pll.par["devices/dlls/thorlabs_tlcam"] = "C:\64_lib"


from pylablib.devices import Thorlabs
from queue import Queue

from threading import Thread
from datetime import datetime
from time import sleep, time


import cupy as cp
import cupyx.scipy.fft as cufft
from cupyx.scipy.fft import get_fft_plan
from scipy import fft as ft
# ft.set_global_backend(cufft)
mempool = cp.get_default_memory_pool()
mempool.free_all_blocks()

import csv

import pyfftw
import multiprocess as mp

nthread = mp.cpu_count()
q_timeout=20
        
def FFT_process(Q):
    file=open('timing.csv','w')
    w=csv.writer(file)
    while True:
        try:
            qitem=Q[0].get(timeout=5) 
        except:
            w.writerow(['exit'])
            break
        if qitem is None:
            break
        start_time = time()
        M=qitem[0]
        # M=cp.random.random((1200,1920)).astype(cp.csingle)
        SLMSize=qitem[1]
        # F=np.zeros_like(M)
        # fft=pyfftw.FFTW(M, F, axes=(0,1), direction='FFTW_FORWARD', flags=('FFTW_MEASURE', ), 
        #         threads=15, planning_timelimit=None )
        # fft()
        plan = get_fft_plan(M, axes=(0, 1), value_type='C2C') 
        F=cufft.fft2(M, axes=(0, 1), plan=plan)
        sSize=SLMSize
        F=F[0:sSize[0],0:sSize[1]]
        F0=cp.asnumpy(F)
        del F
        Q[1].put(F0, timeout=10)
        t=time() - start_time
        w.writerow([t,])
    file.close()

class holo_player:
    
    def __init__(self, screen_id_main=1,main_name='hologram', DMD=False):
        """
        
        Creates a holo_player object, required for displaying the generated 
        holograms when multithreading.
        
        
        Parameters
        ----------
        screen_id_main : Screen id number (Default is 1)

        Returns
        -------
        None.

        """
        self.player_Q=Queue(maxsize=1) # A Queue object to comunicate with the player
        self.screen_main=screeninfo.get_monitors()[screen_id_main] #Screen information
        self.player_thread=None #The thread where the player runs is stored here
        self.main_name=main_name
        self.DMD=DMD
        
        return None
    

        
    def player_start(self):
        """
        Starts a player window

        Returns
        -------
        None.

        """
        if not(self.DMD):
            player=Thread(target=self.player_create,args=(self.player_Q, self.screen_main, ))
        else:
            player=Thread(target=self.player_DMD,args=(self.player_Q, self.screen_main, ))
            
        self.player_Q.put(np.random.rand(100,100))
        player.start() 
        self.player_thread=player
        return None
    
    def player_stop(self):
        """
        Stops the player and joins the thread

        Returns
        -------
        None.

        """
        self.player_Q.put(None)
        print('PLAYER Q NONE')
        self.player_thread.join(timeout=10)
    
    def player_DMD(self, Q,screen_main):
        
        prev_element=np.zeros((100,100))
        fail_count=0
        
        if not(Q is None):
            while True:             
                try:
                    element=Q.get(timeout=100) #To keep windows from detecting the app as frozen a timeout is implemented
                    fail_count=0
                except:
                    element=prev_element     #The last element displayed is stored in case of a timeout over the next iteration                                         
                    fail_count+=1
                if element is None:
                    print('ENDING PLAYER LOOP')
                    break
                if fail_count>50:
                    break
                
                element=np.ravel(element)
                ########INSERT DMD MODULE FUNCTIONS HERE##############
                
            
        print('PLAYER LOOP EXITED CORRECTLY')
        return None
                
    
    def player_create(self, Q,screen_main):
        """
        

        Parameters
        ----------
        Q :  Queue object used to comunicate with the player
        screen_main : Screen information

        Returns
        -------
        None.

        """
        import cv2 as cv #For OpenCV to work properly with multithreading it must be imported here

        # cv.destroyAllWindows() #Clear previous windows

    #Position the hologram window--------------------------------------------------
        main_name=self.main_name
        
        cv.namedWindow(main_name, cv.WND_PROP_FULLSCREEN)  
        cv.moveWindow(main_name, screen_main.x - 1, screen_main.y - 1)
        cv.setWindowProperty(main_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        prev_element=np.zeros((100,100))
        fail_count=0
    #Display elements sent through the Queue    
        if not(Q is None):
            while True:             
                try:
                    element=Q.get(timeout=100) #To keep windows from detecting the app as frozen a timeout is implemented
                    fail_count=0
                except:
                    element=prev_element     #The last element displayed is stored in case of a timeout over the next iteration                                         
                    fail_count+=1
                if element is None:
                    print('ENDING PLAYER LOOP')
                    break
                if fail_count>50:
                    break
                try:
                    cv.imshow(main_name,element) # Image is displayed
                    prev_element=element  #The last element displayed is stored in case of a timeout over the next iteration
                    cv.waitKey(1)
                except:
                    print('imshow failed')  
        cv.destroyWindow(main_name)
        cv.waitKey(1)
        del cv
        print('PLAYER LOOP EXITED CORRECTLY')
        return None

class SLM:
    def __init__(self, SLMSize=(1200,1920), screen_id=1, is_SANTEC=False, far_field_center=(712,1076)):
        
        self.SLMSize=np.array(SLMSize)
        self.screen_main=screeninfo.get_monitors()[screen_id]
        self.screen_id=screen_id
        self.SLM_offset=SLM.SLM_offset=0
        self.SLM_Max=SLM.SLM_Max=1
        self.is_SANTEC=SLM.is_SANTEC=is_SANTEC
        self.FF_center=far_field_center
        
        self.range_scan_result=None
        
    def slm_range_scan(self, cam=None, offset_lim=(0,0.5,0.1), max_lim=(0.4,1,0.1),rise_time=150):
        H=hologram(farFieldCenter=self.FF_center,SLMSize=self.SLMSize, screen_id_main=self.screen_id)
                                                                                    
        offset_points=np.arange(offset_lim[0],offset_lim[1]+offset_lim[2],offset_lim[2])
        max_points=np.arange(max_lim[0],max_lim[1]+max_lim[2],max_lim[2])
        cam.cam.open()
        cam.acqstart()
        
        res=np.zeros((len(max_points),len(offset_points)))
        km=0
        
        for m in max_points:
            kn=0
            for n in offset_points:
                if n<m:                    
                    H.SLM_Max=m
                    H.SLM_offset=n
                    H.display()                
                    frame=cam.wait_and_capture(rise_time).astype(np.float32)
                    res[km,kn]=np.sum(frame)                
                kn+=1
            km+=1
        cam.acqstop()
        cam.cam.close()
        
        self.range_scan_result=res
        
        ind=np.unravel_index(np.argmax(res),res.shape)
        self.SLM_Max=max_points[ind[0]]
        self.SLM_offset=offset_points[ind[1]]
        
        return res
    
    def slm_scaling_scan(self, cam, nholos=4,WFC=None):
        H=hologram(farFieldCenter=self.FF_center,SLMSize=self.SLMSize, SLM_offset=self.SLM_offset)
        FF_center=self.FF_center                       
        holosx=[hologram(farFieldCenter=(FF_center[0],FF_center[1]+k),SLMSize=self.SLMSize, SLM_offset=self.SLM_offset) for k in np.arange(-nholos,nholos)]
        holosy=[hologram(farFieldCenter=(FF_center[0]+k,FF_center[1]),SLMSize=self.SLMSize, SLM_offset=self.SLM_offset) for k in np.arange(-nholos,nholos)]
        
        cam.cam.open()
        cam.acqstart()
        
        
        frames=[self.disp_wait_capture(m, cam, WFC) for m in holosx]
        
            
        cam.acqstop()
        cam.cam.close()
    
        return frames
    
    def slm_random_aperture_scan(self, cam, nshifts=4,WFC=None, offset_max=None):
        
        FF_center=self.FF_center
        H=hologram(farFieldCenter=FF_center,SLMSize=self.SLMSize, SLM_offset=self.SLM_offset)                       
        
        
        cam.cam.open()
        cam.acqstart()
        
        aperture=self.random_SLM_aperture(apscale=80)
        naperture=np.abs(aperture-1)
        
        H0=H.holo_copy()
        H0.Holo=np.exp(1j*np.angle(H0.Holo)*aperture)        
        
        H1=H.holo_copy()
        H1.Holo=np.exp(1j*np.angle(H1.Holo)*naperture)
        
        holos=[self.add_holo(H0,H1.holoPhaseShift(nshifts, n)) for n in np.arange(nshifts)]
        
        orig=self.disp_wait_capture(H, cam, WFC)
        frames=[self.disp_wait_capture(m, cam, WFC, offset_max=offset_max) for m in holos]
        
        ind=np.unravel_index(np.argmax(orig[0]),orig[0].shape)
        
        mvalues=[]
        for k in frames:
            mvalues.append(k[0][ind[0],ind[1]])
            
        cam.acqstop()
        cam.cam.close()
    
        return frames, orig, mvalues
    
    def add_holo(self, H0, H1):
        H=H0.holo_copy()
        H.Holo=H0.Holo+H1.Holo
        return H
    
    def random_SLM_aperture(self, apscale=40):
        apsize=np.round(self.SLMSize/40).astype(np.int32)
        M=np.random.rand(apsize[0],apsize[1])
        M=(M>0.5).astype(np.int32)
        scale=self.SLMSize/apsize
        M=ndi.zoom(M,scale,order=0)
        return M
        
        
    def disp_wait_capture(self, holo, cam, WFC, offset_max=None):
        holo.display(WFC=WFC, offset_max=offset_max)
        frame=cam.wait_and_capture(200)
        thresh=0.5*(frame/np.max(frame))
        C=ndi.center_of_mass(thresh)
        return frame, C
    
class hologram:
    """
    hologram class
        Creates a hologram (complex field), and contains all the methods to 
        display it on the SLM
    """
    
    def __init__(self, precalc_holo=None, farFieldCenter=None, SLM=None, SLMSize=(1200,1920), screen_id_main = 1, 
                 SLM_offset=0, SLM_Max=1, FFT_Q=None):
        """
        

        Parameters
        ----------
        SLMSize : Shape like, optional
            Size of the SLM or screen. The default is (1920,1200), 
            default for a Thorlabs Exulus HD2 SLM.
        farFieldCenter : tupple or array of integers, optional
            Center location of the region of interest in the SLM farfield.
            If a center is given a blazed grating hologram is generated.
            The default is None.
        screen_id_main : SLM screen ID, optional
            The default is 1.
        SLM_offset : Float (faction of the gray level)
        An offset for the graylevel of the SLM, optional
            The default is 0.
        SLM_Max : Float (faction of the gray level), Maximum value that the 
        hologram can take as a fraction of the gray level
            The default is 1.

        Returns
        -------
        None.

        """
        self.farFieldCenter=(farFieldCenter)
        self.FFT_Q=FFT_Q
        
        if SLM is None:           
            self.SLMSize=np.array(SLMSize)
            self.screen_main=screeninfo.get_monitors()[screen_id_main]
            self.SLM_offset=SLM_offset
            self.SLM_Max=SLM_Max
            self.is_SANTEC=False
        else:
            self.SLMSize=SLM.SLMSize
            self.screen_main=SLM.screen_main
            self.SLM_offset=SLM.SLM_offset
            self.SLM_Max=SLM.SLM_Max
            self.is_SANTEC=SLM.is_SANTEC
        if precalc_holo is None:
            if farFieldCenter is None:
                self.Holo=np.zeros(SLMSize,np.csingle)
            else:
                self.Holo=self.blazedGrating(SLMSize,farFieldCenter)
        else:
            self.Holo=precalc_holo
            
        self.holo_mask=np.ones(SLMSize,float) 
        self.reference_hologram=None
    
    def holo_batch(self,ff_centers):
        Holos=[self.holo_copy() for k in np.arange(len(ff_centers))]
        with mp.Pool() as pool:
            FTs=pool.map(blazedg,ff_centers)
        # for k,x in enumerate(ff_centers):
        #     Holos[k].Holo=self.blazedGrating(self.SLMSize,x)
        __spec__=None
        for k,H in enumerate(Holos):
            H.Holo=FTs[k]
        return Holos   

    
    
    def blazedGrating(self,sSize,mInd, ntest=False):
        """  
        Parameters
        ----------
        sSize : Screen size
        mInd : ROW,COL

        Returns
        -------
        M : Blazed grating

        """ 
        # M=np.zeros((sSize))
        
        if ntest:
            s=sSize
            mInd=np.array(mInd)
            sSize=np.round(np.array(sSize)/2)
            # mInd[1]=np.round(mInd[1]*1.6).astype(np.int32)
            
            mInd=np.array((mInd[0]-sSize[0],mInd[1]-sSize[1]))
            mInd=mInd.astype(np.int32)
    
            if self.FFT_Q is None:
                M=np.zeros((s[1],s[1]))
                M[mInd[0],mInd[1]]=1
                M=M.astype(np.csingle)
                F=np.zeros_like(M)
                fft=pyfftw.FFTW(M, F, axes=(0,1), direction='FFTW_FORWARD', flags=('FFTW_MEASURE', ), 
                        threads=nthread, planning_timelimit=None )
                fft()
                # M=self.ft_wshift(M)
                sSize=self.SLMSize
                F=F[0:sSize[0],0:sSize[1]]
            else:
                M=cp.zeros((s[1],s[1]),dtype=cp.csingle)
                M[mInd[0],mInd[1]]=1
                self.FFT_Q[0].put([M,self.SLMSize])
                F=self.FFT_Q[1].get()
            return F
    
        else:    
            s=sSize
            mInd=np.array(mInd)
            sSize=np.round(np.array(sSize)/2)
            # mInd[1]=np.round(mInd[1]*1.6).astype(np.int32)
            
            mInd=np.array((mInd[0]-sSize[0],mInd[1]-sSize[0]))
            mInd=mInd.astype(np.int32)
    
            if self.FFT_Q is None:
                M=np.zeros((s[0],s[0]))
                M[mInd[0],mInd[1]]=1
                M=M.astype(np.csingle)
                F=np.zeros_like(M)
                fft=pyfftw.FFTW(M, F, axes=(0,1), direction='FFTW_FORWARD', flags=('FFTW_MEASURE', ), 
                        threads=nthread, planning_timelimit=None )
                fft()
                # M=self.ft_wshift(M)
                # sSize=self.SLMSize
                # F=F[0:sSize[0],0:sSize[1]]
                psize=(sSize[1]-sSize[0]).astype(np.int32)
                # print(psize)
                F=np.pad(F,([0,0],[psize,psize]))
            else:
                M=cp.zeros((s[0],s[0]),dtype=cp.csingle)
                M[mInd[0],mInd[1]]=1
                self.FFT_Q[0].put([M,self.SLMSize])
                F=self.FFT_Q[1].get()
            return F
    
    def grating_from_FF(self, M):
        M=M.astype(np.csingle)
        F=np.zeros_like(M)
        fft=pyfftw.FFTW(M, F, axes=(0,1), direction='FFTW_FORWARD', flags=('FFTW_MEASURE', ), 
                threads=nthread, planning_timelimit=None )
        fft()
        # M=self.ft_wshift(M)
        sSize=self.SLMSize
        F=F[0:sSize[0],0:sSize[1]]
        return F
    
    def four_point_grating(self,separation):
        """  
        Parameters
        ----------
        sSize : Screen size
        mInd : ROW,COL

        Returns
        -------
        M : Blazed grating

        """ 
        # M=np.zeros((sSize))
        sSize=self.SLMSize
        mInd=self.farFieldCenter
        
        sep_v=[[0,1],[0,-1],[-1,0],[1,0]]
        
        M=np.zeros((sSize[1],sSize[1]))
        mInd=np.array(mInd)
        sSize=np.round(np.array(sSize)/2)
        # mInd[1]=np.round(mInd[1]*1.6).astype(np.int32)
        
        mInd=np.array((mInd[0]-sSize[0],mInd[1]-sSize[1]))
        mInd=mInd.astype(np.int32)
        
        for m in sep_v:
            M[mInd[0]+separation*m[0],mInd[1]+separation*m[1]]=1
        
        
        M=M.astype(np.csingle)
        if self.FFT_Q is None:
            F=np.zeros_like(M)
            fft=pyfftw.FFTW(M, F, axes=(0,1), direction='FFTW_FORWARD', flags=('FFTW_MEASURE', ), 
                    threads=nthread, planning_timelimit=None )
            fft()
            # M=self.ft_wshift(M)
            sSize=self.SLMSize
            F=F[0:sSize[0],0:sSize[1]]
        else:
            self.FFT_Q[0].put([M,self.SLMSize])
            F=self.FFT_Q[1].get()
        
        self.Holo=F
   
   
    def preview(self):
        """
        Plots the nearfield and farfield of the contained hologram

        Returns
        -------
        fig : Figure object pyplotlib

        """
        
        farField= ft.fftshift(ft.fft2(ft.fftshift(self.Holo)))
        fig=plt.figure()
        
        plt.subplot(221)
        plt.imshow(np.angle(self.Holo))
        plt.title('NF PHASE')
        
        plt.subplot(222)
        plt.imshow(np.angle(farField))
        plt.title('FF PHASE')
        
        plt.subplot(223)
        plt.imshow(np.abs(self.Holo))
        plt.title('NF AMP')
        
        plt.subplot(224)
        plt.imshow(np.abs(farField))
        plt.title('FF AMP')
        
        return fig
    
    def circular_mask(self,radius,center=None, store=False, size=None):
        """
        Creates a circular binary mask of the size of the SLM.

        Parameters
        ----------
        radius : Integer. Radius of the mask
        center : Tupple, array. Center of the mask
                 When not specified the center of the mask is assumed 
                 to be the center of the SLM
        store : Bool, Store the mask as part of the hologram object
            DESCRIPTION. The default is False.

        Returns
        -------
        mask : Binary mask

        """
        if size is None:
            SLMSize=np.array(self.SLMSize)
        else:
            SLMSize=np.array(size)
        
        if center is None:
            center=np.round(SLMSize/2)
        
        [X,Y]=np.meshgrid(np.arange(SLMSize[1]),np.arange(SLMSize[0]))
        Y=Y-center[0]
        X=X-center[1]
        R=np.sqrt(X**2+Y**2)
        mask=R<radius
        
        if store:
            self.holo_mask=mask
        
        return mask
    
    def bahtinov_mask(self):
        
        mask_size=np.array(self.SLMSize)
        mask_center=mask_size/2
        l=[(30,0),(21,21),(-21,21)]
        
        m_centers=[mask_center+x for x in l]
        masks=[(np.angle(self.blazedGrating(mask_size, np.round(mc).astype(np.int32)))>0).astype(np.int32) for mc in m_centers]
        
        [X,Y]=np.meshgrid(np.arange(self.SLMSize[1]),np.arange(self.SLMSize[0]))
        Y=Y-mask_center[0]
        X=X-mask_center[1]
        R=np.sqrt(X**2+Y**2)
        
        mask=(X<0).astype(np.int32)*masks[0]+np.logical_and(X>0,Y<0).astype(np.int32)*masks[1]+np.logical_and(X>0,Y>0).astype(np.int32)*masks[2]
        self.holo_mask=mask*(R<600).astype(np.int32)
        
        return mask
    
    def azim_phase_mask(self, mask_center, l=0):
        [X,Y]=np.meshgrid(np.arange(self.SLMSize[1]),np.arange(self.SLMSize[0]))
        Y=Y-mask_center[0]
        X=X-mask_center[1]
        phi=np.arctan2(X,Y)
        M=1*np.exp(1j*phi*l)
        return M
        
    
    def area_grating(self, radius, center=None):
        """
        Generates a hologram for filling a region in the farfield with light
        """
        if center is None:
            center=self.farFieldCenter
        mask=self.circular_mask(radius, center)
        mask=mask.astype(np.float32)
        mask=ndi.gaussian_filter(mask, radius)
        M=self.ft_wshift(mask)
        self.Holo=M
        return mask
    
    def apply_stored_mask(self):
        """
        

        Applies any stored mask directly onto the hologram.

        """
        self.Holo=self.Holo*self.holo_mask
        return self.Holo
    
    def apply_WFC(self, HOLOGRAM, WFC, holo_dump=None, store=False):
        mabs=np.abs(WFC)
        mabs=mabs-np.min(mabs)
        mabs=mabs/np.max(mabs)
        
        mabs_prime=mabs
        mabs_prime[mabs<=0.1]=0
        mabs_prime[mabs>0.1]=1.1-mabs[mabs>0.1]
        
        mang=np.angle(WFC).astype(np.float32)   
        # mabs[mabs<0.3]=0.3
        # mabs[mabs>0.7]=0.7
        # mdump=2*(1-mabs)
        
        # HOLOGRAM=np.exp(1j*mang)*HOLOGRAM+mabs*holo_dump
        if type(HOLOGRAM)==cp.ndarray and type(WFC)!=cp.ndarray :
            HOLOGRAM=cp.asnumpy(HOLOGRAM)
        HOLOGRAM=np.exp(1j*mang)*HOLOGRAM.astype(np.csingle)
        HOLOGRAM=np.exp(1j*np.angle(HOLOGRAM)*mabs_prime)
        
        # HOLOGRAM=HOLOGRAM.astype(np.complex64)/(0.1+np.conj(WFC))
        # HOLOGRAM=HOLOGRAM/np.max(np.abs(HOLOGRAM))
        # HOLOGRAM=HOLOGRAM+0.1*holo_dump
        
        if store:
            self.Holo=HOLOGRAM
        
        return HOLOGRAM
    
    def display_shift_capture(self,rise_time, nshifts, player_Q, slevel=0.3,
                              t_pullup=1,l_pullup=1, drift_ref=None, 
                              ref_holo=None, cam=None, print_time=False, amp_mod=False):
            start_time=time()
            if cam is None:
                cam=cam_capture()
            ref_holo=cp.array(ref_holo,dtype=cp.csingle)    
            if cam.reference_images is None:

                d=cp.asnumpy(0.5+cp.angle(ref_holo))/(2*cp.pi)
                
                if not(player_Q is None):
                    player_Q.put(d)
                
                
                reference_images=cam.multi_wait_and_capture(rise_time*2)
                
                if not(player_Q is None):
                    for k,x in enumerate(reference_images):
                        x=x-np.min(x)
                        x[x<1]=1
                        x=x/np.max(x)
                        reference_images[k]=np.sqrt(x)
                    cam.reference_images=reference_images
                else:
                    cam.reference_images=[np.ones_like(x) for x in reference_images]
            
            # offset=self.SLM_offset
            # maxL=self.SLM_Max
            HOLOGRAM=self.Holo
            

            if type(HOLOGRAM) is np.ndarray:
                HOLOGRAM=cp.array(HOLOGRAM)
            shifts=cp.array(cp.exp(1j*cp.arange(nshifts)*2*cp.pi/cp.array(nshifts)),dtype=cp.csingle)
            HOLOS=[HOLOGRAM*x for x in shifts]
            shifts=cp.asnumpy(shifts)
            if not (ref_holo is None):
                HOLOS=[slevel*H + ref_holo for H in HOLOS]
            
            if amp_mod:
                HOLOS_A=[cp.abs(H) for H in HOLOS]
                HOLOS_A=[A-cp.min(A) for A in HOLOS_A]
                HOLOS_A=[A/cp.max(A) for A in HOLOS_A]
                HOLOS=[cp.exp(1j*cp.angle(H)*HOLOS_A[k]) for k,H in enumerate(HOLOS)]
                
                
            if l_pullup==1:
                HOLOS_p=HOLOS
            else:
                HOLOS_p=[cp.asnumpy(l_pullup*(0.5+cp.angle(H)/(2*cp.pi))) for H in HOLOS]
            HOLOS=[cp.asnumpy(0.5+cp.angle(H)/(2*cp.pi)) for H in HOLOS]

            captures=[]
            # Hs= it.islice(HOLOS, 1, None)
            t=time() - start_time
            if print_time:
                print(t)
                
            if not(drift_ref is None):
                d=slevel*drift_ref+ref_holo
                d=cp.asnumpy(0.5+cp.angle(d)/(2*cp.pi))
                if player_Q is None:
                    self.Holo=d
                    self.display()
                    sleep(t_pullup/1000)
                    self.Holo=d*l_pullup
                    self.display()
                else:
                    player_Q.put(d)
                    sleep(t_pullup/1000)
                    player_Q.put(d*l_pullup)
                drift_images=cam.multi_wait_and_capture(rise_time)
            else:
                drift_images=None
            # k=0
            for k in np.arange(len(HOLOS)):
                
                if player_Q is None:
                    self.Holo=HOLOS[k]
                    self.display()
                else:
                    player_Q.put(HOLOS[k])                               
                    if t_pullup>1:
                        sleep(t_pullup/1000)
                        player_Q.put(HOLOS_p[k])
                c=cam.multi_wait_and_capture(rise_time)
                c=[x*shifts[k]/(cam.reference_images[q]) for q,x in enumerate(c)]
                captures.append(c)
                # k+=1
            
            del HOLOS    
            


            
            return captures, drift_images
    
    def display_shift_capture_beta(self,rise_time, nshifts, player_Q, 
                                   t_pullup=1,l_pullup=1, drift_ref=None, ref_holo=None, cam=None, print_time=False):
            start_time=time()
            if cam is None:
                cam=cam_capture()
                

            
            # offset=self.SLM_offset
            # maxL=self.SLM_Max
            HOLOGRAM=self.Holo

            if type(HOLOGRAM) is np.ndarray:
                HOLOGRAM=cp.array(HOLOGRAM)
            shifts=cp.exp(1j*cp.arange(nshifts)*2*cp.pi/nshifts)
            HOLOS=[HOLOGRAM*x for x in shifts]
            shifts=cp.asnumpy(shifts)
            if not (ref_holo is None):
                HOLOS=[0.3*H + ref_holo for H in HOLOS]
            
            HOLOS_p=[cp.asnumpy(l_pullup*(0.5+cp.angle(H)/(2*cp.pi))) for H in HOLOS]
            HOLOS=[cp.asnumpy(0.5+cp.angle(H)/(2*cp.pi)) for H in HOLOS]

            captures=[]
            # Hs= it.islice(HOLOS, 1, None)
            t=time() - start_time
            if print_time:
                print(t)
                
            if not(drift_ref is None):
                d=0.3*drift_ref+ref_holo
                d=cp.asnumpy(0.5+cp.angle(d)/(2*cp.pi))
                player_Q.put(d)
                sleep(t_pullup/1000)
                player_Q.put(d*l_pullup)
                drift_images=cam.multi_wait_and_capture(rise_time)
            else:
                drift_images=None
            # k=0
            for k in np.arange(len(HOLOS)):
                player_Q.put(HOLOS[k])
                sleep(t_pullup/1000)
                player_Q.put(HOLOS_p[k])
                c=cam.multi_wait_and_capture(rise_time)
                c=[x*shifts[k] for x in c]
                captures.append(c)
                # k+=1
            
            del HOLOS    
            


            
            return captures, drift_images
          
            
    def display(self,
                holo_add=None, 
                apply_mask=False, 
                apply_reference=False, 
                player=None, 
                WFC=None,l=0, 
                dump_loc=(350,900), 
                offset_max=[0,1], 
                t_pullup=None, 
                pullup_level=None,
                shift=None):
        """
        This function controls the display of the hologram either using a player object
        or a new window (a player is needed when multithreading)

        Parameters
        ----------
        holo_add : An additional hologram object can be inserted here. The display method will
        show the phase of the complex addition of both holograms
            The default is None.
        amp_modulation : A threshold for emulating amplitude using difraction efficiency modulation (linearity is assumed)
            if None the phase is displayed unmodulated
            The default is None.
        apply_mask : If True the stored mask is applied on top of the hologram before display
        apply_reference : If True, the stored reference hologram is added to the stored hologram before display
        player : A player object. If provided the hologram is displayed there, otherwise a window is created

        Returns
        -------
        None.

        """
        
        SANTEC=self.is_SANTEC
        screen_main=self.screen_main
        
        if offset_max is None:
            offset=self.SLM_offset
            maxL=self.SLM_Max
        else:
            offset=offset_max[0]
            maxL=offset_max[1]
        


#Apply the different modifieres----------------------------------------------


        if holo_add is None:
            HOLOGRAM=self.Holo
        else:
            HOLOGRAM=self.Holo+holo_add    
        
        if not(shift is None):
            HOLOGRAM=HOLOGRAM*np.exp(1j*shift)
         
        

        if apply_mask:

            angle_mask=(np.abs(self.holo_mask)>0).astype(np.float32)
            HOLOGRAM=self.holo_mask*np.abs(HOLOGRAM)*np.exp(1j*np.angle(HOLOGRAM)*angle_mask)
            print(['MASK SIZE', angle_mask.shape])
            print(['HOLO SIZE', HOLOGRAM.shape])
        
        if l>0:
            azmask=self.azim_phase_mask(np.array(self.SLMSize)/2, l=l)
            HOLOGRAM=HOLOGRAM*azmask
        
        if not(WFC is None):
            if dump_loc==(350,900):
                try:
                    filename=r'C:\TM_meas\HOLO_REELS\dump.pkl'
                    file=open(filename,'wb')
                    H=pkl.load(file)
                    file.close()
                    holo_dump=hologram(precalc_holo=H,farFieldCenter=dump_loc)
                except:
                    holo_dump=hologram(farFieldCenter=dump_loc)
            else:
                holo_dump=hologram(farFieldCenter=dump_loc)                
            HOLOGRAM=self.apply_WFC(HOLOGRAM, WFC, holo_dump.Holo)
           
        if type(HOLOGRAM)!=np.ndarray:
            H=cp.asnumpy(HOLOGRAM)
            del HOLOGRAM
            HOLOGRAM=H
        
        if apply_reference:
            if not(self.reference_hologram is None):
                HOLOGRAM=0.2*HOLOGRAM+self.reference_hologram.Holo
                 
#Make sure the correct data type is used--------------------------------------        
        H_PRIME=HOLOGRAM    
        HOLOGRAM=np.angle(HOLOGRAM)
        HOLOGRAM=HOLOGRAM
        HOLOGRAM=HOLOGRAM.astype(np.float32)
        HOLOGRAM=(HOLOGRAM+np.pi)/(2*np.pi)
#Values should range from 0 to 1----------------------------------------------            
        if np.max(HOLOGRAM)>1:
            HOLOGRAM=self.normalize(HOLOGRAM)
#Apply the offset and maximum gray level values-------------------------------        
        
        # HOLOGRAM=(HOLOGRAM>0.5).astype(np.float32)
        H=(HOLOGRAM*(maxL-offset)+offset)               

        
        if SANTEC: #SANTEC uses an RGB colormap for display
            C=self.getSLMColormap()
            C=C.astype(np.int8)
            s=H.shape
            H=np.round(H*1023)
            H=H.astype(np.uint16)
            H=np.reshape(H,-1)
            M=C[H,:]
            M=np.reshape(M,s+(3,)) 
        else: # GRAY LEVEL SLM (THORLABS)
            M=H.astype(np.float32)
            
        if player is None:
            import cv2 as cv
            
            A=datetime.now()
            D=[A.year,A.month,A.day]
            date_string=''.join(str.format('{:02d}', e) for e in D)
            
            
            main_name=date_string
            cv.namedWindow(main_name, cv.WND_PROP_FULLSCREEN )# 
            cv.moveWindow(main_name, screen_main.x - 1, screen_main.y - 1)
            cv.setWindowProperty(main_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)  #  
            
            cv.imshow(main_name,M)
            # cv2.imshow(dispwin,holoDisp)
            cv.waitKey(1)
        else:
            player.player_Q.put(M)
            if not(t_pullup is None) and not(pullup_level is None):
                sleep(t_pullup)
                M0=(M*pullup_level).astype(np.float32)
                player.player_Q.put(M0)
            
         
            
        return H_PRIME
    def getSLMColormap(self,hz120=False):
        """
        SLM Color map for SANTEC SLM

        Parameters
        ----------
        hz120 : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        mapBGR : TYPE
            DESCRIPTION.

        """
        A=np.random.rand(2,2)
        A=np.round(A*1023)
        
        A=np.arange(0,1024)
        A=A.astype(np.int16)
        
        sA=np.array(A.shape)
        mapBGR=np.zeros([sA[0],3])
        
        for m in np.arange(sA[0]):
            
                binNumber=bin(A[m])
                # binNumber=bin(960)
                s=12-len(binNumber)
                if s>0:
                    for p in np.arange(s): 
                        binNumber='0b'+'0'+binNumber[2:]            
                
                if hz120:
                    R=int('0b'+binNumber[2:5]+'0'+binNumber[2:5]+'0',2)
                    G=int('0b'+binNumber[5:8]+'0'+binNumber[5:8]+'0',2)
                    B=int('0b'+binNumber[8:]+binNumber[8:],2)
                else:
                    R=int('0b'+binNumber[2:5]+'00000',2)
                    G=int('0b'+binNumber[5:8]+'00000',2)
                    B=int('0b'+binNumber[8:]+'0000',2)
                    
                mapBGR[m,:]=np.dstack((B,G,R))    
        
        return mapBGR
    


    
    def normalize(self,M):
        M=M-np.min(M)
        M=M/np.max(M)
        return M
    
    def holo_copy(self):
        C=hologram(SLMSize=(1,1),farFieldCenter=None)
        C.FFT_Q=self.FFT_Q
        C.Holo=self.Holo
        C.reference_hologram=self.reference_hologram
        return C
    
    def holoPhaseShift(self,tn,n):
        """
        Applies a phase shift=tn/n*pi()

        Parameters
        ----------
        tn : TYPE
            DESCRIPTION.
        n : TYPE
            DESCRIPTION.

        Returns
        -------
        C : TYPE
            DESCRIPTION.

        """
        holo=self.Holo
        shifts=np.linspace(0,2*np.pi,num=tn,endpoint=False)
        C=copy.copy(self)
        C.Holo=holo*np.exp(1j*shifts[n])
        return C
    

    
    def batch_holoPhaseShift(self,tn):

        """
        Applies a phase shift=tn/n*pi()

        Parameters
        ----------
        tn : TYPE
            DESCRIPTION.
        n : TYPE
            DESCRIPTION.

        Returns
        -------
        C : TYPE
            DESCRIPTION.

        """
        Holo=self.Holo.astype(np.csingle)
        
        # holo=self.Holo
        def shift_by(phi):
            H=self.Holo*np.exp(1j*phi)
            return H
        shifts=np.linspace(0,2*np.pi,num=tn,endpoint=False)
        
        # with mp.Pool(5) as pool:
        #     ps_holos=pool.map(shift_by, shifts)
        
        ps_holos=[shift_by(p) for p in shifts]
        
        result=[]
        for h in ps_holos:
            C=copy.copy(self)
            C.Holo=h
            result.append(C)
            
        return result
    
    def ft_wshift(self,M):
        """
        Fast fourier transform with fftshift in both planes

        Parameters
        ----------
        M : TYPE
            DESCRIPTION.

        Returns
        -------
        M : TYPE
            DESCRIPTION.

        """
        M=ft.fftshift(ft.fft2(ft.ifftshift(M)))
        abM=np.abs(M)
        abM=abM/np.max(abM)
        M=abM*np.exp(1j*np.angle(M))
        
        return M
    
    def clear_hologram(self):
        self.Holo=None
        

class cam_capture:
    def __init__(self,camSerial="16484",ROI=[448,300,480,300,1,1],buffSize=1
                         ,camExp=2e-3):
        """
        Initializes a camera for measurement

        Parameters
        ----------
        camSerial : Camera serila number
            DESCRIPTION. The default is "13294".
        ROI : R
            DESCRIPTION. The default is [680,300,380,300].
        buffSize : TYPE, optional
            DESCRIPTION. The default is 10.
        camExp : TYPE, optional
            DESCRIPTION. The default is 0.5e-3.

        Returns
        -------
        None.

        """
        
        self.ROI=ROI
        self.subROI=None
        self.serial=camSerial
        self.exp=camExp
        self.buffSize=buffSize
        self.acquisitionStarted=False
        self.acquisition_size=None

        
        
        cam = Thorlabs.ThorlabsTLCamera(serial=camSerial) 

        cam.acquisition_in_progress()
        cam.set_roi(ROI[0],ROI[1]+ROI[0],ROI[2],ROI[3]+ROI[2],hbin=ROI[4],vbin=ROI[5])
        cam.set_exposure(camExp)      
        self.cam=cam
        self.acquisition_size=self.frame_size()
        self.reference_images=None
    def frame_size(self):
        return self.cam.get_data_dimensions()
        
    def ROI_center_size(self, center, size):
        size_half=np.round(size/2)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        ROI=[center[0]-size_half[0],size[0],center[1]-size_half[1],size[1]]
        self.ROI=ROI
        self.cam.set_roi(ROI[0],ROI[1]+ROI[0],ROI[2],ROI[3]+ROI[2])
    
    def change_runtime_ROI(self, ROI):
        self.ROI=ROI
        self.acqstop()
        self.cam.set_roi(ROI[0],ROI[1]+ROI[0],ROI[2],ROI[3]+ROI[2],hbin=ROI[4],vbin=ROI[5])
        self.acqstart()
        
    def acqstart(self):
        
        try:
            if not(self.cam.acquisition_in_progress()):
                cam=self.cam
                cam.setup_acquisition(nframes=self.buffSize)
                cam.start_acquisition()
                self.acquisitionStarted=True
            else:
                self.acqstop()
                # self.acqstart()
                
        except:
            print('ERROR STARTING ACQUSITION')
            
    def acqstop(self):
        self.cam.stop_acquisition()
        # self.cam.clear_acquisition()
    
    def wait_and_capture(self,time_to_wait, A_B='A'):
        self.cam.wait_for_frame()
        self.cam.read_newest_image()
        sleep(time_to_wait/1000)
        self.cam.wait_for_frame()
        frame=self.cam.read_newest_image()
        if frame is None:
            self.cam.stop_acquisition()
            self.cam.close()
            self.cam.open()
            self.acqstart()
            frame=self.cam.read_newest_image()
        
        if not(self.subROI is None):
            if A_B=='A':
                crop=self.subROI[0]
            else:
                crop=self.subROI[1]
                
            frame=frame[crop[0]:crop[0]+crop[1],crop[2]:crop[2]+crop[3]]
            
        return frame
    
    def multi_wait_and_capture(self,time_to_wait):
        frames=[]
        self.cam.wait_for_frame()
        self.cam.read_newest_image()
        sleep(time_to_wait/1000)
        self.cam.wait_for_frame()
        frame=self.cam.read_newest_image()
        if frame is None:
            self.cam.stop_acquisition()
            self.cam.close()
            self.cam.open()
            self.acqstart()
            frame=self.cam.read_newest_image()
        
        if not(self.subROI is None): 
            for crop in self.subROI:
                frames.append(frame[crop[0]:crop[0]+crop[1],crop[2]:crop[2]+crop[3]])
        else:
            frames=[frame]
            
        return frames
    
    def HDR_image(self, exp_times, lthresh=32, hthresh=1008):
        accumulator=np.array(self.wait_and_capture(1))
        accumulator=accumulator.astype(np.float64)*0.0
        in_range=accumulator.astype(np.bool8)
        for k in exp_times:
            self.cam.set_exposure(k)
            frame=self.wait_and_capture(300)
            frame=frame.astype(np.float32)
            ran=np.logical_and(frame>lthresh,frame<hthresh)
            mask=np.logical_and(ran,np.logical_not(in_range))         
            accumulator=accumulator+((frame.astype(np.float64)/k)*mask.astype(np.int32))
            in_range=np.logical_or(mask,in_range)
        return accumulator, in_range
    
def blazedg (mInd):
    """  
    Parameters
    ----------
    sSize : Screen size
    mInd : ROW,COL

    Returns
    -------
    M : Blazed grating

    """ 
    # M=np.zeros((sSize))
    
    sSize=(1200,1920)
    M=np.zeros((sSize[1],sSize[1]))
    mInd=np.array(mInd)
    sSize=np.round(np.array(sSize)/2)
    # mInd[1]=np.round(mInd[1]*1.6).astype(np.int32)
    
    mInd=np.array((mInd[0]-sSize[0],mInd[1]-sSize[1]))
    mInd=mInd.astype(np.int32)
    M[mInd[0],mInd[1]]=1
    M=M.astype(np.csingle)

    F=np.zeros_like(M)
    fft=pyfftw.FFTW(M, F, axes=(0,1), direction='FFTW_FORWARD', flags=('FFTW_MEASURE', ), 
            threads=nthread, planning_timelimit=None )
    fft()
    # M=self.ft_wshift(M)
    sSize=(1200,1920)
    F=F[0:sSize[0],0:sSize[1]]

    return F
                    
        
    


        

        