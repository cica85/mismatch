# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:34:10 2024

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb
from scipy import fft as ft
# c
def imshow(M):        
    if np.iscomplex(M).any():
        plotme=colorize(M)
    else:
        plotme=M
    plt.imshow(plotme)

def colorize(z):

    
    r = np.abs(z)
    r=r-np.min(r)
    r=r/np.max(r)    
    arg = np.angle(z)+np.pi/3 
    h = (arg + np.pi)  / (2 * np.pi) + 0.5

    v=r
    s = 0.8
    c = np.vectorize(hsv_to_rgb) (h,s,v) 
    c = np.array(c)  
    c = c.swapaxes(0,2) 
    c = c.transpose(1,0,2)
    return c  

class mismatch():
    
    """
    CONTAINS METHODS FOR CALCULATING THE ABERRATED WAVEFRONT DUE TO REFRACTIVE 
    INDEX MISMATCH  & FOR OBTAINING A CORRECTION HOLOGRAM FOR AN SLM IN THE
    CONJUGATE PLANE (IMAGED ONTO THE BACKFOCAL PLANE OF AN INFINITY CORRECTED
    MICROSCOPE OBJECTIVE)
    
    """
    
    def __init__(self, n1=1.45, n2=2.46, NA=0.9, lam=810e-9):
        """
        INITIALIZE MISMATCH OBJECT
        
        n1: refractive index of medium 1 (e.g. immersion oil)
        n2: refractive index of medium 2 (e.g. diamond)
        NA: numerical apperture of the microscope objective being used
        lam: wavelength in meters
        """
        
        #STORE PARAMETERS AS PART OF THE OBJECT
        self.n1=n1 
        self.n2=n2
        self.NA=NA
        self.lam=lam        
        self.k=2*np.pi*n1/lam
        
        
        #DEFINE LAMBDA (ANONYMOUS) FUNCTIONS FOR CALCULATING THE ABERRATION
        #BASED ON:
        #https://doi.org/10.1117/1.1382808
        
        csc= lambda v: np.sin(v)**(-1)
        self.get_NA_angle=lambda n,NA: np.arcsin(NA/n)
        self.coeff=lambda a, r: np.sqrt((csc(a)**2-r**2)*np.heaviside(np.sign(1-r),1))
        self.psi= lambda r,d,a1,a2: d*self.k*np.sin(a1)*(self.coeff(a1,r)-self.coeff(a2,r))
        
        return None
    
    def polarmesh(self,res):
        #GENERATES A POLAR MESH WITH A RESOLUTION: res x res pixels
        #res: integer [pixels]
        
        ran=np.linspace(-1, 1,res)
        Y,X=np.meshgrid(ran,ran)
        R=np.sqrt(Y**2+X**2)
        phi=np.arctan2(Y,X)
        return R, phi

    def get_pupil_phase(self,res,depth, display=True):
        """
        Generates the phase of a unit pupil function.

        Parameters
        ----------
        res : resolution of the phase mask in pixels. Integer.
        depth : depth of the focus within medium 2 (e.g. diamond)
        display : keep true if you wish the results to be plotted

        Returns
        -------
        pupil : Dictionary containing the 'phase' of the pupil function and the 
                mesh 'R' used to create it

        """
        R,_=self.polarmesh(res)
        alpha1=self.get_NA_angle(self.n1, self.NA)
        alpha2=self.get_NA_angle(self.n2, self.NA)
        p=self.psi(R,depth,alpha1,alpha2)
        p=(p-p[res//2,res//2])
        
        pupil={'phase': p, 'R': R}

        if display:
            p[R>1]=np.nan
            plt.figure()
            plt.imshow(p)
            plt.set_cmap('inferno')
                        
        return pupil

    def pupil_profiles(self, depths=(1e-5,1e-4,1e-3,),res=100):
        
        """
        Plots the profile of the pupil function phase for the depths contained
        within the 'depths' parameter 
        
        Parameters
        ----------
        depths= iterable (list, tuple, etc.) containing the focusing depths 
        within 
        
        """
        
        r_ax=np.linspace(-1, 1,res)
        plt.figure()
        for d in depths:            
            alpha1=self.get_NA_angle(self.n1, self.NA)
            alpha2=self.get_NA_angle(self.n2, self.NA)
            p=self.psi(r_ax,d,alpha1,alpha2)
            p=(p-p[res//2])
            plt.plot(p[10:-10])
            
    def get_SLM_hologram(self,
                         depth, 
                         slm_size=(1200,1920), 
                         MO_backfocal_rad=200, 
                         MO_center=(100,100), 
                         dump_center=(-150,-120),
                         dump_intensity=1,
                         out_format='csingle'):
        """
        

        Parameters
        ----------
        depth : Depth of focus in medium 2
        slm_size : Size of the SLM screen in pixels. The default is (1200,1920).
        MO_backfocal_rad : Radius of the back apperture of the microscope 
                            objective in pixels
        MO_center : Center of the microscope objective back apperture in 
                    reference to the center of the SLM far_field
                    default is (100,100).
        dump_center : A location in the SLM far_field used to dump extra
                    light
                    The default is (-150,-120).
        out_format : Format of the output:
                    'csingle'
                    'uint8'
                    'uint16'
                    'float'
            DESCRIPTION. The default is 'csingle'.

        Returns
        -------
        SLM : TYPE
            DESCRIPTION.
        farfield : TYPE
            DESCRIPTION.

        """
        farfield=np.zeros((np.min(slm_size),np.min(slm_size))).astype(np.csingle)
        
        
        dump=np.zeros((np.min(slm_size),np.min(slm_size))).astype(np.csingle)
        dumploc=np.array(dump_center)+np.min(slm_size)//2
        
        if not (dumploc is None):
            dump[dumploc[0], dumploc[1]]=1
        
        
        plt.figure()
        plt.imshow(np.abs(dump))
        print(dump.shape)
        
        pupil=self.get_pupil_phase(MO_backfocal_rad*2, depth)
        p=(pupil['R']<=1).astype(np.float32)*np.exp(1j*pupil['phase']*(pupil['R']<=1).astype(np.float32))
        p[pupil['R']>1]=0j
        start=np.array(MO_center)-MO_backfocal_rad+np.min(slm_size)//2
        end=np.array(MO_center)+MO_backfocal_rad+np.min(slm_size)//2
        
        farfield[start[0]:end[0],start[1]:end[1]]=p

        cSLM=ft.ifftshift(ft.ifft2(ft.ifftshift(farfield)))
        dump=ft.ifftshift(ft.ifft2(ft.ifftshift(dump)))
        
        cSLM=cSLM/np.max(np.abs(cSLM))
        dump=dump/np.max(np.abs(dump))
        
        cSLM=cSLM+dump_intensity*dump
        
        pad_size=np.abs(slm_size[0]-slm_size[1])
        
        if pad_size%2==0:
            pad_s=[pad_size//2, pad_size//2]
        else:
            pad_s=[pad_size//2, pad_size//2+1]
        
        SLM=np.zeros(slm_size).astype(np.csingle)
        
        if np.argmax(slm_size)==0:
            SLM[pad_s[0]:-pad_s[1],:]=cSLM
        else:
            SLM[:,pad_s[0]:-pad_s[1]]=cSLM
        
        if out_format=="uint8":
            SLM=np.angle(SLM)
            SLM=(1+SLM/np.pi)/2
            SLM=(SLM*255).astype(np.uint8)
        
        elif out_format=="uint16":
            SLM=np.angle(SLM)
            SLM=(1+SLM/np.pi)/2
            SLM=(SLM*65535).astype(np.uint16)
        
        elif out_format=="float":
            SLM=np.angle(SLM)
            SLM=(1+SLM/np.pi)/2
        
        
        return SLM, farfield
            
if __name__ == '__main__': 
    
    plt.close('all')
    m=mismatch()
    SLM,ff=m.get_SLM_hologram(4e-5, out_format='uint16',MO_center=(0,0), dump_center=(50,-50))
    plt.figure()
    plt.subplot(1,2,1)
    imshow(ff)
    plt.subplot(1,2,2)
    imshow(SLM)