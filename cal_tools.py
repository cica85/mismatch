# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 09:17:05 2023

@author: QNanoLab
"""
import sys
sys.path.append("D:\Angel\Python\Wavefront correction  0.2/")

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import screeninfo
import scipy.ndimage as ndi
import skimage.transform as im
import cv2
import os
import h5py 
import pickle as pkl
import copy

from matplotlib.widgets import Button

from queue import Queue
from queue import Empty

from threading import Thread
from datetime import datetime
from time import sleep
from scipy import fft as ft

import multiprocessing as mp


from holo_tools import hologram
from holo_tools import cam_capture
from holo_tools import holo_player
from holo_tools import FFT_process

nthread = mp.cpu_count()
q_timeout=20

class dashboard:
    def __init__(self, screen_id_main=0, end_Q=None):
        self.dash_Q=Queue(maxsize=2) # A Queue object to comunicate with the dashboard
        self.screen_main=screeninfo.get_monitors()[screen_id_main] #Screen information
        self.end_Q=end_Q
           
    def dash_run(self):
        """
        

        Parameters
        ----------
        screen_main : Screen information

        Returns
        -------
        None.

        """
        # screen_main=self.screen_main
        Q=self.dash_Q
        fig=plt.figure(figsize=(16,4))
        fig.canvas.manager.set_window_title('Calibration Dashboard')
        ax=[]
        
        
        tn=131
        
        for e in np.arange(3):
            n=tn+e  
            sp=plt.subplot(n)
            sp.set_xticks([])
            sp.set_yticks([])
            ax.append(sp)
        
        # prev_element=None
        if not(self.end_Q is None):
            axcolor = 'white'
            bax = fig.add_axes([0.005, 0.95, 0.05, 0.05], facecolor=axcolor)
            end_b = Button(bax,'END')
            end_b.on_clicked(lambda x: self.end_Q.put(True))
    # 'Axis','Image','Plot_point'
        

        while True: 
         
            try:
                element_list=Q.get(timeout=15)
                
            except:
                break
            if element_list is None:
                break                                         
            for element in element_list:
                if element is None:
                    break
                try:
                    ax_ind=element['Axis']
                except:
                    print('update failed')
                    break
    
                    
                if not(element['Image'] is None):
                    
                    if type(ax[ax_ind].get_children()[0])==mpl.image.AxesImage:
                       ax[ax_ind].get_children()[0].set_array(element['Image'])
                    else:
                       ax[ax_ind].imshow(element['Image'],cmap=mpl.colormaps['gray'])
                       ax[ax_ind].set_xlabel(np.max(element['Image']))
                               
      
                
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.draw()
                # prev_element=element  #The last element displayed is stored in case of a timeout over the next iteration



        return None


class calibration:
    
    def __init__(self, cam, far_field_center=(1066,708), slm_size=(1200,1920), 
                 save_dir='D:\Angel\Calibration_results', rise_time=150, 
                 phase_shift_n=4, core_size=8, screen_id_main=1):
        
        self.screen_main=screeninfo.get_monitors()[screen_id_main]
        self.screen_id_main=screen_id_main
        self.save_dir=save_dir
        self.rise_time=rise_time
        self.slm_size=slm_size
        self.FF_center=far_field_center
        
        self.reference_holo=None
        self.reference_on=True #Controls the state of the reference beam on/off
        
        self.central_refernce=hologram(farFieldCenter=far_field_center,
                                     SLMSize=slm_size)
        
        
        self.cam=cam #Camera in use
        self.out_cam=cam # Fiber output camera
        self.in_cam=None # Fiber input camera
        
        self.stop_Q=Queue(maxsize=1)
        self.done_Q=Queue(maxsize=1)
        self.end_gen_Q=Queue(maxsize=1)
        
        self.FFT_Q=[mp.Queue(maxsize=1),mp.Queue(maxsize=1)]
        self.display_Q=Queue(maxsize=50)
        
        self.capture_Q=Queue(maxsize=100000)
        self.capture_Q_b=Queue(maxsize=100000)
        
        self.field_Q=Queue(maxsize=100000)
        self.field_Q_b=Queue(maxsize=100000)
        
        
        self.player_Q=Queue(maxsize=100000)
        self.wfc_Q=Queue(maxsize=100000)
        self.mode_Q=Queue(maxsize=10000)
        
        
        
        self.Y_mesh=None
        self.X_mesh=None
        
        self.WFC=None
        self.WFC_on=False
        
        self.core_size_px=core_size
        self.phase_shift_n=phase_shift_n
        self.input_mode_image=None
        self.input_modes=None
        self.farfield_mask=[]
        self.drift_ref=None
        
        self.angular_intens_image=None
        self.TM=[]
        self.accumulated_image=None
        self.result_shape=None
        
        self.forced_end=False
        self.last_mode=None
        self.t_pullup=0.01
        self.pullup_level=0.7
    
    # def run(self):
    def save_config(self, filename):
        file=open(filename,'wb')
        
        L={"screen_main": self.screen_main,
           "screen_id_main": self.screen_id_main,
           "save_dir": self.save_dir,
           "rise_time": self.rise_time,
           "slm_size": self.slm_size,
           "FF_center": self.FF_center,
           "reference_on": self.reference_on,
           "Y_mesh": self.Y_mesh,
           "X_mesh": self.X_mesh,
           "WFC": self.WFC,
           "WFC_on": self.WFC_on,
           "core_size_px": self.core_size_px,
           "phase_shift_n": self.phase_shift_n,
           "input_mode_image": self.input_mode_image,
           "input_modes": self.input_modes,
           "farfield_mask": self.farfield_mask,
           "drift_ref": self.drift_ref,
           "angular_intens_image": self.angular_intens_image,
           "TM": self.TM,
           "accumulated_image": self.accumulated_image,
           "result_shape": self.result_shape,
           "last_mode": self.last_mode}
        
        
        pkl.dump(L,file)
        file.close()
        
    def load_config(self, filename):
        file=open(filename,'rb')
        L=pkl.load(file)
        file.close()
        
        self.screen_main=L["screen_main"]
        self.screen_id_main=L["screen_id_main"]
        self.save_dir=L["save_dir"]
        self.rise_time=L["rise_time"]
        self.slm_size=L["slm_size"]
        self.FF_center=L["FF_center"]
        self.reference_on=L["reference_on"]
        self.Y_mesh=L["Y_mesh"]
        self.X_mesh=L["X_mesh"]
        self.WFC=L["WFC"]
        self.WFC_on=L["WFC_on"]
        self.core_size_px=L["core_size_px"]
        self.phase_shift_n=L["phase_shift_n"]
        self.input_mode_image=L["input_mode_image"]
        self.input_modes=L["input_modes"]
        self.farfield_mask=L["farfield_mask"]
        self.drift_ref=L["drift_ref"]
        self.angular_intens_image=L["angular_intens_image"]
        self.TM=L["TM"]
        self.accumulated_image=L["accumulated_image"]
        self.result_shape=L["result_shape"]
        self.last_mode=L["last_mode"]

            
        
        
    def display_central_hologram(self, WFC_on=True):
        
        H=hologram(farFieldCenter=self.FF_center,screen_id_main=self.screen_id_main)
        if WFC_on:
            H.display(WFC=self.WFC)
        else:
            H.display()
        sleep(0.15)
                        
    def run_initial_area_scan(self, nbin=1, WFC_on=False, holo_reel_filename=None):
        
        if not(holo_reel_filename is None):
            WFC_on=False
            
        
        self.cam=self.out_cam
        self.reference_on=False
        self.clear_Q() #Make sure the Qs are clean before start
        self.stop_Q.put(True)
        


    ##START DISPLAY WINDOW FOR HOLOGRAMS 
        player=holo_player(screen_id_main=self.screen_id_main)
        player.player_start()
     ##CREATE A DASHBOARD
        dash=dashboard(end_Q=self.end_gen_Q)       
        monitor_Q=dash.dash_Q
    ##DEFINE THREADS
        
        # holo_gen = Thread(target=self.scan_area,kwargs={"radius": self.core_size_px,"nbin":nbin,
        #                                                 "FFT_Q":None,"hologram_reel_filename":holo_reel_filename})
        
        holo_gen = Thread(target=self.scan_area,kwargs={"radius": self.core_size_px,
                                                        "nbin":nbin,
                                                        "initial_scan": True, 
                                                        "nShifts": None,
                                                        "refLoc":None,
                                                        "hologram_reel_filename":None,
                                                        "FFT_Q":self.FFT_Q,
                                                        "apply_WFC":False})    
        # holo_gen = multiprocess.Process(target=self.scan_area,kwargs={"radius": self.core_size_px,"nbin":nbin})
        FFT_proc=mp.Process(target=FFT_process, kwargs={'Q': self.FFT_Q})        
        display = Thread(target=self.display_capture, kwargs={"player": player,"WFC_on":False})
        processing=Thread(target=self.frame_processing, kwargs={"monitor_Q":monitor_Q})
        
    ##START THREADS
        FFT_proc.start()
        sleep(0.1)
        holo_gen.start()
        display.start()
        processing.start()
    ##DASHBOARD START    
        
        dash.dash_run()  
        
   
        
    ## Wait for threads to finish
        holo_gen.join()
        self.FFT_Q[0].put(None)
        FFT_proc.join()
        display.join()
        player.player_stop()
        processing.join()
        
    ##GET IMAGE OF INPUT MODES    
        M=self.process_initial_scan()
        self.input_mode_image=M
        
        return None
    
    def save_holo_reel(self, filename):
        nbin=1
        self.cam=self.out_cam
        
        self.reference_on=False
        self.clear_Q() #Make sure the Qs are clean before start
        self.stop_Q.put(True)
        




    ##DEFINE THREADS
        
        holo_gen = Thread(target=self.scan_area,kwargs={"radius": self.core_size_px,"nbin":nbin,"FFT_Q":self.FFT_Q, "apply_WFC": True})
        # holo_gen = multiprocess.Process(target=self.scan_area,kwargs={"radius": self.core_size_px,"nbin":nbin})
        FFT_proc=mp.Process(target=FFT_process, kwargs={'Q': self.FFT_Q})        
        save_thread = Thread(target=self.holo_reel_maker, kwargs={"filename": filename})

        
    ##START THREADS
        FFT_proc.start()
        sleep(2)
        holo_gen.start()
        save_thread.start()
        
    ## Wait for threads to finish
        holo_gen.join()
        self.FFT_Q[0].put(None)
        FFT_proc.join()
        save_thread.join()
        
        return None
    
       
    def holo_reel_maker(self,filename= r'C:\TM_meas\HOLO_REELS\reel.pkl'):
        Q=self.display_Q
        file=open(filename,'wb')
        k=0
        while True:
            try:
                q_item=Q.get(timeout=5)
            except Empty:
                self.end_Q()
                print("ENDING DUE TO TIMEOUT")
                break
            
            if q_item is None:
                break
            # print(Q.qsize())
            k+=1
            print(k)
            H=q_item['HOLOGRAM'].display(player=None,apply_reference=self.reference_on,WFC=self.WFC)
            pkl.dump(H,file)
        file.close()
    
    def run_initial_superpix_scan(self, super_pix_size=200):
        
        self.reference_on=False
        self.clear_Q()#Make sure the Qs are clean before start
        
    ##START DISPLAY WINDOW FOR HOLOGRAMS   
        player=holo_player(screen_id_main=self.screen_id_main)
        player.player_start()
    ##CREATE A DASHBOARD
        dash=dashboard(end_Q=self.end_gen_Q)
        monitor_Q=dash.dash_Q

    ##DEFINE THREADS
        holo_gen = Thread(target=self.super_pixel_scan,kwargs={"super_pixel_size": super_pix_size})       
        display = Thread(target=self.display_capture, kwargs={"player": player})
        processing=Thread(target=self.frame_processing, kwargs={"monitor_Q":monitor_Q})
        
    ##START THREADS           
        holo_gen.start()
        display.start()
        processing.start()
    ##DASHBOARD START    
        dash.dash_run()
            
        
    ## Wait for threads to finish
        holo_gen.join()
        print("HOLO JOIN")
        display.join()
        print("DISPLAY JOIN")
        processing.join()
        print("PROCESSING JOIN")
        player.player_stop()
        print("PLAYER STOP")

        
    ##GET IMAGE OF INPUT MODES
        print('PROCESSING SCAN')            
        M=self.process_initial_scan(super_pixel_size=super_pix_size)
        self.angular_intens_image=M
        
        return None
    
    def run_wfc(self, super_pix_size=200, phase_shifts=4, file_name='WFC.pkl', image_center=None, shift=None, scan_us=None, DMD=False):
        
        if not(self.in_cam is None): #If an input camera was assingned than the code uses this for WFC
            self.cam=self.in_cam    
        self.reference_on=False #Turn reference off
        self.clear_Q()#Make sure the Qs are clean before start
        
        #Initialize WFC as zeros
        M, ind=self.super_pixel_mask(super_pix_size,index=None)
        self.WFC=np.zeros(np.round(np.array(M.shape)/super_pix_size).astype(np.int32)).astype(np.complex64) 
        
        #When a spot location is not given (image center), a hologram is displayed
        #and an image is taken to determine the location within the frame
        
        if image_center is None:
            H=hologram(farFieldCenter=self.FF_center,screen_id_main=self.screen_id_main) #Hologram for the coordinate defined by the far field center
            H.Holo=np.exp(1j*np.angle(H.Holo)*0.1) #The hologram efficiency is reduced to avoid camera saturation
            H.display() #Hologram is displayed and image is acquired
            self.cam.cam.open()
            self.cam.acqstart()
            sleep(0.5)
            frame=self.cam.wait_and_capture(1)
            self.cam.acqstop()
            self.cam.cam.close()
            cv2.destroyAllWindows()
            fig=plt.figure()
            plt.imshow(frame)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.draw() #Image is diplayed for evaluation by user for 15 seconds
        
            sleep(15)
            frame=frame*(frame>0.5*np.max(frame)).astype(np.int32)
            image_center=ndi.center_of_mass(frame) #spot location is determined as the center of mass of the image
            print(image_center)
  
       
    ##START DISPLAY WINDOW FOR HOLOGRAMS   
        player=holo_player(screen_id_main=self.screen_id_main, DMD=DMD)
        player.player_start()
    ##CREATE A DASHBOARD
        dash=dashboard(end_Q=self.end_gen_Q)
        monitor_Q=dash.dash_Q

    ##DEFINE THREADS
        holo_gen = Thread(target=self.super_pixel_scan,kwargs=
                          {"super_pixel_size": super_pix_size, "nshifts": phase_shifts,'shift': shift, 'scan_us':scan_us})       
        display = Thread(target=self.display_capture, kwargs={"player": player})
        processing=Thread(target=self.frame_processing, kwargs={"monitor_Q":monitor_Q})
        wfc_process=Thread(target=self.process_wfc, kwargs={"center": image_center})
        
    ##START THREADS           
        holo_gen.start()
        display.start()
        processing.start()
        sleep(1)
        wfc_process.start()
    ##DASHBOARD START    
        dash.dash_run()
            
        
    ## Wait for threads to finish
        holo_gen.join(timeout=5)
        display.join(timeout=5)
        processing.join(timeout=5)
        wfc_process.join(timeout=5)
 
        player.player_stop()
        # self.process_wfc(image_center)
        M=self.WFC        
        self.WFC=self.interpolate_complex_field(M,self.slm_size)
        self.save_obj(M,file_name)
        
        if not(self.in_cam is None):
            self.cam=self.out_cam
        
        if self.forced_end==True:
            M=None
            image_center=None
            
        return M, image_center

    def save_obj(self, M, file_name):
        open_file = open(file_name, "wb")
        M=pkl.dump(M,open_file)
        open_file.close()    
        return None
    
    def load_WFC(self, file_name, lp_thresh=None, nearest=False):
        open_file = open(file_name, "rb")
        M=pkl.load(open_file)
        open_file.close()    
        self.WFC=self.interpolate_complex_field(M,self.slm_size,lowpass_thresh=lp_thresh,nearest=nearest)
        return M

    def run_mask_scan(self, mask, nbin=1, acc_size=100):
        """
        SCANS AN AREA IN THE FOURIER PLANE OF THE SLM. THE AREA IS DEFINED BY A MASK

        Parameters
        ----------
        mask : TYPE
            DESCRIPTION.
        nbin : TYPE, optional
            DESCRIPTION. The default is 1.
        acc_size : TYPE, optional
            DESCRIPTION. The default is 100.

        Returns
        -------
        None.

        """
       
        self.clear_Q() #Make sure the Qs are clean before start
        
    ##START DISPLAY WINDOW FOR HOLOGRAMS 
        player=holo_player(screen_id_main=self.screen_id_main)
        player.player_start()


    ##DEFINE THREADS
        
        holo_gen = Thread(target=self.scan_area,kwargs={"radius": self.core_size_px,"initial_scan": False,"nbin":nbin,"scan_mask": mask})
        # holo_gen = multiprocessing.Process(target=self.scan_area,kwargs={"radius": self.core_size_px,"monitor_Q":monitor_Q})   
        display = Thread(target=self.display_capture, kwargs={"player": player})
        processing=Thread(target=self.frame_processing)
    ##CREATE FIGURE FOR PLOTTING
        fig=plt.figure()
        ax=plt.axes()
        acc=np.zeros(acc_size)
    ##START THREADS
         
        holo_gen.start()
        display.start()
        processing.start()
        k=0
        pl, =ax.plot(acc)
        while True:
            try: 
                element=self.field_Q.get(timeout=1)
                I=element["Intensity"]
                acc[np.mod(k,acc_size)]=I
            
                     
                              
            
                pl.set_ydata(acc)
                ax.set_ylim(np.min(acc),np.max(acc))
                ax.set_xlim(0,acc_size)
                ax.set_xticks([np.mod(k,acc_size)])
                

                fig.canvas.draw()
                fig.canvas.flush_events()
                
                
                if not(plt.fignum_exists(fig.number)):
                    break    
            
            except Empty: None
            
            try:
                done=self.done_Q.get_nowait()
                if done:
                    try: self.stop_Q.put_nowait(False)
                    except: None 
            except:
                done=False
            k+=1                
            
   
        
    ## Wait for threads to finish
        holo_gen.join()
        display.join()
        player.player_stop()
        processing.join()
        
    ##GET IMAGE OF INPUT MODES    
        M=self.process_initial_scan()
        self.input_mode_image=M
        
        return None
    
    def find_ff_mask(self):
        self.phase_shift_n=5
        self.WFC_on=False
        self.reference_on=True      
        ref_loc=None
        self.drift_ref=[]
        self.farfield_mask=[]
        self.clear_Q() #Make sure the Qs are clean before start        
    ##START DISPLAY WINDOW FOR HOLOGRAMS           
        player=holo_player(screen_id_main=self.screen_id_main)
        player.player_start()
    ##CREATE A DASHBOARD
        dash=dashboard()
        monitor_Q=dash.dash_Q
        
    ##DEFINE MODES TO BE SCANNED
    
        self.input_modes=np.zeros_like(self.input_mode_image)
        s=np.array(self.input_modes.shape)
        s=np.round(s/2).astype(np.int32)
        self.input_modes[s[0],s[1]]=1

    ##DEFINE THREADS
        holo_gen = Thread(target=self.scan_area,kwargs={"radius": self.core_size_px,
                                                        "initial_scan": False, "nShifts":self.phase_shift_n,"refLoc":ref_loc})       
        display = Thread(target=self.display_capture, kwargs={"player": player,"monitor_Q":None,"WFC_on":self.WFC_on,"dual_polarization":True})
        processing=Thread(target=self.frame_processing, kwargs={"monitor_Q":monitor_Q,"cap_Q":self.capture_Q,"field_Q":self.field_Q })
        processing_b=Thread(target=self.frame_processing, kwargs={"monitor_Q":None,"cap_Q":self.capture_Q_b,"field_Q":self.field_Q_b })
        # tm_calc=Thread(target=self.process_TM)
        
    ##START THREADS  
        holo_gen.start()
        sleep(2)
        display.start()
        processing.start()
        processing_b.start()
        # tm_calc.start()
    ##DASHBOARD START    
        dash.dash_run()  
        
    ##Wait for threads to finish
        holo_gen.join()
        display.join()
        player.player_stop()
        processing.join() 
        processing_b.join()
    ## Get the resulting fields
        elements=[self.field_Q.get_nowait(),self.field_Q_b.get_nowait()]
        fig_names=['FAR FIELD MASK OUTPUT POLARIZATION A','FAR FIELD MASK OUTPUT POLARIZATION B']
        
        for k, element in enumerate(elements):
            drift_ref=element['Field']
            #Fourier plane
            drift_fft=ft.ifftshift(ft.fft2(ft.fftshift(drift_ref))) 
            
            #A hologram object is defined to use the circular mask function (no hologram displayed)
            H=hologram(SLMSize=drift_fft.shape)
            rad=np.round(drift_fft.shape[0]*0.15).astype(np.int32)       
            center_mask=np.logical_not(H.circular_mask(rad))
            
            #The lower spatial frequencies are masked
            M=np.abs(drift_fft*center_mask)
            M=self.normalize(M)
            
            #The image of the amplitude is softened using median and gaussian filter (to calculate the maximum later)
            M[np.logical_not(center_mask)]=0.01
            M=ndi.median_filter(M,(8,8))
            M=ndi.gaussian_filter(M, 5)
            M=self.normalize(M)
            F=plt.figure()
            F.canvas.manager.set_window_title(fig_names[k])
            
            plt.subplot(1,2,1)
            plt.imshow(M)
            
            #The maximum of the image is found (should be very close to the center of the desired mask)
            ind=np.argmax(M)
            [y,x]=np.unravel_index(ind, M.shape)
            print([y,x])
            rad=np.round(M.shape[0]*0.25).astype(np.int32) 
            
            #The mask for the farfield is generated
            field_mask=H.circular_mask(rad,center=(y,x))
            M=M*field_mask
            M=M>0.10
            area=np.sum(M)
            rad=np.sqrt(area/np.pi).astype(np.int32)
            field_mask=H.circular_mask(rad,center=(y,x))
            self.farfield_mask.append([field_mask,(y,x)])
        
            #A reference image for the drift is taken (all drift calculations are offset by this image)
            drift_fft=drift_fft*field_mask
            s=np.array(drift_fft.shape)/2
            s=s.astype(np.int32)
            drift_fft=np.roll(drift_fft,-y+s[0],0)
            drift_fft=np.roll(drift_fft,-x+s[1],1)
            drift_ref=ft.fftshift(ft.ifft2(ft.ifftshift(drift_fft)))
            self.drift_ref.append(drift_ref)       
            plt.subplot(1,2,2)
            plt.imshow(np.angle(drift_ref))
        
        self.WFC_on=False
        
        
        return None
    
    def run_calibration(self, intensity_thresh=0.1, in_modes=None,reel_filename=None, save_reel=False):
        self.TM=[]
        self.phase_shift_n=3
        self.reference_on=True      
        ref_loc=None
        self.clear_Q() #Make sure the Qs are clean before start  
        self.stop_Q.put(True)
        self.WFC_on=False
        
        
        # if reel_filename is None:
        #     self.WFC_on=True
        # else:
        #     self.WFC_on=False
        
    ##START DISPLAY WINDOW FOR HOLOGRAMS           
        player=holo_player(screen_id_main=self.screen_id_main)
        player.player_start()
    ##CREATE A DASHBOARD
        dash=dashboard(end_Q=self.end_gen_Q)
        monitor_Q=dash.dash_Q
        
    ##DEFINE MODES TO BE SCANNED
        if in_modes is None:
            self.input_modes=self.normalize(self.input_mode_image)>intensity_thresh
        else:
            self.input_modes=in_modes

    ##DEFINE THREADS
        if save_reel:
            filename=None
        else:
            filename=reel_filename
            
    
        holo_gen = Thread(target=self.scan_area,kwargs={"radius": self.core_size_px,
                                                        "initial_scan": False, 
                                                        "nShifts":self.phase_shift_n,
                                                        "refLoc":ref_loc,
                                                        "hologram_reel_filename":filename,
                                                        "FFT_Q":self.FFT_Q,
                                                        "apply_WFC":False})      
        if save_reel:
            saver=Thread(target=self.holo_reel_maker, kwargs={"filename":reel_filename})
            holo_gen.start()
            saver.start()
            
            saver.join()
            holo_gen.join()
            
            
        else:
            display = Thread(target=self.display_capture,
                             kwargs={"player": player,
                                     "monitor_Q":monitor_Q,
                                     "WFC_on":False,"dual_polarization":True})
            
            processing=Thread(target=self.frame_processing, 
                              kwargs={"monitor_Q":None,
                                      "cap_Q":self.capture_Q,
                                      "field_Q":self.field_Q })
            
            processing_b=Thread(target=self.frame_processing, 
                                kwargs={"monitor_Q":monitor_Q,
                                        "cap_Q":self.capture_Q_b,
                                        "field_Q":self.field_Q_b })
            
            tm_calc=Thread(target=self.process_TM, 
                           kwargs={"field_Q":self.field_Q,
                                   "farfield_mask": self.farfield_mask[0],
                                   "drift_ref":self.drift_ref[0]})
            sleep(0.05)
            
            tm_calc_b=Thread(target=self.process_TM, 
                           kwargs={"field_Q":self.field_Q_b,
                                   "farfield_mask": self.farfield_mask[1],
                                   "drift_ref":self.drift_ref[1]})
            FFT_proc=mp.Process(target=FFT_process, kwargs={'Q': self.FFT_Q})    
        ##START THREADS
            FFT_proc.start()
            holo_gen.start()
            display.start()
            processing.start()
            processing_b.start()
            sleep(1)
            tm_calc.start()
            sleep(1)
            tm_calc_b.start()
        ##DASHBOARD START    
            dash.dash_run()  
            
        ##Wait for threads to finish
            holo_gen.join()
            display.join()
            player.player_stop()
            processing.join()
            processing_b.join()
            tm_calc.join()
            tm_calc_b.join()
            
            self.FFT_Q[0].put(None)
            self.WFC_on=False
            self.reference_on=False
        return None
    

    def normalize(self,M):
        
        M=M-np.min(M)
        if np.max(M)>0:
            M=M/np.max(M)            
        return M
    
    def scan_area(self,radius=5, 
                  initial_scan=True, 
                  scan_mask=None,
                  nShifts=None, 
                  refLoc=None, 
                  offset=0, maxL=1,
                  monitor_Q=None, 
                  nbin=1, 
                  FFT_Q=None, 
                  hologram_reel_filename=None, 
                  apply_WFC=False):
        """
        

        Parameters
        ----------
        radius : Int. Radius of a circumscribed circle, used only for initial scan
            The default is 5.
        initial_scan : Bool, true for an initial scan
            The default is True.
        nShifts : Number fo phase shifts.
            DESCRIPTION. The default is None.
       refLoc : TYPE, optional
            DESCRIPTION. The default is None.
       offset : TYPE, optional
            DESCRIPTION. The default is 0.
        maxL : TYPE, optional
            DESCRIPTION. The default is 1.
        screen_id : TYPE, optional
            DESCRIPTION. The default is 1.
        intensity_thresh : TYPE, optional
            DESCRIPTION. The default is 0.1.

        Returns
        -------
        None.

        """
        sSize=self.slm_size
        center=self.FF_center
        Q=self.display_Q
        screen_id=self.screen_id_main
        
        #Load a hologram for dumping residual power due to WFC
        dump_loc=(350,900)
        
        try:
            filename=r'C:\TM_meas\HOLO_REELS\dump.pkl'
            file=open(filename,'wb')
            H=pkl.load(file)
            file.close()
            holo_dump=hologram(precalc_holo=H,farFieldCenter=dump_loc)
        except:
            holo_dump=hologram(farFieldCenter=dump_loc)

        # screen_main = screeninfo.get_monitors()[screen_id_main]

        if hologram_reel_filename is None:
            file=None
        else:
            file=open(hologram_reel_filename,'rb')
        
        if initial_scan: #Initialize variables if the scan is an initial scan
            [X,Y]=np.meshgrid(np.arange(radius*2),np.arange(radius*2))
            aSize=X.shape
            R=np.sqrt((X-0.5*aSize[1])**2+(Y-0.5*aSize[0])**2)
            scanUs=R<radius
            scanUs=scanUs[0::nbin,0::nbin]
            Y=Y+center[0]-radius
            X=X+center[1]-radius

            Y=Y[0::nbin,0::nbin]
            X=X[0::nbin,0::nbin]
            aSize=X.shape
            self.Y_mesh=Y
            self.X_mesh=X
            self.result_shape=aSize
            self.scan_us=scanUs
        elif not(scan_mask is None):
            [X,Y]=np.meshgrid(np.arange(scan_mask.shape[1]),np.arange(scan_mask.shape[0]))
            
            scanUs=scan_mask
            Y=Y+center[0]-(scan_mask.shape[0]/2)
            X=X+center[1]-(scan_mask.shape[1]/2)
            Y=Y[0::nbin,0::nbin]
            X=X[0::nbin,0::nbin]
            scanUs=scanUs[0::nbin,0::nbin]
            aSize=X.shape
            self.result_shape=aSize
            
            
        else: #Load variables when an initial scan was performed
            # scanUs=np.ones(aSize)>0
            scanUs=self.input_modes
            Y=self.Y_mesh
            X=self.X_mesh
            aSize=X.shape
            self.result_shape=aSize
            

        
        driftReference=hologram(SLMSize=sSize,farFieldCenter=(center[0],center[1]),screen_id_main=screen_id)
        driftReference.Holo+=self.reference_holo.Holo
        
        pulldown=driftReference.holo_copy()
        pulldown.Holo=1*np.exp(1j*np.random.rand(pulldown.Holo.shape[0],pulldown.Holo.shape[1])*2*np.pi)
        # G=driftReference
        
        # If a an internal reference location is provided this generates the reference hologram 
        #(NOT IN USE, PLACE HOLDER FOR INTERNAL REFERENCE CAL)
        if not(refLoc is None):          
             internal_reference=None
        else:
             internal_reference=None
 
        end_all=False
        
        while True: #Main loop (runs until all holograms have been generated or a break is found due to user interaction)

            
            for k in np.arange(aSize[0]):   #Sweeps y axis of the input pixel mask
                if end_all:#Checks if loop should be exited due to termination
                    break
                for m in np.arange(aSize[1]): #Sweeps x axis of the input pixel mask
                    try:
                        end_all=self.end_gen_Q.get_nowait()#checks if a termination signal exist in the Q
                    except Empty:
                        end_all=False
                        
                    if end_all:
                        self.clear_Q()# If a termination signal is found this clears all the Qs being used for the scan
                        self.forced_end=True
                        break
                    
                    if scanUs[k,m]:#Checks if the current input pixel has been selected for measurment
                        if file is None:
                            A=None
                        else:
                            A=pkl.load(file) #if a hologram reel (precalculated holograms) is provided it gets loaded here
                            
                        G=hologram(precalc_holo=A,SLMSize=sSize,farFieldCenter=(Y[k,m],X[k,m]),screen_id_main=screen_id,FFT_Q=FFT_Q)
                        ## Apply wavefrontcorrection to the hologram:
                        if apply_WFC:
                            G.Holo=G.apply_WFC(G.Holo, self.WFC, holo_dump)
                        
                        #Load reference hologram
                        G.reference_hologram=self.reference_holo
                       
        ##CASE: PHASE SHIFTING--------------------------------------------------------
                        if not(nShifts is None): 
                            ps_holos=G.batch_holoPhaseShift(nShifts)
                            # for p in np.arange(nShifts):                                                                        
                            #     H=G.holoPhaseShift(nShifts,p)
                            for p,H in enumerate(ps_holos):
                                
                ## If an internal reference is porvided, add it after phase shifting the mode hologram
                                if not(internal_reference is None):
                                    H.Holo=H.Holo+internal_reference.Holo
                ## CASE: DISPLAY HOLOGRAM DIRECTLY------------------------------------
                                if Q is None:
                                    H.display(apply_reference=True, amp_modulation=0.1)
                ## CASE: PUT HOLOGRAM IN A QUEUE---------------------------------------
                                else:

                                    if p==0: #Put drift reference into Q for every input mode
                                    
                                        for v in np.arange(2):
                                            #Repeat drift reference (effectively doubles 
                                            #the rise time to allow for better stability of the drift correction)
                                            sendMe={"HOLOGRAM":driftReference,"Subscripts":(k,m),
                                                    "Phase Index":-1}
                                            self.display_Q.put(sendMe)
                                            
                                    sendMe={"HOLOGRAM":H,"Subscripts":(k,m),
                                            "Phase Index":p}
                                    self.display_Q.put(sendMe)
        ##CASE: SINGLE PHASE----------------------------------------------------------
                        else:
                ## CASE: DISPLAY HOLOGRAM DIRECTLY------------------------------------
                                if Q is None:
                                    G.display(apply_reference=True, amp_modulation=0.1)
               ## CASE: PUT HOLOGRAM IN A QUEUE---------------------------------------
                                else:
                                   
                                    sendMe={"HOLOGRAM":G,"Subscripts":(k,m),
                                                "Phase Index":None}
                                    Q.put(sendMe)
                                    
            try:
                print("waiting for stop")
                self.done_Q.put(True)
                stop_signal=self.stop_Q.get(timeout=5)
                print(stop_signal)
                if stop_signal:
                    print("HOLO GEN END")
                    break
            except:
                print("HOLO GEN END")
                break
                        
        self.display_Q.put(None)
        if not(file is None):
            file.close()
            
        return None
 
    
    def super_pixel_scan(self, super_pixel_size=50, screen_id=1, nshifts=1, shift=None, scan_us=None):
        #This method generates an aperture in the shape of a window that is moved in steps accross the grating
        #in a raster scan fasion. A series of phase shifts given by nshifts is produced for each step. 
        #Mainly used for wavefront correction. 
        s_size=self.slm_size
        center=self.FF_center
        Q=self.display_Q
        self.phase_shift_n=nshifts
        
        m_size=np.round(np.array(s_size)/super_pixel_size)#Size of an intermediate mask where each pixel represents one super pixel
        m_size=m_size.astype(np.int32)#Size must me integer for later operations
        center_holo=hologram(SLMSize=s_size,farFieldCenter=(center[0],center[1])) #This is the central hologram that will be masked
        dump=hologram(SLMSize=s_size,farFieldCenter=(300,1295))# A location to dump any residual power
        [M_ref, ind]=self.super_pixel_mask(super_pixel_size) #The mask for the "reference superpixel" is generated
        H_shifts=[]#List initialization for the holograms produced for phase shifting
        ref=np.exp(1j*np.angle(center_holo.Holo)*M_ref)+dump.Holo #The reference superpixel mask is applied

        if nshifts==1: #Phase shifted holograms are generated
            ref=ref*0
        amp=[]    
        for p in np.arange(nshifts):
            H_shifts.append(center_holo.holoPhaseShift(nshifts,p))
            amp_corr=(np.cos(2*np.pi*(p)/nshifts)+1)/2
            amp.append(1.199*amp_corr**3-2.778*amp_corr**2+2.576*amp_corr+0.0067)
        # amp=np.flip(amp)
        amp=amp/np.max(amp)
        print(amp)
        
        sleep(0.5)
            
        
        
        
        for k in np.arange(np.prod(m_size)):
            try:
                end_all=self.end_gen_Q.get_nowait() #Check if the loop should be terminated due to external input
            except Empty:
                end_all=False
                
            if end_all:
                self.clear_Q()
                self.forced_end=True
                break
            
            
            [M, b]=self.super_pixel_mask(super_pixel_size,k.astype(np.int32),shift=shift)
            
            
            p=0
            for H in H_shifts: #Mask the phase shifted holograms and put into Queue
                
                
                
                if k==ind:
                    # C=center_holo.holo_copy()
                    # C.Holo=M*amp[p]*np.abs(C.Holo)*np.exp(1j*np.angle(C.Holo)*M*amp[p])
                    im_ref=True
                   
                else:
                    im_ref=False
                    
                C=H.holo_copy()
                C.Holo=np.exp(1j*np.angle(H.Holo)*M)+ref
                m=np.unravel_index(k.astype(np.int32), m_size)
                if nshifts==1:
                    pindex=None
                else:
                    pindex=p
                
                p+=1
                
                if scan_us is None:
                    sendMe={"HOLOGRAM":C,"Subscripts":m, "Shape":m_size,
                        "Phase Index":pindex, "Total shifts": nshifts, "Reference": im_ref}
                else:
                    if np.ravel(scan_us)[k]:
                        sendMe={"HOLOGRAM":C,"Subscripts":m, "Shape":m_size,
                            "Phase Index":pindex, "Total shifts": nshifts, "Reference": im_ref}
                    else:
                        sendMe={"HOLOGRAM":None,"Subscripts":m, "Shape":m_size,
                            "Phase Index":pindex, "Total shifts": nshifts, "Reference": im_ref}
                try:
                    Q.put(sendMe,timeout=10)
                except:
                    print('Display Q full')
                    self.end_Q()
                    break
        print("SUPERPIXEL GEN ENDED")
        Q.put(None)
        
        
        

    
    def super_pixel_mask(self, super_pixel_size,index=None,shift=None):
    #This function generates a square mask on the slm. Size is given by the super_pixel_size parameter
    #index is the linear index in a matrix of size SLMSize/super_pixel_size. Shift performes a circular shift
    #in both axis by the given number of pixels
        s=np.array(self.slm_size)/(super_pixel_size)
        s=np.round(s)
        s=s.astype(np.int32)
        M=np.zeros(s)
        if index is None:
            ind=np.round(s/2)
            ind=ind.astype(np.int32)
            M[ind[0],ind[1]]=1
            ind=np.ravel_multi_index(ind, s)
            M=im.resize(M,self.slm_size,order=0)
        else:
            ind=np.unravel_index(index, s)        
            M[ind]=1
            M=im.resize(M,self.slm_size,order=0)
            if not(shift is None):
                M=np.roll(M, shift,0)
                M=np.roll(M, shift,1)
            
        return M, ind
        
            
        
        
    def display_capture(self, player=None, monitor_Q=None, WFC_on=False, dual_polarization=False):
        """
        This function displays Hologram objects found in the display Q of the current calibration object. 
        After displaying, images are acquired using a cam_capture object (holo_tools.py)
        The function then puts the acquired frames into a queue for processing. If two output polarizations
        are being used each frame goes to a different queue (self.capture_Q and self.capture_Q_b). 
        The queues are defined in the initialization of the calibration object.
        
        
        Parameters
        ----------
        player : Hologram player object (defined in the holo_tools module)
        monitor_Q : A queue to a monitoring GUI (legacy, UNUSED REMOVE IN FUTURE)
        WFC_on : Use wavefront correction during display 
        dual_polarization : Specify if two output polarizations are being acquired

        -------
        None.

        """
        Q=self.display_Q 
        rise_time=self.rise_time
        frame_size=None
        
        #Make sure camera is initialized
        cam=self.cam
        cam.cam.open()
        cam.acqstart()
        
        #Load WFC if needed
        if WFC_on:
            WFC=self.WFC 
        else:
            WFC=None
        
        #Display consumer loop
        while True:
            try: 
                q_item=Q.get(timeout=q_timeout)
            except Empty: 
                #Timeout to avoid hanging in case the hologram generation thread isn't working
                self.capture_Q.put(None)
                if dual_polarization:
                    self.capture_Q_b.put(None)
                print("DISPLAY QUEUE TIMEOUT")
                self.end_gen_Q.put(True)
                self.end_Q()
                cam.acqstop()
                break
            
            if q_item is None: #End loop if a None has been put in the Q
                break
            # Display the hologram---------------------------------------------
            if not(q_item['HOLOGRAM'] is None):
                q_item['HOLOGRAM'].display(player=player,apply_reference=self.reference_on,WFC=WFC,t_pullup=None,pullup_level=self.pullup_level)
                b=1
                rise_time=self.rise_time
            else:
                b=0
                rise_time=1
            # t_pull=np.int32(self.t_pullup*1000)
            # Check if two output polarization images are required-------------
            if dual_polarization:
                #Aquire the two output images----------------------------------
                #These are defined in the cam_capture object
                frames=cam.multi_wait_and_capture(rise_time)
                frames=[F*b for F in frames] #MAKES FRAME 0 IF HOLOGRAM IS NONE (IN CASE THE HOLOGRAM IS BEING IGNORED)
                frame=frames[0].astype(np.float32)
                q_item_b=copy.copy(q_item)
                q_item.update({'Capture': frame})
                frame=frames[1].astype(np.float32)
                q_item_b.update({'Capture': frame})
                
                try:
                #Send the images to their corresponding Qs
                    self.capture_Q.put(q_item, timeout=5)
                    self.capture_Q_b.put(q_item_b, timeout=5)
                except:
                    print("CAPTURE Q FULL, ENDING")
                    self.end_Q()
                    cam.acqstop()
                    break
            
                
            
                del q_item
                del q_item_b
                
            else:
                #If only one output polarization is required-------------------
                
                if q_item['HOLOGRAM'] is None:
                    if not(frame_size is None):
                        frame=0.01+np.zeros(frame_size).astype(np.float32)
                    else:
                        frame=cam.wait_and_capture(10)
                        frame_size=frame.shape
                        frame=0.01+frame*0
                else:
                    frame=cam.wait_and_capture(rise_time) #MAKES FRAME 0 IF HOLOGRAM IS NONE (IN CASE THE HOLOGRAM IS BEING IGNORED)
                frame=frame.astype(np.float32)
                q_item.update({'Capture': frame})
            
                try:
                    self.capture_Q.put(q_item, timeout=5)
                except:
                    print("CAPTURE Q FULL, ENDING")
                    self.end_Q()
                    cam.acqstop()
                    break
            
                del q_item               
            
        
        self.capture_Q.put(None)
        if dual_polarization:
            self.capture_Q_b.put(None)
        
        cam.acqstop()
        cam.cam.close()
        print("DISPLAY ENDED")
        return None
    
    def frame_processing(self, monitor_Q=None,cap_Q=None,field_Q=None ):    
        print('Processing started')
        
        if cap_Q is None:
            cap_Q=self.capture_Q
        if field_Q is None:
            field_Q=self.field_Q
       
        Drift=None
        
        npshift=self.phase_shift_n
        shifDen=npshift/2
        firstRun=True
        
        while True:
            try:
                capture=cap_Q.get(timeout=q_timeout)

                
            except Empty:
                print("Frame Processing time out")
                self.end_gen_Q.put(True)
                self.end_Q()
                break
            if capture is None:
                break
            try:
                frame_size=capture['Shape']
                M=np.zeros(frame_size)
            except KeyError:
                frame_size=self.result_shape
                M=np.zeros(frame_size)
                
            M[capture['Subscripts']]=1
            sigma=np.array(frame_size)
            sigma=sigma/20
            M=ndi.gaussian_filter(M,sigma)
            
            m=capture["Phase Index"];
            I=capture["Capture"]
            
            
            if firstRun:
                firstRun=False
                acc_image=I.astype(np.float64)
            else:  
                acc_image=acc_image+I.astype(np.float64)
            
            I=I.astype('D')
            
            if m is None:
                # mval=np.max(capture["Capture"])
                # use_us=capture["Capture"]>(0.7*mval)
                # totalI=np.average(capture["Capture"][use_us])
                totalI=np.max(capture["Capture"])
                
                sendMe={"Intensity":totalI,
                        "Subscripts":capture["Subscripts"],"Drift":None}  
                
                if not(monitor_Q is None):
                    monitor=[{'Axis': 2,'Image': self.normalize(np.transpose(M))},
                             {'Axis': 1,'Image': self.normalize(np.log(np.transpose(capture["Capture"])+1))}]
                             #{'Axis': 0,'Image': self.normalize(np.transpose(np.angle(capture['HOLOGRAM'].Holo[0:99,0:99])))}]
                    try:
                        monitor_Q.put(monitor,block=False)
                    except:
                        print('Monitor Full')
                        # monitor_Q.queue.clear() 
                        
                try:
                    field_Q.put(sendMe, timeout=5)
                except:
                    self.end_Q()
                    print("RESULT Q FULL, ENDING")
                    break
                del capture['HOLOGRAM']  
                del capture
                
              
            elif m==-2:
                print("SLM PULLDOWN")
            elif m==-1:
                Drift=I  
                
            elif m==0:
                acc=I.astype(np.csingle)*np.exp(0*1j)
                # print('ACC INIT')
            elif m<(npshift-1):
                acc=acc+(I.astype(np.csingle)*np.exp(1j*m*np.pi/shifDen))
                # print('ACCUMULATING')
            elif m==(npshift-1):
                acc=acc+(I.astype(np.csingle)*np.exp(1j*m*np.pi/shifDen))
                acc=acc/npshift
                
                # print(np.max(I).astype(np.int32))
                
                sendMe={"Field":acc,"Subscripts":capture["Subscripts"],
                        "Drift":Drift}
                
                if not(monitor_Q is None):
                    # if Drift is None:
                    #     Drift=np.abs(acc)
                        
                    monitor=[{'Axis': 2,'Image': self.normalize(np.transpose(M))},
                             {'Axis': 1,'Image': self.normalize(np.abs(acc))},
                             {'Axis': 0,'Image': self.normalize(np.angle(acc))}]
                    try:
                        monitor_Q.put(monitor,block=False)
                    except:
                        print('Monitor Full')
                        # monitor_Q.queue.clear() 
                field_Q.put(sendMe)
                del capture['HOLOGRAM']  
                del capture
                
                
                # print('Field calculation succesful')
                # A=np.angle(acc)
                # A=A-np.min(A)
                # A=A/np.max(A)
                # window_name='Field'
                # cv2.namedWindow(window_name)
                # cv2.moveWindow(window_name,1500,0)
                # cv2.imshow(window_name,A)
                # cv2.waitKey(1)  
            
        field_Q.put(None)
        self.accumulated_image=acc_image
        print("FRAME PROCESSING ENDED")
        if not(monitor_Q is None):
            monitor_Q.put(None)
        return None
    
    def process_wfc(self,center, ff_size=(500,500),pow_thresh=0.5):
        Q=self.field_Q
        ff_size=np.array(ff_size)
        center=(ff_size/2)-center
        center=np.round(center)
        center=center.astype(np.int32)
        while True:
            try:
                element=Q.get(timeout=q_timeout)                
            except Empty:
                print("WFC Processing time out")
                self.end_Q()
                break
            if element is None:
                break
            
            
            M=np.zeros(ff_size)
            M=M.astype(np.complex64)
            field=element["Field"]
            M[0:field.shape[0],0:field.shape[1]]=field
            M=np.roll(M,tuple(center),(0,1))
            
            M=self.findCorrection(M)
            
            amp=np.abs(M).flatten()
            thresh_pix=(self.normalize(amp)>pow_thresh)
            
            phase=np.angle(M).flatten()
            phase=phase[thresh_pix]
            phase=np.unwrap(phase)
            m_phase=np.average(phase)
            m_amp=np.average(amp[thresh_pix])
                       
            
            self.WFC[element["Subscripts"]]=m_amp*np.exp(1j*m_phase)            
            self.wfc_Q.put(M,timeout=1)
            del element
            
        self.wfc_Q.put(None,timeout=1)
        print("WFC END")
        return None
    
    def findCorrection(self, field, mask_size=20):
        farField=ft.fftshift(ft.fft2(ft.fftshift(field)))
        maxI=np.argmax(np.abs(farField))
        maxI=np.unravel_index(maxI,field.shape)
        maxI=np.round(maxI-np.array(field.shape)/2)
        maxI=maxI.astype(np.int16)
        farField=np.roll(farField,-maxI[0],0)
        farField=np.roll(farField,-maxI[1],1)
        mask=self.createMask(farField.shape,mask_size)
        nearField=ft.fftshift(ft.ifft2(ft.fftshift(farField*mask)))
        fAbs=np.abs(nearField)
        mAbs=np.max(fAbs)
        fAbs=fAbs/mAbs
        fAbs=fAbs>0.1
        nearField=nearField*fAbs.astype(np.int16)
        # if mAbs<10:
        #     nearField=nearField*0
        return nearField
    
    def createMask(self, mSize,maskRad):
        mSize=np.flip(mSize)
        M=np.meshgrid(np.arange(mSize[0]),np.arange(mSize[1]))
        M[0]=M[0]-mSize[0]/2
        M[1]=M[1]-mSize[1]/2
        R=np.sqrt(M[0]**2+M[1]**2)
        mask=R<maskRad
        mask=mask.astype(np.int8)
        return mask
            
    def process_TM(self,field_Q=None,farfield_mask=None,drift_ref=None):
        in_mod=self.input_modes
        in_mod=np.sum(in_mod.astype(np.int32))
        
        
        
        if (field_Q is None) or (farfield_mask is None) or (drift_ref is None) :            
            field_Q=self.field_Q
            farfield_mask=self.farfield_mask[0]
            drift_ref=self.drift_ref[0]
            
        tm_size=(np.prod(farfield_mask[0].shape),in_mod)
        print(tm_size)
        TM=trans_matrix(tm_size)
        
        while True: 
            try:
                element=field_Q.get(timeout=q_timeout)
            except Empty:
                print("Frame Processing time out")
                self.end_Q()
                break
            
            if element is None:
                break
            
            subs=element["Subscripts"]
            mode_coord=(self.Y_mesh[subs],self.X_mesh[subs])
            drift=self.process_drift(drift_pattern=element['Drift'],drift_ref=drift_ref, farfield_mask=farfield_mask)
            F=element['Field']
            F=F*np.exp(-1j*drift)
            F=self.center_farfield(F, farfield_mask)
            # save_me={'Field':F.copy(),'Drift':drift}
            # self.mode_Q.put_nowait(save_me)
            TM.write_new_mode(F, mode_coord)
        
        self.TM.append(TM)
            
    def process_drift(self, drift_pattern=None, drift_ref=None,farfield_mask=None):
        
        if drift_ref is None:
            drift_ref=self.drift_ref[0]
            farfield_mask=self.farfield_mask[0]
        
        ##Dummy function
        if drift_pattern is None:
            d=(0,)
        else:
            drift=self.center_farfield(drift_pattern, farfield_mask)
            drift=drift-drift_ref
            
            drift_mask=self.normalize(np.abs(drift))>0.9
            d=np.mean(np.unwrap(np.angle(drift[drift_mask])))
            
        return d
    
    def center_farfield(self, M, farfield_mask):
        
        mask=farfield_mask[0]
        y=farfield_mask[1][0]
        x=farfield_mask[1][1]
        
        F=ft.ifftshift(ft.fft2(ft.fftshift(M)))
        F=F*mask
        s=np.array(F.shape)/2
        s=s.astype(np.int32)
        F=np.roll(F,-y+s[0],0)
        F=np.roll(F,-x+s[1],1)
        F=ft.fftshift(ft.ifft2(ft.ifftshift(F)))
        
        return F
        
        

    def process_initial_scan(self, super_pixel_size=None):
        
        if super_pixel_size is None:
            M=np.zeros(self.result_shape)
        else:
            s=np.array(self.slm_size)/np.array(super_pixel_size)
            s=np.round(s)
            s=s.astype(np.int32)
            M=np.zeros(s)

        
        
        while True: 
            try:
                element=self.field_Q.get(timeout=15)
            except Empty:
                print("Frame Processing time out")
                self.end_Q()
                break
            
        
            
            if element is None:
                break
            M[element['Subscripts']]=element["Intensity"]                        
        return M
    
    def interpolate_complex_field(self,field,out_size,lowpass_thresh=None, shift=True, nearest=False):
        self.accumulated_image=field
        
        if nearest:
            print("")
            out_field_r=im.resize(np.real(field),out_size,order=0)
            out_field_i=im.resize(np.imag(field),out_size,order=0)
            out_field=out_field_r+1j*out_field_i
            
        else:
            pad_size=np.array(out_size)-np.array(field.shape)
            center=np.array(out_size)/2
            [X,Y]=np.meshgrid(np.arange(out_size[1]),np.arange(out_size[0]))
            Y=Y-center[0]
            X=X-center[1]
            R=np.sqrt(X**2+Y**2)
    
            mabs=np.abs(field)
            mabs=ndi.median_filter(mabs,size=2)
    
            mod=np.mod(pad_size,2).astype(np.int32)
            pad_size=np.floor(pad_size/2).astype(np.int32)
    
            pad_vert=(pad_size[0],pad_size[0]+mod[0])
            pad_lat=(pad_size[1],pad_size[1]+mod[1])
            
            if shift:
                fm=ft.ifftshift(ft.fft2(ft.fftshift(mabs*np.exp(1j*np.angle(field)))))
            else:
                fm=ft.fft2((mabs*np.exp(1j*np.angle(field))))
            
            fm_abs=np.abs(fm)
            fm_abs=fm_abs-np.min(fm_abs)
            fm_abs=fm_abs/np.max(fm_abs)
            # fm=fm_abs*np.exp(1j*np.angle(fm))
            
            fm_abs=np.sum((fm_abs>0.02).astype(np.int32))
            
            if lowpass_thresh is None:
                lowpass_thresh=np.sqrt(fm_abs/np.pi)
                print(lowpass_thresh)
            
            mask=R<lowpass_thresh
            mask=mask.astype(np.float32)
            mask=ndi.gaussian_filter(mask, lowpass_thresh/2)
            
            fm=np.pad(fm,(pad_vert,pad_lat),mode='constant')*mask
            
            
            
            if shift:
                out_field=ft.fftshift(ft.ifft2(ft.ifftshift(fm)))
            else:
                out_field=ft.ifft2(fm)

        
        
        return out_field
    
    def clear_Q(self):
        
        self.wfc_Q.queue.clear()
        self.done_Q.queue.clear() 
        self.stop_Q.queue.clear()    
        self.end_gen_Q.queue.clear()        
        self.capture_Q.queue.clear()
        self.field_Q.queue.clear()
        self.capture_Q_b.queue.clear()
        self.field_Q_b.queue.clear()
        self.player_Q.queue.clear()
        self.stop_Q.queue.clear()
        self.done_Q.queue.clear()
        self.display_Q.queue.clear()
        while not self.FFT_Q[0].empty():
            self.FFT_Q[0].get_nowait()  
        # self.FFT_Q[0].clear()
        # self.FFT_Q[1].clear()
        # self.mode_Q.clear()

        self.forced_end=False
                
        
        return None
        
    def end_Q(self):
        
        self.display_Q.queue.clear()
        self.capture_Q.queue.clear()
        self.field_Q.queue.clear()
        
        self.stop_Q.put(True)
        self.display_Q.put(None)
        self.capture_Q.put(None)
        self.field_Q.put(None)

        
        return None    

class trans_matrix:
    
    def __init__(self, tm_size, meas_dir='C:\TM_meas', tm_name='TM.pkl', new_TM=True):

        if new_TM:
            folder_name=self.new_name()
            tm_path=[meas_dir,'\\',folder_name]
            tm_path=''.join(tm_path)
            
            
            os.mkdir(tm_path)
            tm_path=[tm_path,'\\',tm_name]
            tm_path=''.join(tm_path)
            
            fh = h5py.File(tm_path, 'w')
            fh.create_dataset('TM_real',shape=tm_size)
            fh.create_dataset('TM_imag',shape=tm_size)
            fh.create_dataset('mode_XY',shape=(tm_size[1],2))        
            fh.close()
        else:
            tm_path=[meas_dir,'\\',tm_name]
            tm_path=''.join(tm_path)
            fh = h5py.File(tm_path, 'r')
            tm_size=len(fh['TM_real'])
            fh.close()
        
        self.mode_index=0
        self.tm_size=tm_size
        self.tm_file=None
        self.tm_path=tm_path       
        self.input_modes=None
        self.output_modes=None        

    def file_return(self):
        fh = h5py.File(self.tm_path, 'r')
        return fh

        
    def write_new_mode(self, field, mode_coord):
        fh = h5py.File(self.tm_path, 'a')
        field.shape=np.prod(field.shape)
        fh['TM_real'][:,self.mode_index]=np.real(field)
        fh['TM_imag'][:,self.mode_index]=np.imag(field)
        fh['mode_XY'][self.mode_index,:]=mode_coord
        fh.close()
        self.mode_index+=1
    
    def read_mode(self, out_mode_ind):
        fh = h5py.File(self.tm_path, 'r')
        in_modes=fh['TM_real'][out_mode_ind,:]+1j*fh['TM_imag'][out_mode_ind,:]
        fh.close()
        return in_modes
    
    def get_in_coord(self):
        fh = h5py.File(self.tm_path, 'r')
        mode_coord=fh['mode_XY'][:,:]
        fh.close()
        return mode_coord
    
    def load_tm(self):
        fh = h5py.File(self.tm_path, 'r')
        tm_all=fh['TM_real'][:,:]+1j*fh['TM_imag'][:,:]
        fh.close()
        return tm_all
        
        
    def new_name(self):
        A=datetime.now()
        D=[A.year,A.month,A.day,int(datetime.timestamp(A))]
        date_string=''.join(str.format('{:02d}', e) for e in D)
        D=[A.hour,A.minute,A.second]
        date_string=[date_string,'_',''.join(str.format('{:02d}', e) for e in D)]
        date_string=''.join(date_string)
        
        return date_string