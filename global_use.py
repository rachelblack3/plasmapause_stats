import numpy as np
from datetime import datetime,date,timedelta
import glob
import os
from spacepy import pycdf
import exohiss_studies as exohiss
import multiband_studies as multi
import get_Kp as kp

# constants

global_constants = {"Electron q": 1.6*10**(-19),
                    "Electron m":9.1*10**(-31),
                    "Proton m": 1.67*10**(-27),
                    "Convert to nT": (10**(-9)),
                    "Pi": 3.14159,
                    "f_s": 1.0/35000.0,
                    "Slider": int(1024/2),
                    "Kletzing box": 16384,
                    "Sliding box": 1024,
                    "N total": 208896, 
                    "N Kletzing": 12}  


Duration = global_constants["f_s"] * global_constants["N total"]
global_constants["Duration K"] = global_constants["f_s"] * global_constants["Kletzing box"] * global_constants["N Kletzing"]
global_constants["Duration"] = Duration


# functions
def get_date_string(date):
        ''' Method that rpovides date strings
        Outputs:
    
        date_string: string object of date
        year: string object of date year
        month: string object of date month
        day: string object of date day '''

        date_string= str(date.strftime("%Y%m%d"))

        if (date.day <10):
            day = "0"+str(date.day)
        else:
            day = str(date.day)


        if (date.month<10):
            month = "0"+str(date.month)
        else:
            month = str(date.month)
        

        year = str(date.year)
        
        return date_string,year,month,day

def what_dates(default):

    ''' Function that asks command prompt the required date range
        Outputs:

        start_date: Datetime object of start date
        end_date: Datetime object of end date
        no_days: Integer number of days
    '''
    if (default == 'Zhu18'):
        start_date = exohiss.zhu18["start_date"]
        end_date = exohiss.zhu18["end_date"]
    
    elif (default == 'check'):
        start_date = date(year=2015, month=3, day=22)
        end_date = date(year=2015, month=3, day=24)

    elif (default == 'wang1'):
        start_date = exohiss.wang1["start_date"]
        end_date = exohiss.wang1["end_date"]
    
    elif (default == 'full duration'):
        start_date = date(year=2013, month=2, day=10)
        end_date = date(year=2013, month=2, day=11)

    elif (default == 'wang2'):
        start_date = exohiss.wang2["start_date"]
        end_date = exohiss.wang2["end_date"]

    elif (default == True):
        start_date = date(year=2013, month=9, day=1)
        end_date = date(year=2015, month=9, day=1)
    
    elif (default == 'fu17'):
        start_date = multi.fu17["start_date"]
        end_date = multi.fu17["end_date"]

    elif (default == 'gao17'):
        start_date = multi.gao17["start_date"]
        end_date = multi.gao17["end_date"]

    else:
        print(default)
        # Ask for start date and make datetime object
        year,month,day =  [eval(i) for i in input("Please enter start year, month, day: ").split(",")]
        start_date = date(year=year, month=month, day=day)                                                       

        # Ask for end date and make datetime object
        year_end,month_end,day_end =  [eval(i) for i in input("Please enter end year, month, day: ").split(",")]
        end_date = date(year=year_end, month=month_end, day=day_end)

    # Find integer number of days in given date range
    no_days = end_date - start_date
    no_days = int(no_days.days)                     

    return (start_date,end_date,no_days)

def h5_time_conversion(bytes_time) -> datetime:
    """
    Converts ISO 8601 datetime, which contains leap seconds, into Python
    datetime object

    :param bytes_time: time in bytes to convert
    :return: datetime in Python datetime object
    """
    date_str, time_str = bytes_time.decode("UTF-8").replace("Z", "").split("T")
    year, month, day = date_str.split("-")
    date = datetime(int(year), int(month), int(day))
    no_ms_time, dot, ms = time_str.partition(".")
    hours, minutes, seconds = no_ms_time.split(":")
    date = date + timedelta(
        hours=int(hours), minutes=int(minutes), seconds=int(seconds)
    )
    if dot:
        date = date + timedelta(milliseconds=int(ms))
    return date

def get_epoch(epoch):
    # get epoch in correct format for comparisons etc.
        epoch_new = []

        # First - saving the epoch elements to a list as a string - acording to a particular desired format
        for i in range(len(epoch)):
            epoch_new.append(datetime.strftime(epoch[i],'%Y-%m-%d %H-%M-%S'))

        # Chaning back to a datetime object of same format    
        for i in range(len(epoch_new)):
            epoch_new[i] = datetime.strptime(epoch_new[i],'%Y-%m-%d %H-%M-%S')

        return epoch_new

def find_closest(epoch_known, wanted_time):

        n = len(epoch_known)

        nup = n-1
        nlow=0
        mid=np.int((nup+nlow)/2)

        while (nup - nlow)>1:
            mid= np.int((nup+nlow)/2)
            if (wanted_time > epoch_known[mid]):
                nup = nup
                nlow=mid
            else:
                nup = mid
                nlow = nlow
        
        wanted_index = nlow
        corresponding_time = epoch_known[wanted_index]

        return(corresponding_time,wanted_index)


class omni_dataset:
    ''' Class for creating omni_dataset over full duration specified '''

    def __init__(self,start,end):
        self.start_date = start
        self.end_date = end

        self.omni_cache = None

    @property
    def omni_stats(self):
            if self.omni_cache is None:
                print("filling cache")
                omni_file = '/data/spacecast/wip/jinng_data/solar_wind/omni_high_res_combined_2000_to_2020.txt'
                file_format = '/data/spacecast/wip/jinng_data/solar_wind/high_res_format'
                f_format = open(f'{file_format}.txt',"r")
                line_formats=f_format.readlines()

                for line in line_formats:
                    print(line)
                f_format.close()
                # From this, AE is line 11, so index line position 10
                # Initialise empty lists to store AE and omni_epoch

                AE=[]
                epoch_omni=[]
                Kp=[]
                print('the start and end are',self.start_date,self.end_date)
                start_d = str(self.start_date.strftime("%Y-%m-%d"))
                
                no_days = self.end_date - self.start_date
                no_days = int(no_days.days)


                f=open(omni_file,"r")
                lines=f.readlines()
                
                print("the self.start_date",self.start_date)
                i=0
                for line in lines:
                    
                    line = [x for x in line.rstrip("\n").split(" ") if x != ""]
                    Date = date(
                        int(line[0].replace(" ", "")), 1, 1
                    ) + timedelta(
                        days=int(line[1]) - 1
                    )
                    if Date == self.start_date:
                        print(start_d, 'string exists in file')
                        print('Line Number:', i)
                        first=i
                        start_d = self.start_date
                        break
                    i=i+1
                        
                for line in lines[first:(first+(no_days*24*60))]:
                
                    line = [x for x in line.rstrip("\n").split(" ") if x != ""]
                    Date = datetime(int(line[0].replace(" ", "")), 1, 1
                    ) + timedelta(
                        days=int(line[1]) - 1,
                        hours=int(line[2]),
                        minutes=int(line[3]),
                    )
                    epoch_omni.append(Date)
                    AE.append(float(line[10]))
                print('OMNI dataset created')
                f.close()

                self.omni_cache = AE, epoch_omni

                return AE,epoch_omni
    
            else:
                print("data from cache")
                return self.omni_cache
            
    @property
    def omni_stats_low_res(self):
            """ This is for Kp and Dst, which are given on an hourly rate """

            omni_file = '/data/spacecast/wip/jinng_data/solar_wind/omni_low_res_combined_2000_to_2020.txt'
            file_format = '/data/spacecast/wip/jinng_data/solar_wind/low_res_format'
            f_format = open(f'{file_format}.txt',"r")
            line_formats=f_format.readlines()

            for line in line_formats:
                print(line)
            f_format.close()
            # From this, AE is line 11, so index line position 10
            # Initialise empty lists to store AE and omni_epoch

            Dst=[]
            epoch_omni=[]
            Kp=[]

            print('the start and end are',self.start_date,self.end_date)
            start_d = str(self.start_date.strftime("%Y-%m-%d"))
            
            no_days = self.end_date - self.start_date
            no_days = int(no_days.days)


            f=open(omni_file,"r")
            lines=f.readlines()

            i=0
            for line in lines:
                
                line = [x for x in line.rstrip("\n").split(" ") if x != ""]
                Date = date(
                    int(line[0].replace(" ", "")), 1, 1
                ) + timedelta(
                    days=int(line[1]) - 1
                )
                print(self.start_date)
                if Date == self.start_date:
                    print(start_d, 'string exists in file')
                    print('Line Number:', i)
                    first=i
                    start_d = self.start_date
                    break
                i=i+1
                    
            for line in lines[first:(first+(no_days*24*60))]:
            
                line = [x for x in line.rstrip("\n").split(" ") if x != ""]
                Date = datetime(int(line[0].replace(" ", "")), 1, 1
                ) + timedelta(
                    days=int(line[1]) - 1,
                    hours=int(line[2]),
                )
                epoch_omni.append(Date)
                Kp.append(float(line[3]))
                Dst.append(float(line[4]))

            print('OMNI dataset created')
            f.close()

            self.omni_cache = Kp, Dst, epoch_omni

            return Kp,Dst,epoch_omni

       
    @property       
    def Kp(self):

        kp_data = kp.getKpindex(self.start_date,self.end_date,'Kp')
        Kp = kp_data[1]
        Kp_epoch = kp_data[0]
        print("Kp dataset created")

        return Kp, Kp_epoch

class DataFiles:
    ''' class for accessing all of the data files that I need on a given day.  
    
            PARAMETERS:
            date: DateTime object 
           '''

    def __init__(self,date):
        self.date = date
        

    # finding survey data filepath for given day
    @property
    def survey_data(self):
    
        # String version of dates for filename  
        date_string,year,month,day =self.get_date_string()

        # root folder
        survey_folder ='/data/spacecast/satellite/RBSP/emfisis/data/RBSP-A/L2'
        # 'stem' name for burst CDFs
        survey_file= 'rbsp-a_WFR-spectral-matrix-diagonal_emfisis-L2_'

        survey_path = os.path.join(survey_folder, year, month, day,survey_file + date_string + "_v*.cdf")
        
        # find the latest version
        survey_path = glob.glob(survey_path)[-1]
        
        return survey_path
    
    # getting the high frequnecy survey data (i.e. for plasma frequencies)
    @property
    def hfr_data(self):

        # String version of dates for filename  
        date_string,year,month,day =self.get_date_string()

        # root folder
        hfr_folder = '/data/spacecast/satellite/RBSP/emfisis/data/RBSP-A/L2'

        # 'stem' name for burst cdfs
        hfr_file ='rbsp-a_HFR-spectra_emfisis-L2_'

        hfr_path = os.path.join(hfr_folder, year, month, day, hfr_file + date_string + "_v*.cdf")

        # find the latest version
        hfr_path = glob.glob(hfr_path)[-1]

        return hfr_path
    
    @property
    def survey_WNA(self):

        # String version of dates for filename  
        date_string,year,month,day =self.get_date_string()

        # root folder
        wna_folder ='/data/spacecast/satellite/RBSP/emfisis/data/RBSP-A/wna-survey'
        # 'stem' name for burst CDFs
        wna_file= 'rbsp-a_wna-survey_emfisis-l4_'
        wna_path = os.path.join(wna_folder, year,wna_file + date_string + "_v*.cdf")
        print(wna_path)
        # find the latest version
        wna_path = glob.glob(wna_path)[-1]

        return wna_path


    # finding the magnetic data filepath for given day
    @property
    def magnetic_data(self):

        # String version of dates for filename  
        date_string,year,month,day =self.get_date_string()

        # root folder
        l3_folder ='/data/spacecast/satellite/RBSP/emfisis/data/RBSP-A/L3'
        # 'stem' name for burst CDFs
        l3_file= 'rbsp-a_magnetometer_1sec-geo_emfisis-L3_'

        l3_path = os.path.join(l3_folder, year, month, day,l3_file + date_string + "_v*.cdf")

        # find the latest version
        l3_path = glob.glob(l3_path)[-1]
        
        return l3_path

    # finding the LANL data filepath for given day 
    @property
    def lanl_data(self):
        # String version of dates for filename  
        date_string,year,month,day =self.get_date_string()

        # root folder
        lanl_folder ='/data/spacecast/satellite/RBSP/emfisis/data/RBSP-A/LANL/MagEphem'
        # 'stem' name for burst CDFs
        lanl_file= 'rbspa_def_MagEphem_TS04D_'

        lanl_path = os.path.join(lanl_folder, year, lanl_file + date_string + "_v*.h5")

        # find the latest version
        lanl_path = glob.glob(lanl_path)[-1]
        
        return lanl_path
    


    # Get L4 data filepath (aka density file) for given day
    
    def l4_data(self):

        # String version of dates for filename  
        date_string,year,month,day =self.get_date_string()

        # root folder
        l4_folder ='/data/spacecast/satellite/RBSP/emfisis/data/RBSP-A/L4'
        # 'stem' name for burst CDFs
        l4_file= 'rbsp-a_density_emfisis-L4_'

        l4_path = os.path.join(l4_folder, year, month, day,l4_file + date_string + "_v*.cdf")

        # find the latest version
        l4_path = glob.glob(l4_path)
 
        # check if there is actually any data files for this day:

        if not l4_path:
            exists = False
        else:
            exists = True
            l4_path = l4_path[-1]
        
        return l4_path, exists
    
    # Get burst filepaths for all CDFs on given day
    @property
    def burst_paths(self):

        # root folder
        wfr_folder ='/data/spacecast/wave_database_v2/RBSP-A/L2'

        # 'stem' name for burst CDFs
        burst6 = 'rbsp-a_WFR-waveform-continuous-burst_emfisis-L2_'

        # String version of dates for filename  
        date_string,year,month,day =self.get_date_string()

        
        # Full filepath
        wfr_burst_path = os.path.join(wfr_folder, year, month, day, burst6 + date_string + "*_v*.cdf")

        # files are all CDFs for a given day
        cdf_files= glob.glob(wfr_burst_path)

        return cdf_files
    
    
    def get_date_string(self):
        ''' Method that rpovides date strings
        Outputs:
    
        date_string: string object of date
        year: string object of date year
        month: string object of date month
        day: string object of date day '''

        date_string= str(self.date.strftime("%Y%m%d"))

        if (self.date.day <10):
            day = "0"+str(self.date.day)
        else:
            day = str(self.date.day)


        if (self.date.month<10):
            month = "0"+str(self.date.month)
        else:
            month = str(self.date.month)
        

        year = str(self.date.year)
        
        return date_string,year,month,day



    
class AccessLANLAttrs:

    ''' Class for finding and working on Survey data attributes
     
      PARAMETERS:
      survey_cdf: A CDF containing all survey data '''

    def __init__(self, lanl_data):
        self.data = lanl_data

    @property
    def L_star(self):

        # get L*
        lstar = np.array(self.data['Lstar'][:, 0])

        
        return lstar
    
    @property
    def MLT(self):

        # get MLT
        MLT = np.array(self.data['EDMAG_MLT'])

        return MLT
    
    @property
    def MLAT_N_S(self):

        # get all MLAT
        MLAT = np.array(self.data['EDMAG_MLAT'])

        # want to plot north and south on same axis - so split up
        south_mask = MLAT<5.
        north_mask = MLAT>0.
        MLAT_north = np.where(north_mask,MLAT, np.nan)
        MLAT_south = np.where(south_mask,MLAT, np.nan)

        return MLAT_north,MLAT_south
    
    @property
    def epoch(self):

        lanl_times=np.array([h5_time_conversion(x) for x in self.data['IsoTime']])

        return lanl_times
    
    @property
    def day_limits(self):

        L_star = self.L_star
        
        left=np.array([h5_time_conversion(x) for x in self.data['IsoTime']])[L_star >= 0][0]
        right=np.array([h5_time_conversion(x) for x in self.data['IsoTime']])[L_star >= 0][-1]

        return left,right
    
    @property
    def apogee_perigee(self):

        apogee_times = np.array([h5_time_conversion(x) for x in self.data["ApogeeTimes"]])
        perigee_times = np.array([h5_time_conversion(x) for x in self.data["PerigeeTimes"]])

        return apogee_times,perigee_times



class AccessHFRAttrs:
    ''' Class for finding and working on HFR survey data attributes
     
      PARAMETERS:
      survey_cdf: A CDF containing all hfr data '''

    def __init__(self, hfr_cdf):
        self.hfr_cdf = hfr_cdf

    @property
    def frequency(self):

        freq_hfr = self.hfr_cdf['HFR_frequencies'][0]

        return freq_hfr
    
    @property
    def epoch(self):
        # Epoch in DateTime format of: 
        epoch = self.hfr_cdf['Epoch']

        return epoch
    
    @property
    def Emagnitude(self):

        Etotal = self.hfr_cdf['HFR_Spectra']

        return Etotal
    
    def epoch_convert(self):
    # get epoch in correct format for comparisons etc.
        epoch_edit = []
        epoch = self.hfr_cdf['Epoch']

        # First - saving the epoch elements to a list as a string - acording to a particular desired format
        for i in range(len(epoch)):
            epoch_edit.append(datetime.strftime(epoch[i],'%Y-%m-%d %H-%M-%S'))

        # Chaning back to a datetime object of same format    
        for i in range(len(epoch_edit)):
            epoch_edit[i] = datetime.strptime(epoch_edit[i],'%Y-%m-%d %H-%M-%S')

        return epoch_edit
    


class AccessSurveyAttrs:
    ''' Class for finding and working on Survey data attributes
     
      PARAMETERS:
      survey_cdf: A CDF containing all survey data '''
    

    def __init__(self, survey_cdf):
        self.survey_cdf = survey_cdf

    @property
    def frequency(self):
        # Frequency in Hz
        frequency = self.survey_cdf['WFR_frequencies'][0]

        return frequency
    
    @property
    def epoch(self):
        # Epoch in DateTime format of: 
        epoch = self.survey_cdf['Epoch']

        return epoch
    
    @property
    def Bmagnitude(self):

        Bu2 = self.survey_cdf['BuBu']
        Bv2 = self.survey_cdf['BvBv']
        Bw2 = self.survey_cdf['BwBw']

        # Define empty list for total mag field array 

        Btotal = np.zeros(self.survey_cdf['BuBu'].shape)

        # Create total mag B array

        for p in range(0,np.shape(Btotal)[0]):
            Btotal[p,:] =Bu2[p,:]+ Bv2[p,:] + Bw2[p,:]

        return Btotal
    
    @property
    def Emagnitude(self):

        Eu2 = self.survey_cdf['EuEu']
        Ev2 = self.survey_cdf['EvEv']
        Ew2 = self.survey_cdf['EwEw']

        # Define empty list for total mag field array 

        Etotal = np.zeros(Eu2.shape)

        # Create total mag B array

        for p in range(0,np.shape(Etotal)[0]):
            Etotal[p,:] =Eu2[p,:]+ Ev2[p,:] + Ew2[p,:]

        return Etotal
    
    def epoch_convert(self):
    # get epoch in correct format for comparisons etc.
        epoch_edit = []
        epoch = self.survey_cdf['Epoch']

        # First - saving the epoch elements to a list as a string - acording to a particular desired format
        for i in range(len(epoch)):
            epoch_edit.append(datetime.strftime(epoch[i],'%Y-%m-%d %H-%M-%S'))

        # Chaning back to a datetime object of same format    
        for i in range(len(epoch_edit)):
            epoch_edit[i] = datetime.strptime(epoch_edit[i],'%Y-%m-%d %H-%M-%S')

        return epoch_edit
    
    @property
    def frequency_bin_widths(self):

        widths = self.survey_cdf['WFR_bandwidth'][0]
        return widths
    
    @property
    def bin_edges(self):
        """ 
        setting the bin edges 
        min_bin: lower edge of first bin 
        """
        survey_freq = self.frequency

        min_bin = survey_freq[0]-(self.frequency_bin_widths[0]/2)

        freq_bins = []
        freq_bins.append(min_bin)

        # Starting from the minimum bin, add all widths to get lower and upper bands on all semi_logarithmic bins

        for i in range(0,65):
            
            freq_bins.append(freq_bins[i]+self.frequency_bin_widths[i])
            min_bin=freq_bins[i]
        
        return freq_bins
    
    def clean_data(self, flag):

        ''' function for cleaning the e and b magntidue for the survey (thruster firing)'''
        Emag = self.Emagnitude
        Bmag = self.Bmagnitude
        epoch = self.epoch_convert()

        eclipse_t = self.get_badtimes("/data/emfisis_burst/wip/rablack75/BackReduction/rbsp-a_eclipse_times.txt")
        thruster_t = self.get_badtimes("/data/emfisis_burst/wip/rablack75/BackReduction/rbsp-a_thruster_firing_times.txt")
        charging_t = self.get_badtimes("/data/emfisis_burst/wip/rablack75/BackReduction/rbsp-a_charging_times.txt")

        for i in range(0,np.shape(Emag)[0]):
            # Accounting for bad times - set etotal to -1 on these times 
            # so easy to ignore
            for k in range(len(thruster_t)):
                if (thruster_t[0][k]<epoch[i]<thruster_t[1][k]):
                    Emag[i,:]=np.nan
                    Bmag[i,:]=np.nan

            for k in range(len(charging_t)):
                if (charging_t[0][k]<epoch[i]<charging_t[1][k]):
                    Emag[i,:]=np.nan 
                    Bmag[i,:]=np.nan

            for k in range(len(eclipse_t)):
                if (eclipse_t[0][k]<epoch[i]<eclipse_t[1][k]):
                    Emag[i,:]=np.nan
                    Bmag[i,:]=np.nan  

        if flag == 'B':
            mag = Bmag
        else:
            mag = Emag

        return mag

    def get_badtimes(self,filepath):
        ''' Make a list of all the 'bad' times from specified file '''
        starts=[]
        ends=[]
        file = np.loadtxt(filepath,dtype='str',skiprows=1,usecols=[0, 1],delimiter=',')
        start_times=file[:,0]
        end_times=file[:,1]
        durations =[]

    
        for i in range(len(start_times)):
            
            start_times[i]=start_times[i].replace("T"," ")
            if (start_times[i][-8:-1]=='0000000'):
                start_times[i]=start_times[i][:-5]
            starts.append(datetime.strptime(start_times[i], '%Y-%m-%d %H:%M:%S.%f'))

        for i in range(len(end_times)):
            end_times[i]=end_times[i].replace("T"," ")
            if (end_times[i][-8:-1]=='0000000'):
                end_times[i]=end_times[i][:-5]
            ends.append(datetime.strptime(end_times[i], ' %Y-%m-%d %H:%M:%S.%f'))
            
        durations.append(starts)
        durations.append(ends)
        
        return durations

class AccessWNAAttrs:
    ''' Class for finding and working on HFR survey data attributes
     
      PARAMETERS:
      survey_cdf: A CDF containing all hfr data '''

    def __init__(self, wna_cdf):
        self.wna_cdf = wna_cdf

    @property
    def frequency(self):

        freq_hfr = self.hfr_cdf['HFR_frequencies'][0]

        return freq_hfr
    
    @property
    def epoch(self):
        # Epoch in DateTime format of: 
        epoch = self.hfr_cdf['Epoch']

        return epoch
    
    @property
    def Emagnitude(self):

        Etotal = self.hfr_cdf['HFR_Spectra']

        return Etotal
    
    def epoch_convert(self):
    # get epoch in correct format for comparisons etc.
        epoch_edit = []
        epoch = self.hfr_cdf['Epoch']

        # First - saving the epoch elements to a list as a string - acording to a particular desired format
        for i in range(len(epoch)):
            epoch_edit.append(datetime.strftime(epoch[i],'%Y-%m-%d %H-%M-%S'))

        # Chaning back to a datetime object of same format    
        for i in range(len(epoch_edit)):
            epoch_edit[i] = datetime.strptime(epoch_edit[i],'%Y-%m-%d %H-%M-%S')

        return epoch_edit

class AccessL3Attrs:
    ''' Class for finding and working on L3 data attributes
     
      PARAMETERS:
      mag_cdf: A CDF containing all L3 data '''

    def __init__(self, mag_file):
        self.mag_cdf = mag_file

    @property
    def Bmagnitude(self):
        # Frequency in Hz
        Bmagnitude = self.mag_cdf['Magnitude']

        return Bmagnitude
    
    @property
    def epoch(self):
        # Epoch in DateTime format of: 
        epoch = self.mag_cdf['Epoch']

        return epoch
    
    def epoch_convert(self):
    # get epoch in correct format for comparisons etc.
        epoch_edit = []
        epoch = self.mag_cdf['Epoch']

        # First - saving the epoch elements to a list as a string - acording to a particular desired format
        for i in range(len(epoch)):
            epoch_edit.append(datetime.strftime(epoch[i],'%Y-%m-%d %H-%M-%S'))

        # Chaning back to a datetime object of same format    
        for i in range(len(epoch_edit)):
            epoch_edit[i] = datetime.strptime(epoch_edit[i],'%Y-%m-%d %H-%M-%S')

        return(epoch_edit)
    
    @property
    def f_ce(self):
        # Finding the gyrofrequencies for plotting
       
        gyro_1 = np.zeros(self.Bmagnitude.shape)
        gyro_05 = np.zeros(self.Bmagnitude.shape)
        gyro_005 = np.zeros(self.Bmagnitude.shape)
        gyro_p = np.zeros(self.Bmagnitude.shape)
        f_lhr = np.zeros(self.Bmagnitude.shape)
        # Clean magnetometer data
        mag_cleaned = self.clean_magnetometer(self.Bmagnitude)


        for i in range(0,len(gyro_1)):
            gyro_p[i] = global_constants["Electron q"]*mag_cleaned[i]*global_constants["Convert to nT"]/(2*global_constants["Pi"]*global_constants["Proton m"])
            gyro_1[i] = global_constants["Electron q"]*mag_cleaned[i]*global_constants["Convert to nT"]/(2*global_constants["Pi"]*global_constants["Electron m"])
            gyro_05[i] = 0.5*gyro_1[i]
            gyro_005[i] = 0.05*gyro_1[i]
            f_lhr[i] = np.sqrt(gyro_1[i]*gyro_p[i])

        return gyro_1, gyro_05, gyro_005, f_lhr

    def clean_magnetometer(self,unclean_data):

        magfill = self.mag_cdf['magFill']
        magInval = self.mag_cdf['magInvalid']
        magCal = self.mag_cdf['calState']

        # Find the indicies where we have invalid data, save to 'whereFill' 
        whereFill = []
        for i in range(len(magfill)):
            if (magfill[i] == 1 or magInval[i]==1 or magCal[i]==1):
                whereFill.append(i)
                
        # Make unclean data into list
        magnitude = np.array(unclean_data).tolist()

        # Use list of indicies to set all invlaid data points to NaNs
        for ele in sorted(whereFill, reverse = True):
            
            magnitude[ele] = np.nan
        
        return magnitude


class AccessL4Attrs:
    ''' Class for finding and working on L4 data attributes.
     
      PARAMETERS:
      density_cdf: A CDF containing all L4 data '''

    def __init__(self, density_file):
        self.density_cdf = density_file

    @property
    def density(self):
        # density in cm^(-3)
        density = self.density_cdf['density']

        return density
    
    @property 
    def f_pe(self):
        # plasma frequency in Hz
        f_pe = self.density_cdf['fpe']

        return f_pe
        
    @property
    def epoch(self):
        # Epoch in DateTime format of: 
        epoch = self.density_cdf['Epoch']

        return epoch
    
    def epoch_convert(self):
    # get epoch in correct format for comparisons etc.
        epoch_edit = []
        epoch = self.density_cdf['Epoch']

        # First - saving the epoch elements to a list as a string - acording to a particular desired format
        for i in range(len(epoch)):
            epoch_edit.append(datetime.strftime(epoch[i],'%Y-%m-%d %H-%M-%S'))

        # Chaning back to a datetime object of same format    
        for i in range(len(epoch)):
            epoch_edit[i] = datetime.strptime(epoch_edit[i],'%Y-%m-%d %H-%M-%S')

        return(epoch_edit)

class cross_dataset:
    ''' A class for performing operations across datasets '''
    def __init__(self,survey_data, l4_data, burst_time):

        self.survey_epoch = survey_data['Epoch']
        self.mag_epoch = l4_data['Epoch']
        self.Bmag = l4_data['Magnitude']
        self.burst_time = burst_time
    
    def calc_gyro(self):
        # get epoch in correct format for comparisons etc.

        epoch_mag = get_epoch(self.mag_epoch)

        mag_field=self.Bmag

        # Finding the closest index in this magnetometer list to the burst time object

        mag_t,mag_index = find_closest(epoch_mag,self.burst_time)

        # Finding the gyrofrequencies for plotting

        gyro_one= global_constants["Electron q"]*mag_field[mag_index]/(2*global_constants["Pi"]*global_constants["Electron m"]) # in T
        gyro_one = gyro_one*global_constants["Convert to nT"]                                                                   # in nT
        gyro_half=0.5*gyro_one
        gyro_low=0.05*gyro_one


        return gyro_one,gyro_half,gyro_low
    
    
    def get_epoch(epoch):
    # get epoch in correct format for comparisons etc.
        epoch_new = []

        # First - saving the epoch elements to a list as a string - acording to a particular desired format
        for i in range(len(epoch)):
            epoch_new.append(datetime.strftime(epoch[i],'%Y-%m-%d %H-%M-%S'))

        # Chaning back to a datetime object of same format    
        for i in range(len(epoch_new)):
            epoch_new[i] = datetime.strptime(epoch_new[i],'%Y-%m-%d %H-%M-%S')

        return epoch_new