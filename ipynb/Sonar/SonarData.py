from scipy.io import wavfile
from scipy import signal
import os
import numpy as np
import wave
import Functions as fn
import matplotlib.pyplot as plt


class SonarData(object):
    ######################################################################################
    def __init__(self, dataship = '', datarun=-1):
        self.__sonardir = os.getenv('SONARPATH')
        self.__data = {}
        self.__lofar = {}
        self.__LoadClassification(dataship, datarun)
    ######################################################################################
    def __LoadClassification(self, dataship = '', datarun = -1):
        dname = self.__sonardir + '/Data/Classification/'
        for ship in os.listdir(dname):
            # Check ship
            if ship == dataship or dataship == '':
                for fname in os.listdir(dname + '/' + ship + '/'):
                    run = int(fname.split('_')[-1].replace('.wav',''))
                    if run == datarun or datarun == -1:
                        try:
                            data = wavfile.read(dname+'/' + ship + '/' + fname)
                        except Exception as err:
                            print 'WARNING in reading file ', dname+'/' + ship + '/' + fname
                            print err.__str__()
                            print 'Trying non scipy mode'
                            try:
                                data = self.__readwav(dname+'/' + ship + '/' + fname)
                            except Exception as err2:
                                print 'ERROR: could not read file ', fname
                                print err2.__str__()
                                continue
                            print 'Loading worked!'
                            pass
                        if not len(data[1]):
                            print 'WARNING: file ' + fname + ' has 0 length data'
                        if not self.__data.has_key(ship):
                            self.__data[ship] = {}
                        Fs = float(data[0])
                        X = data[1]
                        X = (((X - np.min(X)) * (2.0))/(np.max(X)-np.min(X))) + (-1)
                        self.__data[ship][run] = (Fs, X)
    ######################################################################################
    def GetShipRuns(self, ship):
        if self.__data.has_key(ship):
            return self.__data[ship].keys()
        return []
    ######################################################################################
    def GetRawData(self, ship=-1, run=-1):
        if ship == -1 and run == -1:
            return self.__data
        if ship != -1 and run == -1:
            if self.__data.has_key(ship):
                return self.__data[ship]
        if ship != -1 and run != -1:
            if self.__data.has_key(ship):
                if self.__data[ship].has_key(run):
                    return self.__data[ship][run]
        return {}
    ######################################################################################
    def SetRawData(self, data, ship=-1, run=-1):
        if ship == -1 and run == -1:
            self.__data = data
        if ship != -1 and run == -1:
            self.__data[ship] = data
        if ship != -1 and run != -1:
            if not self.__data.has_key(ship):
                self.__data[ship] = {}
            self.__data[ship][run] = data
    ######################################################################################
    def GetLOFARData(self, ship=-1, run=-1):
        if ship == -1 and run == -1:
            return self.__lofar
        if ship != -1 and run == -1:
            if self.__lofar.has_key(ship):
                return self.__lofar[ship]
        if ship != -1 and run != -1:
            if self.__lofar.has_key(ship):
                if self.__lofar[ship].has_key(run):
                    return self.__lofar[ship][run]
        return {}
    ######################################################################################
    def SetLOFARData(self, data, ship=-1, run=-1):
        if ship == -1 and run == -1:
            self.__lofar = data
        if ship != -1 and run == -1:
            self.__lofar[ship] = data
        if ship != -1 and run != -1:
            if not self.__lofar.has_key(ship):
                self.__lofar[ship] = {}
            self.__lofar[ship][run] = data
    ######################################################################################
    def LOFAR(self, useship = -1, userun = -1, param = {}):
        for ship in self.__data.keys():
            if ship != useship and useship != -1:
                continue
            print '====> Doing LOFAR on ', ship, ' runs ',
            for run in self.__data[ship].keys():
                if run != userun and userun != -1:
                    continue
                print run, ', ',
                ## Do LOFAR
                Fs = self.__data[ship][run][0]
                data = self.__data[ship][run][1]
                # Decimation
                rate = 0
                order = 1
                if param.has_key('Decimation'):
                    rate = param['Decimation']['Rate']
                    order = param['Decimation']['FIROrder']
                    data = signal.decimate(data, rate, order, 'fir')
                    Fs = Fs/rate
                NFFT = 512 if not param.has_key('NFFT') else param['NFFT']
                overlap = 0 if not param.has_key('Overlap') else param['Overlap']
                [intensity, freqs, t, h] = plt.specgram(data, NFFT=NFFT, Fs=Fs, noverlap=overlap)
                plt.close()
                intensity=intensity / fn.tpsw(intensity)
                intensity=np.log10(intensity);
                intensity[np.nonzero(intensity < -0.2)] = 0;
                if not self.__lofar.has_key(ship):
                    self.__lofar[ship] = {}
                if not self.__lofar[ship].has_key(run):
                    self.__lofar[ship][run] = {}
                self.__lofar[ship][run]['Signal'] = intensity
                self.__lofar[ship][run]['NFFT'] = NFFT
                self.__lofar[ship][run]['Overlap'] = overlap
                self.__lofar[ship][run]['DecRate'] = rate
                self.__lofar[ship][run]['DecFIROrder'] = order
                self.__lofar[ship][run]['Fs'] = Fs
                self.__lofar[ship][run]['Freqs'] = freqs
                self.__lofar[ship][run]['Timew'] = t
            print ' done!'
    ######################################################################################
    def PlotLOFAR(self, ship, run, param = {}):
        if not self.__lofar.has_key(ship):
            print 'ERROR: LOFAR analysis not done yet'
            return
        if not self.__lofar[ship].has_key(run):
            print 'ERROR: LOFAR analysis not done yet'
            return
        intensity = self.__lofar[ship][run]['Signal']
        freqs = self.__lofar[ship][run]['Freqs']
        t = self.__lofar[ship][run]['Timew']
        plt.imshow(intensity.transpose(), extent=[0,np.max(freqs),np.max(t), 0], aspect='auto')
        plt.xlabel('Frequency [kHz]')
        plt.ylabel('Time [s]')
    ######################################################################################
    def __wav2array(self, nchannels, sampwidth, data):
        """data must be the string containing the bytes from the wav file."""
        num_samples, remainder = divmod(len(data), sampwidth * nchannels)
        if remainder > 0:
            raise ValueError('The length of data is not a multiple of '
                             'sampwidth * num_channels.')
        if sampwidth > 4:
            raise ValueError("sampwidth must not be greater than 4.")

        if sampwidth == 3:
            a = np.empty((num_samples, nchannels, 4), dtype=np.uint8)
            raw_bytes = np.fromstring(data, dtype=np.uint8)
            a[:, :, :sampwidth] = raw_bytes.reshape(-1, nchannels, sampwidth)
            a[:, :, sampwidth:] = (a[:, :, sampwidth - 1:sampwidth] >> 7) * 255
            result = a.view('<i4').reshape(a.shape[:-1])
        else:
            # 8 bit samples are stored as unsigned ints; others as signed ints.
            dt_char = 'u' if sampwidth == 1 else 'i'
            a = np.fromstring(data, dtype='<%s%d' % (dt_char, sampwidth))
            result = a.reshape(-1, nchannels).transpose()[0]
        return result
    ######################################################################################
    def __readwav(self, file):
        """
        Read a wav file.

        Returns the frame rate, sample width (in bytes) and a numpy array
        containing the data.

        This function does not read compressed wav files.
        """
        wav = wave.open(file)
        rate = wav.getframerate()
        nchannels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        nframes = wav.getnframes()
        data = wav.readframes(nframes)
        wav.close()
        array = self.__wav2array(nchannels, sampwidth, data)
        return (rate, array)


# end of file



