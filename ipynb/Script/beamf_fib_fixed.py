


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import signal



print 'DEFINING ANGLES AND SENSOR ARRAY'


sound_speed = 1482.0 # meter per second
nsensors = 20
Lsensor = 2.14 # in meters



print 'STARTING FIB FIXED CALCULATION'


Fs = 16000
Ts = 1.0/Fs
freq_range = np.arange(100, 3200, 100)
freq_ref = 1700
angles_steering = np.arange(0, 180, 1)
#angles_mainlobe = np.arange(80,110,1)
#angles_sidelobe = np.setdiff1d(angles_mainlobe, angles_steering)
angles_mainlobe = np.array([90])
angles_sidelobe = np.array([0, 180])




M = nsensors # number of sensors
J = 300 # number of taps for each FIR

a = 0.01 # Side lobe gain
b = 0.95 # frequency invariant 'strength'

W = np.zeros((M*J, 1))
j = complex(0,1) # complex variable


# Build Q matrix
Q = np.zeros((M*J, M*J))
A = np.zeros(M*J)

# Build A vector
print 'BUILDING VECTOR A'
Staps = np.array([np.exp(-2*j*np.pi*freq_ref*Ts*itap) for itap in range(J)])
for thetaMainLobe in angles_mainlobe:
    Ssteer = np.array([np.exp(-2*j*np.pi*freq_ref*m*Lsensor/sound_speed*np.cos(thetaMainLobe)) for m in range(M)])
    A = A + np.kron(Staps, Ssteer)
A = np.array([A]).T

# Build Q Matrix
print 'BUILDING MATRIX Q'
# Adjust the gain for the mainlobe
print 'CALCULATING MAIN LOBE GAIN'
for thetaMainLobe in angles_mainlobe:
    Ssteer = np.array([np.exp(-2*j*np.pi*freq_ref*m*Lsensor/sound_speed*np.cos(thetaMainLobe)) for m in range(M)])
    S = np.array([np.kron(Staps, Ssteer)]).T
    S = S.dot(np.conjugate(S).T)
    Q = Q + S
# Adjust the gain for the sidelobe
print 'CALCULATING SIDE LOBE GAIN'
for thetaSideLobe in angles_sidelobe:
    Ssteer = np.array([np.exp(-2*j*np.pi*freq_ref*m*Lsensor/sound_speed*np.cos(thetaSideLobe)) for m in range(M)])
    S = np.array([np.kron(Staps, Ssteer)]).T
    S = S.dot(np.conjugate(S).T)
    Q = Q + a*S
# Adjust the invariant response in frequency and angles
print 'ADJUSTING THE INVARIANCE RESPONSE'
for freq in freq_range:
    Staps = np.array([np.exp(-2*j*np.pi*freq*Ts*itap) for itap in range(J)])
    for theta in angles_steering:
        Ssteer_frq = np.array([np.exp(-2*j*np.pi*freq*m*Lsensor/sound_speed*np.cos(theta)) for m in range(M)])
        Ssteer_ref = np.array([np.exp(-2*j*np.pi*freq_ref*m*Lsensor/sound_speed*np.cos(theta)) for m in range(M)])
        Sfrq = np.array([np.kron(Staps, Ssteer_frq)]).T
        Sref = np.array([np.kron(Staps, Ssteer_ref)]).T
        S = Sfrq - Sref
        S = S.dot(np.conjugate(S).T)
        Q = Q + b*S


print 'ESTIMATING W'
Qinv = np.linalg.inv(Q)

W = Qinv.dot(A)

print 'BEAM PATTERN'
freq_range = np.array([1000])
beampattern = np.zeros((freq_range.shape[0], angles_steering.shape[0]))

for ifreq, freq in enumerate(freq_range):
    Staps = np.array([np.exp(-2*j*np.pi*freq*Ts*itap) for itap in range(J)])
    for itheta, theta in enumerate(angles_steering):
        Ssteer = np.array([np.exp(-2*j*np.pi*freq*m*Lsensor/sound_speed*np.cos(theta)) for m in range(M)])
        S = np.array([np.kron(Staps, Ssteer)]).T
        beampattern[ifreq, itheta] = np.conjugate(W).T.dot(S)[0,0]



raise Exception()


print 'BEAMFORMING'

beampattern = np.zeros(angles_steering.shape[0])
# Source signal
nsources = 1
source_angle = np.array([30,45, 60, 75, 80]) # in degrees
source_freqs = np.array([1000,1000, 1000, 1000, 1000]) # Hz
source_ampli = np.array([1, 1, 1, 1, 1])
# Simulate received signal
angle_res = angles_steering[1] - angles_steering[0]
delay_max = nsensors * Lsensor / sound_speed # considering the angle of 90 degrees, sin = 1
heap_size = int(delay_max * Fs)*2
heap = np.zeros((angles_steering.shape[0], heap_size))
beamf = np.zeros(angles_steering.shape[0])

#total_samples = int(delay_max * Fs)
total_samples = J

x_time = np.arange(0, total_samples*10*Ts, Ts)
received = np.zeros((nsensors, x_time.shape[0]))
for i in range(nsensors):
    for j in range(nsources):
        delay = i * Lsensor * np.sin(source_angle[j] * np.pi / 180.0) / sound_speed # in seconds
        received[i] = received[i] + source_ampli[j] * np.sin((x_time - delay) * 2 * np.pi * source_freqs[j])

# BEAMFORMING
for iang, ang in enumerate(angles_steering):
    summed_signal = np.zeros(np.max([J,x_time.shape[0]]) - np.min([J,x_time.shape[0]]) + 1)
    for i in range(nsensors):
        delay = i * Lsensor * np.sin(ang * np.pi / 180.0) / sound_speed
        delay_samples = np.abs(int(delay * Fs))
        #summed_signal = summed_signal + W[i].dot(received[i][delay_samples:delay_samples+total_samples+1])
        summed_signal = summed_signal + np.convolve(W[i], received[i],mode='valid')
    beamf[iang] = summed_signal.dot(summed_signal)/float(total_samples)


raise Exception()
plt.figure(figsize=(10,5))
plt.plot(array_angles, beamf)
plt.xticks(array_angles[::10])
plt.ylabel('Power')
plt.xlabel('DOA [degrees]')
plt.title('Delay and Sum Beamforming')



print 'FIXED FIB ESTIMATED'







