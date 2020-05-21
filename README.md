# Adaptive Modulation Schemes for Inter-Satellite Links
Bachelor project made by group 20gr651 at Electronics &amp; IT, Aalborg University.

Explanation of files

<b>OneWeb_constellation.gif:</b> A graphical simulation of the OneWeb constellation.

<b>Base-band simulations:</b> Simulation scripts able to simulate the bit-error rate and symbol-error rates of BPSK, QPSK, 16-QAM and 64-QAM. All these are run in base-band, omitting the waveform modulation part. There are two editions of every base-band script. The core version plots a constellation diagram with noise, and prints out the SER and BER. The SNR version exports a text file with the SER and BER data, ready to be plotted or processed externally.

<b>Pass-band simulations:</b> Contains a BPSK pass-band simulator, that modulates the I and Q values on a cosine and sine converting them to pass-band, and downconverts to base-band recovering the transmitted signals.

<b>Adaptive Modulation:</b> Contains the adaptive modulation simulator, able of switching between BPSK, QPSK, 16-QAM and 64-QAM depending on the SNR. Includes a model of two satellites in a cross-plane constellation with completely user definable parameters.
