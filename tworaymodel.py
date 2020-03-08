import numpy as np
import matplotlib.pyplot as plt
import os

# 4 set of bands to simulate (4 subplots)
b1 = ['900']
b2 = ['800', '900', '2100']
b3 = ['800', '900', '1800', '2100']
b4 = ['800', '900', '1800', '2100', '2600']

# u value used to simplified two-ray model, assuming that two rays are always combined coherently, with a reflection
# factor |Γ|, so the sum is multiplied by u=1+Γ. (FSPL is when u=1 (only 1 ray)).
u = 1.6

R = -0.9    # R (or Γ) reflection factor used in two-ray ground-reflection model (in general is
          # dependent on the angle of incidence)
ht = 6      # transmitter height
hr = 4      # receiver height
maxd = 25   # Maximum distance from transmitter (m)


d = np.linspace(1, maxd, 2000)
d_ref = np.sqrt((ht + hr) ** 2 + d ** 2)
d_los = np.sqrt((ht - hr) ** 2 + d ** 2)
G_los = 1
G_gr = 1
bandslist = [b1, b2, b3, b4]

fig, ax = plt.subplots(2, 2, figsize=(20,15))
fig.suptitle(f'Models comparison for: ht={ht}m, hr={hr}m, Γ={R} and u={u} simplified model')
ax = ax.ravel()

for i, bands in enumerate(bandslist):
    tworayloss = 0
    freespaceloss = 0
    freq = np.asarray(bands).astype(int)
    lam = 3 * 10 ** 2 / freq

    for lam in lam:
        phi = 2 * np.pi * (d_ref - d_los) / lam
        loscoef = np.sqrt(G_los) / d_los
        reflcoef = R * np.sqrt(G_gr) * np.exp(-1j * phi) / d_ref
        rs = lam * (loscoef + reflcoef) / (4 * np.pi)

        tworayloss += 10*np.log10((abs(rs))**2)
        freespaceloss += 20*np.log10(lam / (4 * np.pi * d_los))

    freespace_u = 10*len(freq)*np.log10(u**2) + freespaceloss
    norm = max(tworayloss[0], freespace_u[0])

    tworayloss = tworayloss - norm
    freespaceloss = freespaceloss - norm
    freespace_u = freespace_u-norm

    ax[i].semilogx(d, tworayloss, d, freespace_u, d, freespaceloss)
    p = (1 - np.sum(tworayloss>freespace_u)/len(d))*100
    ax[i].text(1, 0.90*min(tworayloss), f'{p:0.2f}% of values (u={u} model)\ngreater than analytical model')
    ax[i].text(1, min(tworayloss), f'Mean diff. [(u={u}), analytical two-ray]:='
                                        f'{np.mean(freespace_u-tworayloss):+.1f}dB\n'
                                        f'Mean diff. [FSPL, analytical two-ray]:='
                                        f'{np.mean(freespaceloss-tworayloss):+.1f}dB)')



    ax[i].legend((f'Two-Ray ground-reflection\n analytical model (Γ={R})', f'u={u}', 'Free Space (FSPL)'), loc=6)
    bandsstring = ', '.join(freq.astype(str))
    bandsstring = '(' + bandsstring + ')'
    fname = f'ht = {ht}m, hr={hr}m, Γ={R}, u={u}'
    title = f'Bands={bandsstring}MHz'
    ax[i].set_title(title)
    ax[i].set_xlabel('Distance (m)')
    ax[i].set_ylabel('Normalized Path Loss (dB)')

    xticks = np.append(1, np.linspace(5, maxd, int(maxd/5)).astype(int))
    ax[i].set_xticks(xticks)
    ax[i].set_xticklabels(xticks.astype(str))

plt.savefig(os.getcwd()+'\\Figures\\'+fname+'.png')
# plt.show()

