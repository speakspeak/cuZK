import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_csv('energy.log')

fig,axs = plt.subplots(1,3,sharex=True,sharey=True,figsize=(12,4),constrained_layout=True)

for i,(param,ax) in enumerate(zip(data["Param"].unique(),axs)):
  mask = data["Param"] == param
  sns.lineplot(x=data.loc[mask,"Graphics"],y=data.loc[mask,"Time"],color="blue",label="Time",marker="o",errorbar="sd",ax=ax)
  if i==0:
    ax.set_ylabel("Time [ms]")
  ax.set_xlabel("Graphics Frequency [MHz]")
  ax.set_title(f"n={param}")
  ax2=ax.twinx()
  sns.lineplot(x=data.loc[mask,"Graphics"],y=data.loc[mask,"Energy"],color="red",label="Energy",marker="D",ax=ax2,errorbar="sd")
  ax2.set_ylabel("Energy [mJ]")
  ax.get_legend().remove()
  ax2.get_legend().remove()

fig.legend((ax.lines[0], ax2.lines[0]), ('Time [ms]', 'Energy [mJ]'), loc='outside right upper')
fig.suptitle("MSM performance on V100 accelerator (cuZK).")
plt.show()