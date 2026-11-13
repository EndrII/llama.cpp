import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":9.5,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.25,"pdf.fonttype":42})
G="#9ca3af"; O="#c2410c"
# Local LLM Advisor leaderboard, Qwen3.5 4B Q4 (llama.cpp, CUDA); RX 7900 XT: this work (stock Qwen3.5-4B, INT4 fast, tg128 / pp512)
d=[("RTX 5090",16035.6,313.64,0),("RTX 4090",14882.2,220.18,0),("RX 7900 XT\n(INT4 fast, this work)",11012.2,200.6,1),("RTX 5080",10057.3,200.46,0),
   ("RTX 5070 Ti",8889.2,187.62,0),("RTX 5070",7449.3,154.54,0),("RTX 3090",7365.8,179.09,0),("RTX 5060 Ti 16 GB",4775.9,107.64,0)]
fig,axs=plt.subplots(1,2,figsize=(7.6,3.6),sharey=True)
y=list(range(len(d)))[::-1]
for ax,idx,title,fmt,xmax in ((axs[0],1,"Prefill, tokens/s (pp512)","{:,.0f}",19500),(axs[1],2,"Decode, tokens/s (tg128)","{:.1f}",385)):
    ax.barh(y,[r[idx] for r in d],color=[O if r[3] else G for r in d],height=.68)
    for yy,r in zip(y,d): ax.text(r[idx]+xmax*0.012,yy,fmt.format(r[idx]),va="center",fontsize=7.8,fontweight="bold" if r[3] else "normal")
    ax.set_title(title,fontsize=9.5,loc="left"); ax.set_xlim(0,xmax); ax.grid(axis="y",visible=False)
axs[0].set_yticks(y); axs[0].set_yticklabels([r[0] for r in d],fontsize=8)
fig.tight_layout(); fig.savefig("fig/sum_gpus.pdf"); plt.close()
