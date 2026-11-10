import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.25,"pdf.fonttype":42})
G="#6b7280"; T="#0f766e"; O="#c2410c"

# ---- Fig 1: stock Qwen3.5-4B, three panels
names=["Q4_K_M","INT4\nquality","INT4\nfast"]; cols=[G,T,O]
pre=[5477,7140,11012]; dec=[145.2,200.7,200.6]; kld=[0.0278,0.0241,0.0969]
fig,axs=plt.subplots(1,3,figsize=(7.6,3.1))
def panel(ax,v,title,fmt,ratio=None,ylim=None,note=None):
    b=ax.bar(range(3),v,color=cols,width=.66)
    for i,x in enumerate(v):
        lab=fmt.format(x)+(f"\n{ratio[i]}" if ratio else "")
        ax.text(i,x+(ylim*0.015 if ylim else 0),lab,ha="center",va="bottom",fontsize=7.6)
    ax.set_xticks(range(3)); ax.set_xticklabels(names,fontsize=7.2); ax.set_title(title,fontsize=9.5,loc="left"); ax.grid(axis="x",visible=False)
    if ylim: ax.set_ylim(0,ylim)
panel(axs[0],pre,"Prefill, tokens/s\n(higher is better)","{:,.0f}",["1.00×","1.30×","2.01×"],ylim=13200)
panel(axs[1],dec,"Decode, tokens/s\n(higher is better)","{:.0f}",["1.00×","1.38×","1.38×"],ylim=250)
panel(axs[2],kld,"Error vs BF16, KLD\n(lower is better)","{:.3f}",None,ylim=0.115)
fig.tight_layout(); fig.savefig("fig/sum_stock.pdf"); plt.close()

# ---- Fig 2: speed-quality plane, both models
fig,ax=plt.subplots(figsize=(5.6,3.5))
st={"Q4_K_M":(0.0278,5477),"quality":(0.0241,7140),"fast":(0.0969,11012)}
os_={"Q4_K_M":(0.0311,5461),"quality":(0.0250,7094),"fast":(0.1006,10934)}
cc={"Q4_K_M":G,"quality":T,"fast":O}
for k,(x,y) in st.items(): ax.scatter(x,y,s=70,c=cc[k],marker="o",zorder=3,edgecolor="white")
for k,(x,y) in os_.items(): ax.scatter(x,y,s=70,c=cc[k],marker="s",zorder=3,edgecolor="white")
ax.plot([st["quality"][0],st["fast"][0]],[st["quality"][1],st["fast"][1]],c="#999",lw=.8,zorder=1)
ax.plot([os_["quality"][0],os_["fast"][0]],[os_["quality"][1],os_["fast"][1]],c="#999",lw=.8,zorder=1,ls="--")
ax.annotate("Q4_K_M",(0.0295,5477),xytext=(8,-14),textcoords="offset points",fontsize=8,color=G)
ax.annotate("rotated INT4, quality",(0.0245,7117),xytext=(10,-3),textcoords="offset points",fontsize=8,color=T)
ax.annotate("rotated INT4, fast",(0.0988,10970),xytext=(-100,6),textcoords="offset points",fontsize=8,color=O)
ax.scatter([],[],marker="o",c="#444",label="stock Qwen3.5-4B"); ax.scatter([],[],marker="s",c="#444",label="Ossicle-4B (fine-tune)")
ax.set_xlabel("KL divergence from the model's own BF16 (lower is better)"); ax.set_ylabel("prefill pp512 (tokens/s)")
ax.set_xlim(0.01,0.115); ax.set_ylim(4500,12000); ax.legend(fontsize=8,frameon=False,loc="lower right")
fig.tight_layout(); fig.savefig("fig/sum_plane.pdf"); plt.close()

# ---- Fig 3: decode = fewer launches
pts=[(847,138.3),(732,143.3),(708,147.1),(620,155.0),(596,159.6),(524,166.9),(476,172.3),(412,173.9),(348,186.1),(308,190.8),(284,193.2),(260,195.5),(244,197.6),(236,200.7)]
fig,ax=plt.subplots(figsize=(5.6,3.0))
ax.plot([p[0] for p in pts],[p[1] for p in pts],"-o",c=T,ms=4)
ax.axhline(296.7,ls="--",lw=.9,c="#444"); ax.text(840,300,"memory-bandwidth ceiling for this model on this GPU: about 297 tokens/s",fontsize=7.8,va="bottom")
ax.annotate("first INT4 decode path:\n847 launches, 138 tokens/s",(847,138.3),xytext=(70,40),textcoords="offset points",fontsize=8,ha="center",arrowprops=dict(arrowstyle="-",lw=.5))
ax.annotate("final build:\n236 launches, 201 tokens/s",(236,200.7),xytext=(30,-40),textcoords="offset points",fontsize=8,ha="center",arrowprops=dict(arrowstyle="-",lw=.5))
ax.set_xlim(880,200); ax.set_ylim(100,320); ax.set_xlabel("GPU kernel launches per generated token (fewer is better)"); ax.set_ylabel("decode (tokens/s)")
fig.tight_layout(); fig.savefig("fig/sum_decode.pdf"); plt.close()

# ---- decode panel of fig 1 on tg128 from an empty context; new GPU comparison figure
