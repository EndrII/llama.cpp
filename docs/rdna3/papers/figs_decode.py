import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.25,"pdf.fonttype":42})
C={"old":"#6b7280","fuse":"#0f766e","new":"#c2410c"}

# Fig A: decode progression (tg128, llama-bench; empty context except the starred point)
st=[("first INT4\n(RTN)",91.9,"old"),("Q6_K head,\nfused rot",132.6,"old"),("fusion\nstart",138.3,"old"),
    ("v2 end\n(412)",173.9,"fuse"),("DPP, gate,\nhnorm\n(348)",186.1,"fuse"),("attn\nprep\n(308)",190.8,"fuse"),
    ("conv\nfold\n(284)",193.2,"fuse"),("z+qkv\n(260)",195.5,"fuse"),("q/k/v\n(244)",197.6,"fuse"),
    ("reg norm,\ngate 4+4\n(244)",199.5,"fuse"),("this\nround\n(236)",199.7,"new")]
fig,ax=plt.subplots(figsize=(7.0,3.3))
b=ax.bar(range(len(st)),[s[1] for s in st],color=[C[s[2]] for s in st],width=.68)
for i,s in enumerate(st):
    if i==len(st)-1: ax.text(i,150,f"{s[1]:.1f}",ha="center",fontsize=7,color="white",rotation=90)
    else: ax.text(i,s[1]+2,f"{s[1]:.1f}",ha="center",fontsize=7)
ax.scatter([len(st)-1],[201.6],marker="*",s=80,c="#111",zorder=5)
ax.annotate("201.6 with a 128-token\ncontext (star)",xy=(len(st)-1,201.6),xytext=(len(st)-3.4,228),fontsize=7,ha="center",arrowprops=dict(arrowstyle="-",lw=.6,color="#444"))
ax.set_xticks(range(len(st))); ax.set_xticklabels([s[0] for s in st],fontsize=6.3); ax.set_ylabel("decode tg128 (tokens/s)"); ax.set_ylim(0,245); ax.grid(axis="x",visible=False)
fig.tight_layout(); fig.savefig("fig/decode_progress.pdf"); plt.close()

# Fig B: where a decode token goes (final build, rocprofv3 kernel trace, microseconds per token)
rows=[("FFN gate+up (32)",1245.4,"w"),("FFN down (32)",645.5,"w"),("LM head Q6_K (1)",663.9,"w"),("GDN qkv+z (24)",600.2,"w"),("GDN ssm_out (24)",373.6,"s"),
      ("rmsnorm+rotate+quant (64)",228.6,"s"),("GDN recurrence (24)",198.5,"s"),("attn qkv (8)",166.1,"w"),("attn_output (8)",103.7,"s"),
      ("attention tile+prep (16)",95.2,"s"),("tail: norm, quantize, get_rows (3)",6.5,"s"),("idle gaps between 236 launches",1020.0,"g")]
col={"w":C["fuse"],"s":C["new"],"g":C["old"]}
fig,ax=plt.subplots(figsize=(6.4,3.3))
y=list(range(len(rows)))[::-1]
ax.barh(y,[r[1] for r in rows],color=[col[r[2]] for r in rows],height=.68)
for yy,r in zip(y,rows): ax.text(r[1]+12,yy,f"{r[1]:,.0f}",va="center",fontsize=7)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows],fontsize=7); ax.set_xlabel("microseconds per token (traced; the untraced token is about 5.0 ms)"); ax.set_xlim(0,1450); ax.grid(axis="y",visible=False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=col["w"],label="weight streaming (near the ceiling)"),Patch(color=col["s"],label="latency-bound kernels"),Patch(color=col["g"],label="launch gaps")],fontsize=7,frameon=False,loc="center right")
fig.tight_layout(); fig.savefig("fig/decode_breakdown.pdf"); plt.close()
