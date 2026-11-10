import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.25,"pdf.fonttype":42})
C={"q":"#6b7280","a8":"#0f766e","a4":"#c2410c","old":"#7c3aed"}

# Fig 1: Pareto pp512 vs KLD
fig,ax=plt.subplots(figsize=(6.2,4.2))
std={"Q8_0":(0.0023,4775),"Q6_K":(0.0063,4035),"Q5_K_M":(0.0146,4484),"Q4_K_M":(0.0303,4482),"Q4_K_S":(0.0346,4552),"Q4_0":(0.0581,4602),"IQ4_XS":(0.0346,4675),"Q3_K_L":(0.1011,4109),"Q3_K_M":(0.1077,4118),"Q3_K_S":(0.1492,3905)}
off={"Q8_0":(4,-12),"Q6_K":(4,-12),"Q5_K_M":(-6,8),"Q4_K_M":(-30,-12),"Q4_K_S":(5,-12),"Q4_0":(4,5),"IQ4_XS":(-14,8),"Q3_K_L":(-6,-13),"Q3_K_M":(4,5),"Q3_K_S":(4,5)}
ax.scatter([v[0] for v in std.values()],[v[1] for v in std.values()],c=C["q"],s=22,label="stock llama.cpp quants (ROCm 10.1)",zorder=3)
for k,(x,y) in std.items(): ax.annotate(k,(x,y),textcoords="offset points",xytext=off[k],fontsize=6.5,color=C["q"])
sw=[("quality",0.0252,5846),("+qkv,gate,ffn-gate",0.0519,6784),("balanced",0.0713,7257),("+ffn-down",0.0871,7857),("fast",0.0964,8248)]
ax.plot([s[1] for s in sw],[s[2] for s in sw],"-o",c=C["a8"],ms=4,label="rotated INT4 + GPTQ (this work), per-kind A4 sweep",zorder=4)
for n,x,y in sw: ax.annotate(n,(x,y),textcoords="offset points",xytext=(5,-3),fontsize=6.5,color=C["a8"])
ax.scatter([0.197],[8041],marker="s",c=C["old"],s=26,zorder=3,label="earlier file: RTN, A4 everywhere"); ax.annotate("RTN rotq4g4",(0.197,8041),textcoords="offset points",xytext=(-50,-12),fontsize=6.5,color=C["old"])
ax.axhline(8000,ls="--",lw=.8,c="#444"); ax.text(0.006,8060,"8k target",fontsize=7)
ax.set_xlabel("KL divergence vs BF16 (lower is better)"); ax.set_ylabel("prefill pp512 (tokens/s)"); ax.set_xlim(0,0.21); ax.set_ylim(3600,8500)
ax.legend(fontsize=7,loc="upper center",bbox_to_anchor=(0.5,-0.17),frameon=False,ncol=1); fig.tight_layout(); fig.savefig("fig/pareto.pdf",bbox_inches="tight"); plt.close()

# Fig 2: simulation ladder
steps=[("RTN, scale/128\nA4",0.2195),("MSE scale/32\nA4",0.1549),("+GPTQ\nA4",0.1243),("GPTQ\nA8",0.0294),("GPTQ\nA16",0.0289)]
fig,ax=plt.subplots(figsize=(6.2,2.8))
b=ax.bar(range(5),[s[1] for s in steps],color=[C["a4"]]*3+[C["a8"]]*2,width=.62)
for i,s in enumerate(steps): ax.text(i,s[1]+.005,f"{s[1]:.3f}",ha="center",fontsize=7.5)
ax.set_xticks(range(5)); ax.set_xticklabels([s[0] for s in steps],fontsize=7.5); ax.set_ylabel("KLD vs BF16 (simulation)"); ax.set_ylim(0,.26); ax.grid(axis="x",visible=False)
fig.tight_layout(); fig.savefig("fig/ladder.pdf"); plt.close()

# Fig 3: per-kind sensitivity
base=0.0269
kinds=[("k",.0281),("v",.0280),("q",.0291),("attn out",.0325),("z-gate",.0335),("qkv",.0345),("ssm out",.0398),("ffn gate",.0432),("ffn up",.0482),("ffn down",.0505)]
fig,ax=plt.subplots(figsize=(6.2,2.9))
ax.barh([k for k,_ in kinds],[v-base for _,v in kinds],color=C["a4"],height=.65)
for i,(k,v) in enumerate(kinds): ax.text(v-base+.0004,i,f"+{v-base:.4f}",va="center",fontsize=7.5)
ax.set_xlabel("added KLD when only this tensor kind uses 4-bit activations (GPTQ, A8 elsewhere)"); ax.set_xlim(0,.029); ax.grid(axis="y",visible=False)
fig.tight_layout(); fig.savefig("fig/kinds.pdf"); plt.close()

# Fig 4: clip
cl=[(0.6,.1462),(0.7,.0935),(0.85,.0917),(1.0,.1128)]
fig,ax=plt.subplots(figsize=(4.2,2.5))
ax.plot([c for c,_ in cl],[k for _,k in cl],"-o",c=C["a4"],ms=4)
for c,k in cl: ax.annotate(f"{k:.3f}",(c,k),textcoords="offset points",xytext=(0,6),ha="center",fontsize=7)
ax.set_xlabel("activation clip factor"); ax.set_ylabel("KLD (A4 everywhere)"); ax.set_ylim(.08,.16)
fig.tight_layout(); fig.savefig("fig/clip.pdf"); plt.close()
